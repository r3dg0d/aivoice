"""Torch DSP for training targets and losses (no torchaudio dependency).

``causal=True`` STFTs pad only on the left, so frame t covers samples [t*hop - (n_fft-hop), t*hop + hop):
a streaming front-end built from them has zero lookahead. ``causal=False`` is the usual centered STFT
(n_fft/2 lookahead), kept for losses, where lookahead does not matter.
"""

from __future__ import annotations

import functools
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


@functools.lru_cache(maxsize=32)
def _mel_basis_np(sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float | None) -> np.ndarray:
    import librosa

    return librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax).astype(np.float32)


def mel_basis(sr: int, n_fft: int, n_mels: int, fmin: float = 0.0, fmax: float | None = None) -> torch.Tensor:
    return torch.from_numpy(_mel_basis_np(sr, n_fft, n_mels, fmin, fmax))


def stft_mag(x: torch.Tensor, n_fft: int, hop: int, win: int | None = None, causal: bool = False,
             power: float = 1.0) -> torch.Tensor:
    """x [B, T] -> magnitude [B, n_fft//2+1, frames] (always computed in float32).

    cuFFT has no bf16 kernels, so under bf16 autocast torch.stft raised and the RT-VC vocoder stage could not
    train; half-precision FFTs would also cost loss accuracy. The STFT runs in fp32 with autocast disabled."""
    win = win or n_fft
    if x.dtype != torch.float32 or torch.is_autocast_enabled():
        with torch.autocast(device_type=x.device.type, enabled=False):
            return stft_mag(x.float(), n_fft, hop, win, causal=causal, power=power)
    window = torch.hann_window(win, device=x.device, dtype=x.dtype)
    if causal:
        x = F.pad(x, (n_fft - hop, 0))
        spec = torch.stft(x, n_fft, hop, win, window, center=False, return_complex=True)
    else:
        spec = torch.stft(x, n_fft, hop, win, window, center=True, pad_mode="reflect", return_complex=True)
    mag = spec.abs().clamp_min(1e-9)
    return mag if power == 1.0 else mag ** power


class MelSpectrogram(nn.Module):
    """Log-mel spectrogram (natural log, clamped), HiFi-GAN convention."""

    def __init__(self, sr: int = 16000, n_fft: int = 1024, hop: int = 256, win: int | None = None, n_mels: int = 80,
                 fmin: float = 0.0, fmax: float | None = None, causal: bool = False) -> None:
        super().__init__()
        self.n_fft, self.hop, self.win, self.causal = n_fft, hop, win or n_fft, causal
        self.register_buffer("basis", mel_basis(sr, n_fft, n_mels, fmin, fmax), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mag = stft_mag(x, self.n_fft, self.hop, self.win, self.causal)
        return torch.log(torch.clamp(self.basis @ mag, min=1e-5))


class MFCC(nn.Module):
    """MFCCs from a log-mel spectrogram via an orthonormal DCT-II (RT-VC uses MFCC inputs)."""

    def __init__(self, n_mfcc: int = 30, **mel_kw) -> None:
        super().__init__()
        self.mel = MelSpectrogram(**mel_kw)
        n_mels = self.mel.basis.shape[0]
        n = torch.arange(n_mels, dtype=torch.float32)
        k = torch.arange(n_mfcc, dtype=torch.float32)[:, None]
        dct = torch.cos(math.pi / n_mels * (n + 0.5) * k) * math.sqrt(2.0 / n_mels)
        dct[0] /= math.sqrt(2.0)
        self.register_buffer("dct", dct, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dct @ self.mel(x)


def a_weighting_db(freqs: np.ndarray) -> np.ndarray:
    f2 = np.maximum(freqs, 1e-6) ** 2
    ra = (12194.0 ** 2 * f2 ** 2) / ((f2 + 20.6 ** 2) * np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2)) * (f2 + 12194.0 ** 2))
    return 20 * np.log10(np.maximum(ra, 1e-12)) + 2.0


class Loudness(nn.Module):
    """A-weighted loudness in dB per frame (DDSP convention: mean of weighted power spectrum)."""

    def __init__(self, sr: int = 16000, n_fft: int = 1024, hop: int = 160, causal: bool = True) -> None:
        super().__init__()
        self.n_fft, self.hop, self.causal = n_fft, hop, causal
        freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
        self.register_buffer("weight", torch.from_numpy(a_weighting_db(freqs).astype(np.float32))[:, None],
                             persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mag = stft_mag(x, self.n_fft, self.hop, causal=self.causal)
        db = 20 * torch.log10(mag) + self.weight
        return 10 * torch.log10(torch.mean(10 ** (db / 10), dim=1).clamp_min(1e-10))


def f0_yin(x: np.ndarray, sr: int, hop: int, fmin: float = 50.0, fmax: float = 800.0,
           frame: int | None = None, threshold: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    """F0 (Hz, 0 = unvoiced) and periodicity in [0,1] per hop, causal frames (window ends at the hop).

    Uses aivoice's YIN (cumulative mean normalised difference); periodicity = 1 - min CMND, the usual
    aperiodicity proxy. Used as training targets where the paper uses an external pitch tracker; this
    is an assumption recorded in each SPEC.md that uses it.
    """
    frame = frame or int(sr * 0.04)
    tau_min, tau_max = int(sr / fmax), min(int(sr / fmin), frame - 1)
    x = np.concatenate([np.zeros(frame - hop, np.float32), x.astype(np.float32)])
    n = max(0, (len(x) - frame) // hop + 1)
    f0 = np.zeros(n, np.float32)
    per = np.zeros(n, np.float32)
    if n == 0:
        return f0, per
    w = frame - tau_max
    frames = np.lib.stride_tricks.sliding_window_view(x, frame)[::hop][:n].astype(np.float64)  # [n, frame]
    ref = frames[:, :w]
    lag = np.lib.stride_tricks.sliding_window_view(frames, w, axis=1)[:, : tau_max + 1]  # [n, tau, w]
    d = (ref ** 2).sum(1)[:, None] + (lag ** 2).sum(2) - 2 * np.einsum("nw,ntw->nt", ref, lag)
    d = np.maximum(d, 0.0)
    cm = np.cumsum(d[:, 1:], axis=1)
    cmnd = np.ones_like(d)
    cmnd[:, 1:] = d[:, 1:] * np.arange(1, tau_max + 1) / np.maximum(cm, 1e-12)
    energy = (ref ** 2).mean(1)
    for i in range(n):
        band = cmnd[i, tau_min:tau_max + 1]
        j = int(np.argmin(band))
        below = np.nonzero(band < threshold)[0]
        if len(below):
            j = int(below[0])
            while j + 1 < len(band) and band[j + 1] < band[j]:
                j += 1
        per[i] = float(np.clip(1.0 - band[j], 0.0, 1.0))
        if band[j] < threshold and energy[i] > 1e-7:
            t = j + tau_min
            if 1 <= t < tau_max:  # parabolic interpolation
                a, b, c = cmnd[i, t - 1], cmnd[i, t], cmnd[i, t + 1]
                den = a - 2 * b + c
                t = t + (0.5 * (a - c) / den if abs(den) > 1e-12 else 0.0)
            f0[i] = sr / t
    return f0, per


def hz_to_cents(f0: torch.Tensor, ref: float = 10.0) -> torch.Tensor:
    return 1200.0 * torch.log2(f0.clamp_min(1e-5) / ref)


__all__ = ["MFCC", "Loudness", "MelSpectrogram", "a_weighting_db", "f0_yin", "hz_to_cents", "mel_basis", "stft_mag"]
