"""Causal front-end: log-mel (pitch tracker input) and MFCC (EMA inverter input) at 200 Hz.

Deviation R-F1 (aivoice): the official helpers use a centred STFT, i.e. 512 samples (32 ms) of
lookahead. Here frame t covers samples [80(t+1) - 1024, 80(t+1)), so a frame is complete as soon as its
last hop has arrived. Models trained with this front-end are therefore genuinely zero-lookahead.
Deviation R-F3: log-mel is used as the pitch-tracker input (the official code feeds linear mel power).
"""

from __future__ import annotations

import functools
import math

import numpy as np
import torch
from torch import nn

from ...common.streaming_nn import State


@functools.lru_cache(maxsize=8)
def _htk_mel(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    import librosa

    return librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, htk=True, norm=None).astype(np.float32)


class CausalMelMFCC(nn.Module):
    def __init__(self, sr: int = 16000, n_fft: int = 1024, hop: int = 80, n_mels: int = 128, n_mfcc: int = 24) -> None:
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        self.register_buffer("basis", torch.from_numpy(_htk_mel(sr, n_fft, n_mels)), persistent=False)
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        n = torch.arange(n_mels, dtype=torch.float32)
        k = torch.arange(n_mfcc, dtype=torch.float32)[:, None]
        dct = torch.cos(math.pi / n_mels * (n + 0.5) * k) * math.sqrt(2.0 / n_mels)
        dct[0] /= math.sqrt(2.0)
        self.register_buffer("dct", dct, persistent=False)

    def _stft(self, x: torch.Tensor) -> torch.Tensor:
        # fp32 STFT with autocast off: cuFFT has no bf16 kernels (bf16 training crashed here)
        with torch.autocast(device_type=x.device.type, enabled=False):
            return torch.stft(x.float(), self.n_fft, self.hop, self.n_fft, self.window.float(), center=False, return_complex=True)

    def _frames(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        spec = self._stft(x)
        power = spec.abs() ** 2                                 # [B, F, t]
        logmel = torch.log(self.basis @ power + 1e-6)
        mfcc = self.dct @ (10.0 * logmel / math.log(10.0))      # dB-scaled log-mel, ortho DCT-II
        return logmel, mfcc

    def forward(self, wav: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """wav [B, T] -> logmel [B, n_mels, T/hop], mfcc [B, n_mfcc, T/hop]."""
        x = nn.functional.pad(wav, (self.n_fft - self.hop, 0))
        return self._frames(x)

    def step(self, wav: torch.Tensor, state: State | None) -> tuple[tuple[torch.Tensor, torch.Tensor], State]:
        buf = wav.new_zeros(wav.shape[0], self.n_fft - self.hop) if state is None else state["buf"]
        buf = torch.cat([buf, wav], dim=-1)
        n = (buf.shape[-1] - self.n_fft) // self.hop + 1 if buf.shape[-1] >= self.n_fft else 0
        if n <= 0:
            c = self.basis.shape[0]
            return (wav.new_zeros(wav.shape[0], c, 0), wav.new_zeros(wav.shape[0], self.dct.shape[0], 0)), {"buf": buf}
        out = self._frames(buf[:, : (n - 1) * self.hop + self.n_fft])
        return out, {"buf": buf[:, n * self.hop:]}

    def loudness(self, wav: torch.Tensor) -> torch.Tensor:
        """Loudness target as the official helper defines it: mean STFT magnitude over frequency (power=1)."""
        x = nn.functional.pad(wav, (self.n_fft - self.hop, 0))
        return self._stft(x).abs().mean(1, keepdim=True)  # [B, 1, t]
