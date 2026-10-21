"""Losses and discriminators shared by the reproductions (aivoice implementations of standard designs).

* HiFi-GAN multi-period (MPD) and multi-scale (MSD) discriminators, LSGAN objective, feature matching
  (Kong et al. 2020) - used by Zero-VC (HiFi-GAN-like decoder) and TVTSyn.
* Multi-resolution log-mel L1 and multi-scale spectral loss (DDSP, Engel et al. 2020) - used by RT-VC.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm, weight_norm

from .dsp import MelSpectrogram, stft_mag


class PeriodDiscriminator(nn.Module):
    def __init__(self, period: int, ch: int = 32, max_ch: int = 1024) -> None:
        super().__init__()
        self.period = period
        chs = [1, ch, ch * 4, ch * 16, max_ch, max_ch]
        self.convs = nn.ModuleList(
            weight_norm(nn.Conv2d(chs[i], chs[i + 1], (5, 1), (3 if i < 4 else 1, 1), padding=(2, 0)))
            for i in range(5))
        self.post = weight_norm(nn.Conv2d(max_ch, 1, (3, 1), padding=(1, 0)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        b, c, t = x.shape
        if t % self.period:
            x = F.pad(x, (0, self.period - t % self.period), mode="reflect")
        x = x.view(b, c, -1, self.period)
        feats = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            feats.append(x)
        x = self.post(x)
        feats.append(x)
        return x.flatten(1), feats


class ScaleDiscriminator(nn.Module):
    """HiFi-GAN/MelGAN scale discriminator; ``ch``=16, ``max_ch``=1024 are the published widths."""

    def __init__(self, use_spectral_norm: bool = False, ch: int = 16, max_ch: int = 1024) -> None:
        super().__init__()
        norm = spectral_norm if use_spectral_norm else weight_norm
        w = [1, ch, min(ch * 4, max_ch), min(ch * 16, max_ch), max_ch, max_ch, max_ch]
        ks, ss, gs = (15, 41, 41, 41, 41, 5), (1, 4, 4, 4, 4, 1), (1, 4, 16, 64, 256, 1)
        layers = []
        for i in range(6):
            g = gs[i]
            while g > 1 and (w[i] % g or w[i + 1] % g):
                g //= 2
            layers.append(norm(nn.Conv1d(w[i], w[i + 1], ks[i], ss[i], groups=g, padding=ks[i] // 2)))
        self.convs = nn.ModuleList(layers)
        self.post = norm(nn.Conv1d(max_ch, 1, 3, padding=1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        feats = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            feats.append(x)
        x = self.post(x)
        feats.append(x)
        return x.flatten(1), feats


class HiFiGANDiscriminator(nn.Module):
    """MPD (periods 2,3,5,7,11) + MSD (3 scales). ``ch_mult`` < 1 shrinks it for tiny-overfit tests."""

    def __init__(self, periods: tuple[int, ...] = (2, 3, 5, 7, 11), msd_scales: int = 3, ch_mult: float = 1.0) -> None:
        super().__init__()
        c = max(4, int(32 * ch_mult))
        m = max(16, int(1024 * ch_mult))
        self.mpd = nn.ModuleList(PeriodDiscriminator(p, c, m) for p in periods)
        self.msd = nn.ModuleList(ScaleDiscriminator(i == 0, max(4, int(16 * ch_mult)), m) for i in range(msd_scales))
        self.pool = nn.AvgPool1d(4, 2, padding=2)

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        outs, feats = [], []
        for d in self.mpd:
            o, f = d(x)
            outs.append(o)
            feats.append(f)
        y = x
        for i, d in enumerate(self.msd):
            if i:
                y = self.pool(y)
            o, f = d(y)
            outs.append(o)
            feats.append(f)
        return outs, feats


def lsgan_d_loss(real: list[torch.Tensor], fake: list[torch.Tensor]) -> torch.Tensor:
    return sum(torch.mean((1 - r) ** 2) + torch.mean(f ** 2) for r, f in zip(real, fake)) / len(real)


def lsgan_g_loss(fake: list[torch.Tensor]) -> torch.Tensor:
    return sum(torch.mean((1 - f) ** 2) for f in fake) / len(fake)


def feature_matching_loss(real: list[list[torch.Tensor]], fake: list[list[torch.Tensor]]) -> torch.Tensor:
    loss, n = 0.0, 0
    for fr, ff in zip(real, fake):
        for a, b in zip(fr, ff):
            loss = loss + torch.mean(torch.abs(a.detach() - b))
            n += 1
    return loss / max(n, 1)


class MelL1(nn.Module):
    """Log-mel L1 (HiFi-GAN reconstruction loss)."""

    def __init__(self, sr: int, n_fft: int = 1024, hop: int = 256, n_mels: int = 80, fmax: float | None = None) -> None:
        super().__init__()
        self.mel = MelSpectrogram(sr, n_fft, hop, n_mels=n_mels, fmax=fmax)

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(self.mel(y_hat), self.mel(y))


class MultiResolutionMelL1(nn.Module):
    def __init__(self, sr: int, resolutions: tuple[tuple[int, int, int], ...] = ((512, 128, 64), (1024, 256, 80),
                                                                                 (2048, 512, 128))) -> None:
        super().__init__()
        self.mels = nn.ModuleList(MelSpectrogram(sr, n, h, n_mels=m) for n, h, m in resolutions)

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return sum(F.l1_loss(m(y_hat), m(y)) for m in self.mels) / len(self.mels)


class MultiScaleSpectralLoss(nn.Module):
    """DDSP loss: sum over FFT sizes of |S - S_hat|_1 + |log S - log S_hat|_1 (75% overlap)."""

    def __init__(self, fft_sizes: tuple[int, ...] = (2048, 1024, 512, 256, 128, 64), alpha: float = 1.0) -> None:
        super().__init__()
        self.fft_sizes, self.alpha = fft_sizes, alpha

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        for n in self.fft_sizes:
            a, b = stft_mag(y_hat, n, n // 4), stft_mag(y, n, n // 4)
            loss = loss + F.l1_loss(a, b) + self.alpha * F.l1_loss(torch.log(a), torch.log(b))
        return loss / len(self.fft_sizes)


__all__ = ["HiFiGANDiscriminator", "MelL1", "MultiResolutionMelL1", "MultiScaleSpectralLoss", "PeriodDiscriminator",
           "ScaleDiscriminator", "feature_matching_loss", "lsgan_d_loss", "lsgan_g_loss"]
