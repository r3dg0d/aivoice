"""RT-VC articulatory encoder: causal pitch tracker (pitch / periodicity / loudness) and causal EMA
inverter (12 articulatory channels), both at 200 Hz. Layout follows the official model classes."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ...common.streaming_nn import State
from ..config import EMAConfig, PitchConfig
from .convnet import CausalConvNet, mlp


def f2p(f, base: float):
    return 12 * np.log2(f / base) if isinstance(f, np.ndarray) else 12 * torch.log2(f / base)


class PitchTracker(nn.Module):
    def __init__(self, cfg: PitchConfig, n_mels: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.conv = CausalConvNet(n_mels, cfg.hidden_dim, cfg.hidden_dim, cfg.kernel_size)
        self.pitch_head = mlp(cfg.hidden_dim, cfg.hidden_dim, cfg.nbins, cfg.forward_expansion, cfg.n_midlayers)
        self.per_head = mlp(cfg.hidden_dim, cfg.hidden_dim, 1, cfg.forward_expansion, cfg.n_midlayers)
        self.loud_head = mlp(cfg.hidden_dim, cfg.hidden_dim, 1, cfg.forward_expansion, cfg.n_midlayers)
        pmin, pmax = 12 * np.log2(cfg.fmin / cfg.f_base), 12 * np.log2(cfg.fmax / cfg.f_base)
        self.pmin, self.dp = float(pmin), float((pmax - pmin) / (cfg.nbins - 1))

    def _heads(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        h = h.transpose(1, 2)
        return {"logits": self.pitch_head(h), "periodicity": torch.sigmoid(self.per_head(h)).transpose(1, 2),
                "loudness": self.cfg.loudness_scale * torch.sigmoid(self.loud_head(h)).transpose(1, 2)}

    def forward(self, logmel: torch.Tensor) -> dict[str, torch.Tensor]:
        return self._heads(self.conv(logmel))

    def step(self, logmel: torch.Tensor, state: State | None) -> tuple[dict[str, torch.Tensor], State]:
        h, st = self.conv.step(logmel, state)
        return self._heads(h), st

    def bin_of(self, f0: torch.Tensor) -> torch.Tensor:
        f = f0.clamp(self.cfg.fmin, self.cfg.fmax)
        return (12 * torch.log2(f / self.cfg.f_base) - self.pmin) / self.dp

    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        """Official decoding: sigmoid probs, weighted bin average over 9 bins from argmax-4 -> Hz. [B,t,n] -> [B,1,t]."""
        p = torch.sigmoid(logits)
        start = (p.argmax(-1) - 4).clamp(min=0)
        idx = (start[..., None] + torch.arange(9, device=p.device)).clamp(max=self.cfg.nbins - 1)
        w = torch.gather(p, -1, idx)
        b = (idx * w).sum(-1) / w.sum(-1).clamp_min(1e-8)
        return (self.cfg.f_base * 2 ** ((self.pmin + b * self.dp) / 12))[:, None]

    def targets(self, f0: torch.Tensor) -> torch.Tensor:
        """Gaussian-blurred bin targets for BCE (assumption R-P1). f0 [B, t] Hz (0 = unvoiced) -> [B, t, nbins]."""
        b = self.bin_of(f0.clamp_min(self.cfg.fmin))
        grid = torch.arange(self.cfg.nbins, device=f0.device, dtype=f0.dtype)
        t = torch.exp(-0.5 * ((grid - b[..., None]) / self.cfg.label_sigma_bins) ** 2)
        return t * (f0 > 0)[..., None]


class EMAInverter(nn.Module):
    def __init__(self, cfg: EMAConfig, n_mfcc: int) -> None:
        super().__init__()
        self.conv = CausalConvNet(n_mfcc, cfg.hidden_dim, cfg.hidden_dim, cfg.kernel_size)
        self.out = nn.Sequential(nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.ReLU(),
                                 nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.ReLU(),
                                 nn.Linear(cfg.hidden_dim, cfg.channels))

    def forward(self, mfcc: torch.Tensor) -> torch.Tensor:
        return self.out(self.conv(mfcc).transpose(1, 2)).transpose(1, 2)  # [B, 12, t]

    def step(self, mfcc: torch.Tensor, state: State | None) -> tuple[torch.Tensor, State]:
        h, st = self.conv.step(mfcc, state)
        return self.out(h.transpose(1, 2)).transpose(1, 2), st
