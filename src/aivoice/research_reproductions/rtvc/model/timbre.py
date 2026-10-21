"""RT-VC speaker encoder (enrollment only, offline): WavLM-large CNN feature extractor (frozen) ->
dilated conv encoder -> periodicity-weighted mean (+ std) pooling -> MLP -> speaker embedding.
Layout follows the official SpeakerEncoder; widths of the conv encoder are assumptions (SPEC R-S2)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ...common.teachers import teacher_dir
from ..config import SpeakerConfig
from .convnet import mlp


class _ResBlock(nn.Module):  # non-causal: the speaker encoder sees the whole reference
    def __init__(self, ch: int, k: int, d: int) -> None:
        super().__init__()
        self.c1 = nn.Conv1d(ch, ch, k, padding=(k - 1) // 2 * d, dilation=d)
        self.b1 = nn.BatchNorm1d(ch)
        self.c2 = nn.Conv1d(ch, ch, k, padding=(k - 1) // 2 * d, dilation=d)
        self.b2 = nn.BatchNorm1d(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.b2(self.c2(F.relu(self.b1(self.c1(x))))))


class WavLMCNN(nn.Module):
    """The convolutional feature extractor of WavLM-large (512 ch, 50 Hz), loaded from a local snapshot."""

    def __init__(self, directory: str | None = None, standin: bool = False) -> None:
        super().__init__()
        if standin:
            from ...common.blocks import CausalSEANetEncoder

            self.fe = CausalSEANetEncoder(512, 8, (2, 4, 5, 8))
            self.standin = True
        else:
            import transformers

            d = teacher_dir("wavlm-large", directory)
            self.fe = transformers.WavLMModel.from_pretrained(str(d)).feature_extractor
            self.standin = False

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        return self.fe(wav[:, None]) if self.standin else self.fe(wav)


class SpeakerEncoder(nn.Module):
    def __init__(self, cfg: SpeakerConfig, wavlm_dir: str | None = None, standin: bool = False) -> None:
        super().__init__()
        self.cfg = cfg
        self.cnn = WavLMCNN(wavlm_dir, standin)
        if cfg.freeze_wavlm_cnn:
            self.cnn.requires_grad_(False)
        self.inp = nn.Conv1d(512, cfg.hidden_dim, 3, padding=1)
        self.stack = nn.Sequential(*[_ResBlock(cfg.hidden_dim, 3, d) for _ in range(cfg.nstacks) for d in cfg.dilations])
        self.outc = nn.Conv1d(cfg.hidden_dim, cfg.hidden_dim, 3, padding=1)
        self.head = mlp(cfg.hidden_dim * (2 if cfg.use_std else 1), cfg.spkr_emb_dim, cfg.spkr_emb_dim,
                        cfg.forward_expansion, cfg.n_midlayers)

    def forward(self, wav: torch.Tensor, periodicity: torch.Tensor) -> torch.Tensor:
        """wav [B, T] 16 kHz; periodicity [B, 1, t@200Hz] -> [B, spkr_emb_dim]."""
        with torch.set_grad_enabled(self.training and not self.cfg.freeze_wavlm_cnn):
            h = self.cnn(wav)
        h = self.outc(self.stack(self.inp(h)))       # [B, C, t@50Hz]
        p = periodicity[:, :, ::4]                     # 200 Hz -> 50 Hz (official: [:, :, ::4])
        n = min(p.shape[-1], h.shape[-1])
        h, p = h[..., :n], p[..., :n]
        w = p / (p.sum(-1, keepdim=True) + 1e-4)
        mu = (w * h).sum(-1)
        if self.cfg.use_std:
            sd = torch.sqrt(((w * h ** 2).sum(-1) - mu ** 2).clamp_min(1e-6))
            mu = torch.cat([mu, sd], -1)
        return self.head(mu)
