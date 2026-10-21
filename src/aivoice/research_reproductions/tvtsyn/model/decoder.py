"""TVTSyn decoder: three windowed causal transformer stages with KV caches, conditioned on the per-frame
time-varying timbre by conditional LayerNorm + gated fusion after stages 1 and 2. Causal pitch and energy
predictors sit after stage 1 (ground-truth token-averaged values during training, predictions at
inference), and a causal SEANet decoder upsamples 50 Hz frames to 16 kHz (x8, x5, x4, x2).

Deviation T-D3: the official conditional LayerNorm uses GroupNorm(1, C), which on [B, C, T] normalises
over channels *and time*, i.e. over the whole training segment (not causal; per-chunk at streaming time).
aivoice normalises over channels per frame, which is causal and identical in training and streaming."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ...common.blocks import CausalSEANetDecoder
from ...common.streaming_nn import CausalConv1d, State, Transformer
from ..config import DecoderConfig


def _frame_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return F.layer_norm(x.transpose(1, 2), (x.shape[1],), eps=eps).transpose(1, 2)


class CondLNFusion(nn.Module):
    def __init__(self, c: int, cond: int) -> None:
        super().__init__()
        self.affine = nn.Conv1d(cond, 2 * c, 1)
        self.gate = nn.Conv1d(cond, 1, 1)
        self.proj = nn.Conv1d(c + cond, c, 1)
        for m in (self.affine, self.gate):  # zero init: starts as plain LN (as the official module)
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        g, b = self.affine(cond).chunk(2, 1)
        y = (1 + g) * _frame_norm(x) + b
        return self.proj(torch.cat([y, torch.sigmoid(self.gate(cond)) * _frame_norm(cond)], 1))


class VariancePredictor(nn.Module):
    """conv-ELU-dropout-BN x2 -> 1x1 conv, all causal (BN is a per-frame affine map in eval mode)."""

    def __init__(self, c: int, k: int, dropout: float) -> None:
        super().__init__()
        self.c1, self.c2, self.c3 = CausalConv1d(c, c, k), CausalConv1d(c, c, k), CausalConv1d(c, 1, 1)
        self.b1, self.b2 = nn.BatchNorm1d(c), nn.BatchNorm1d(c)
        self.drop = nn.Dropout(dropout)

    def _mid(self, x, bn):
        return bn(self.drop(F.elu(x)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c3(self._mid(self.c2(self._mid(self.c1(x), self.b1)), self.b2))

    def step(self, x: torch.Tensor, st: State | None) -> tuple[torch.Tensor, State]:
        st = st or {"c1": None, "c2": None}
        h, s1 = self.c1.step(x, st["c1"])
        h, s2 = self.c2.step(self._mid(h, self.b1), st["c2"])
        return self.c3(self._mid(h, self.b2)), {"c1": s1, "c2": s2}


class Decoder(nn.Module):
    def __init__(self, cfg: DecoderConfig, content_dim: int, timbre_dim: int) -> None:
        super().__init__()
        d = cfg.d_model
        self.cfg = cfg
        self.inp = nn.Linear(content_dim, d, bias=False) if content_dim != d else nn.Identity()
        self.b1 = Transformer(d, cfg.b1, cfg.heads, cfg.context)
        self.ln1 = CondLNFusion(d, timbre_dim)
        self.pitch = VariancePredictor(d, cfg.var_kernel, cfg.var_dropout)
        self.energy = VariancePredictor(d, cfg.var_kernel, cfg.var_dropout)
        self.pitch_emb = CausalConv1d(1, d, cfg.var_kernel)
        self.energy_emb = CausalConv1d(1, d, cfg.var_kernel)
        self.b2 = Transformer(d, cfg.b2, cfg.heads, cfg.context)
        self.ln2 = CondLNFusion(d, timbre_dim)
        self.b3 = Transformer(d, cfg.b3, cfg.heads, cfg.context)
        self.wave = CausalSEANetDecoder(d, cfg.seanet_filters, cfg.ratios)
        self.hop = self.wave.hop

    def forward(self, x: torch.Tensor, tv: torch.Tensor, pitch: torch.Tensor | None = None,
                energy: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """x [B, C, T], tv [B, timbre_dim, T]; pitch/energy [B, 1, T] (training targets) -> wav [B, 1, T*320]."""
        z = self.ln1(self.b1(self.inp(x.transpose(1, 2))).transpose(1, 2), tv)
        pp, pe = self.pitch(z), self.energy(z)
        z = z + self.pitch_emb(pitch if pitch is not None else pp) + self.energy_emb(energy if energy is not None else pe)
        z = self.ln2(self.b2(z.transpose(1, 2)).transpose(1, 2), tv)
        z = self.b3(z.transpose(1, 2)).transpose(1, 2)
        return self.wave(z), {"pitch": pp, "energy": pe}

    def step(self, x: torch.Tensor, tv: torch.Tensor, st: State | None) -> tuple[torch.Tensor, State]:
        st = st or {k: None for k in ("b1", "p", "e", "pe", "ee", "b2", "b3", "wave")}
        h, s_b1 = self.b1.step(self.inp(x.transpose(1, 2)), st["b1"])
        z = self.ln1(h.transpose(1, 2), tv)
        pp, s_p = self.pitch.step(z, st["p"])
        pe, s_e = self.energy.step(z, st["e"])
        a, s_pe = self.pitch_emb.step(pp, st["pe"])
        b, s_ee = self.energy_emb.step(pe, st["ee"])
        z = z + a + b
        h, s_b2 = self.b2.step(z.transpose(1, 2), st["b2"])
        z = self.ln2(h.transpose(1, 2), tv)
        h, s_b3 = self.b3.step(z.transpose(1, 2), st["b3"])
        y, s_w = self.wave.step(h.transpose(1, 2), st["wave"])
        return y, {"b1": s_b1, "p": s_p, "e": s_e, "pe": s_pe, "ee": s_ee, "b2": s_b2, "b3": s_b3, "wave": s_w}
