"""Zero-VC timbre encoder (paper Eq. 1): WavLM-large layer-7 hidden states pooled into one timbre vector
by learnable attention pooling. WavLM is frozen; only the pooling is trained. At inference the vector
is computed once per target voice and cached (see streaming/inference.py)."""

from __future__ import annotations

import torch
from torch import nn

from ...common.blocks import AttentionPooling
from ..config import TimbreConfig


class TimbreEncoder(nn.Module):
    def __init__(self, cfg: TimbreConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.pool = AttentionPooling(cfg.ssl_dim, cfg.embed_dim, cfg.pool_hidden)

    def forward(self, ssl_feats: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """ssl_feats [B, T, ssl_dim] (WavLM-large hidden_states[7]) -> [B, embed_dim]."""
        return self.pool(ssl_feats, mask)
