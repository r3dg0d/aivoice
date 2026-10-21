"""The Zero-VC inference network: content encoder -> (timbre-conditioned) causal HiFi-GAN decoder.

Training-only parts (anonymiser, discriminators, distillation heads/teachers) are not referenced here.
"""

from __future__ import annotations

import torch
from torch import nn

from ...common.streaming_nn import State
from ..config import ZeroVCConfig
from .decoder import build_decoder
from .encoder import StreamingContentEncoder
from .timbre import TimbreEncoder


class ZeroVC(nn.Module):
    def __init__(self, cfg: ZeroVCConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.content = StreamingContentEncoder(cfg.content)
        self.timbre = TimbreEncoder(cfg.timbre)
        self.decoder = build_decoder(cfg.decoder, cfg.content.out_dim, cfg.timbre.embed_dim)
        if self.content.hop != self.decoder.hop:
            raise ValueError(f"encoder hop {self.content.hop} != decoder hop {self.decoder.hop}")
        self.hop = self.content.hop

    def forward(self, wav: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Offline (training) path: wav [B, T], timbre vector g [B, E] -> [B, 1, T]."""
        return self.decoder(self.content.features(wav), g)

    def step(self, wav: torch.Tensor, g: torch.Tensor, state: State | None) -> tuple[torch.Tensor, State]:
        state = state or {"content": None, "decoder": None}
        h, sc = self.content.step(wav, state["content"])
        y, sd = self.decoder.step(h, state["decoder"], g)
        return y, {"content": sc, "decoder": sd}
