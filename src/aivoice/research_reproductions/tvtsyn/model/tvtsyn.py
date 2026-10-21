"""TVTSyn inference graph: content encoder (frozen after stage 1) -> time-varying timbre -> decoder."""

from __future__ import annotations

import torch
from torch import nn

from ...common.streaming_nn import State
from ..config import TVTSynConfig
from .decoder import Decoder
from .encoder import ContentEncoder
from .timbre import TimeVaryingTimbre


class TVTSyn(nn.Module):
    def __init__(self, cfg: TVTSynConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.content = ContentEncoder(cfg.content)
        self.timbre = TimeVaryingTimbre(cfg.timbre, cfg.content.dim)
        self.decoder = Decoder(cfg.decoder, cfg.content.dim, cfg.timbre.d_out)
        self.hop = self.content.hop

    @property
    def lookahead_frames(self) -> int:
        return self.content.lookahead_frames

    def forward(self, wav: torch.Tensor, g: torch.Tensor, pitch=None, energy=None):
        z = self.content(wav)["z_q"]
        tv, _ = self.timbre(z, g)
        return self.decoder(z, tv, pitch, energy)

    def step(self, wav: torch.Tensor, g: torch.Tensor, st: State | None) -> tuple[torch.Tensor, State]:
        st = st or {"content": None, "timbre": None, "decoder": None}
        z, s_c = self.content.step(wav, st["content"])
        if z.shape[-1] == 0:
            return wav.new_zeros(wav.shape[0], 1, 0), {**st, "content": s_c}
        tv, s_t = self.timbre.step(z, g, st["timbre"])
        y, s_d = self.decoder.step(z, tv, st["decoder"])
        return y, {"content": s_c, "timbre": s_t, "decoder": s_d}
