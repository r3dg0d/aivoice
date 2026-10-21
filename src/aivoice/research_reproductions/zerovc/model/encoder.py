"""Zero-VC streaming content encoder (paper: "a pretrained streaming encoder module (e.g., a distilled
streaming w2v-bert-2.0)"). Strictly causal: 20 ms frames, right_context = 0.

Variant A (closest to the paper): the causal student is distilled from w2v-BERT 2.0 hidden states
(L1 + cosine regression, DistilHuBERT-style). Variant B (aivoice practical replacement, NOT the paper):
the same student trained with cross-entropy on HuBERT-base layer-9 k-means pseudo-labels, a cheaper
teacher whose discrete targets carry less speaker information. Both expose the same ``features``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ...common.blocks import CausalSEANetEncoder, ChannelsTransformer
from ...common.streaming_nn import State
from ..config import ContentConfig


class StreamingContentEncoder(nn.Module):
    def __init__(self, cfg: ContentConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.cnn = CausalSEANetEncoder(cfg.dim, cfg.n_filters, cfg.strides)
        self.tf = ChannelsTransformer(cfg.dim, cfg.dim, cfg.depth, cfg.heads, cfg.context, cfg.right_context,
                                      dim_out=cfg.out_dim)
        self.hop = self.cnn.hop
        # training-only heads (dropped from inference checkpoints' use; kept in state for resume)
        if cfg.variant == "A":
            self.head = nn.Linear(cfg.out_dim, cfg.teacher_dim)
        elif cfg.variant == "B":
            self.head = nn.Linear(cfg.out_dim, cfg.km_codes)
        else:
            raise ValueError(f"content variant must be A or B, got {cfg.variant!r}")

    @property
    def lookahead_frames(self) -> int:
        return self.tf.lookahead_frames

    def features(self, wav: torch.Tensor) -> torch.Tensor:
        """wav [B, T] -> content [B, out_dim, T/hop]."""
        return self.tf(self.cnn(wav[:, None]))

    def step(self, wav: torch.Tensor, state: State | None) -> tuple[torch.Tensor, State]:
        state = state or {"cnn": None, "tf": None}
        h, s_cnn = self.cnn.step(wav[:, None], state["cnn"])
        h, s_tf = self.tf.step(h, state["tf"])
        return h, {"cnn": s_cnn, "tf": s_tf}

    def distill_loss(self, wav: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        """Variant A: target [B, T', teacher_dim] teacher hidden states. Variant B: target [B, T'] code ids."""
        h = self.head(self.features(wav).transpose(1, 2))
        n = min(h.shape[1], target.shape[1])
        h, target = h[:, :n], target[:, :n]
        if self.cfg.variant == "A":
            l1 = F.l1_loss(h, target)
            cos = 1 - F.cosine_similarity(h, target, dim=-1).mean()
            return {"loss": l1 + cos, "l1": l1.detach(), "cos": cos.detach()}  # DistilHuBERT: L1 + cosine
        ce = F.cross_entropy(h.reshape(-1, h.shape[-1]), target.reshape(-1).long())
        acc = (h.argmax(-1) == target).float().mean()
        return {"loss": ce, "ce": ce.detach(), "acc": acc.detach()}
