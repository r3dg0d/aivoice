"""TVTSyn streaming content encoder ("PASR with quantiser"): causal SEANet CNN (total stride 320) ->
8-layer windowed MHSA -> factorised VQ (512 -> 8-d, 4096 codes, L2-normalised, -> 512) -> classifier over
HuBERT-base layer-9 k-means-200 pseudo-labels (training only)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.parametrizations import weight_norm

from ...common.blocks import CausalSEANetEncoder, ChannelsTransformer
from ...common.streaming_nn import State
from ..config import ContentConfig


class FactorizedVQ(nn.Module):
    """Factorised VQ (low-dim lookup, L2-normalised = cosine), straight-through, commitment + codebook loss."""

    def __init__(self, dim: int, size: int, cdim: int, commitment: float, codebook_weight: float,
                 l2: bool = True) -> None:
        super().__init__()
        self.inp = weight_norm(nn.Conv1d(dim, cdim, 1))
        self.out = weight_norm(nn.Conv1d(cdim, dim, 1))
        self.codebook = nn.Embedding(size, cdim)
        self.commitment, self.cb_w, self.l2 = commitment, codebook_weight, l2

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """z [B, dim, T] -> dict(z_q [B, dim, T], indices [B, T], commit, codebook)."""
        ze = self.inp(z)                                         # [B, cdim, T]
        e = ze.transpose(1, 2)
        cb = self.codebook.weight
        # as in the official FactorizedVectorQuantize: L2 normalisation is used ONLY for the nearest-code lookup
        # (cosine distance); losses and the straight-through output use the raw projection and raw code vector.
        # Normalising them too (the earlier port) collapsed the content encoder to ~3 codes on real data.
        en, cbn = (F.normalize(e, dim=-1), F.normalize(cb, dim=-1)) if self.l2 else (e, cb)
        idx = (en @ cbn.t()).argmax(-1) if self.l2 else torch.cdist(e, cb[None]).argmin(-1)
        zq = self.codebook(idx)
        commit = F.mse_loss(e, zq.detach()) * self.commitment
        cbl = F.mse_loss(zq, e.detach()) * self.cb_w
        zq = e + (zq - e).detach()                               # straight-through
        return {"z_q": self.out(zq.transpose(1, 2)), "indices": idx, "commit": commit, "codebook": cbl}


class ContentEncoder(nn.Module):
    def __init__(self, cfg: ContentConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.cnn = CausalSEANetEncoder(cfg.dim, cfg.n_filters, cfg.strides)
        # no output LayerNorm, as in the official ProjectedTransformer: with it the encoder never left the
        # label-marginal plateau at the paper's lr 5e-4 (CE stuck at ~5.1 on real data; 0.25 without it)
        self.tf = ChannelsTransformer(cfg.dim, cfg.dim, cfg.depth, cfg.heads, cfg.context, list(cfg.right_context),
                                      final_norm=False)
        self.vq = FactorizedVQ(cfg.dim, cfg.codebook_size, cfg.codebook_dim, cfg.commitment, cfg.codebook_loss_weight,
                               cfg.l2_normalize)
        self.classifier = nn.Conv1d(cfg.dim, cfg.num_codes, 1)  # training only
        self.hop = self.cnn.hop

    @property
    def lookahead_frames(self) -> int:
        return self.tf.lookahead_frames

    def forward(self, wav: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.vq(self.tf(self.cnn(wav[:, None])))

    def step(self, wav: torch.Tensor, state: State | None) -> tuple[torch.Tensor, State]:
        st = state or {"cnn": None, "tf": None}
        h, s1 = self.cnn.step(wav[:, None], st["cnn"])
        h, s2 = self.tf.step(h, st["tf"])
        return (self.vq(h)["z_q"] if h.shape[-1] else h), {"cnn": s1, "tf": s2}

    def pasr_loss(self, wav: torch.Tensor, codes: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self(wav)
        logits = self.classifier(out["z_q"])
        n = min(logits.shape[-1], codes.shape[-1])
        ce = F.cross_entropy(logits[..., :n], codes[:, :n].long())
        acc = (logits[..., :n].argmax(1) == codes[:, :n]).float().mean()
        return {"loss": ce + out["commit"] + out["codebook"], "ce": ce, "acc": acc, "commit": out["commit"],
                "codebook": out["codebook"]}
