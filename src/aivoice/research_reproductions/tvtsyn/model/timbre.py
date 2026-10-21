"""TVTSyn time-varying timbre: Global Timbre Memory + gated Slerp.

g = [x-vector (512) ; ECAPA (192)] (704-d, frozen teachers). GTM expands g into 48 key/value tokens
(MLP + learned priors). Each content frame queries them with cross-attention (RoPE on queries, absolute
positions, so streaming is exact). The result is projected and L2-normalised to 192-d, and a per-frame
gate alpha in [0,1] spherically interpolates (Slerp, *not* linear) between it and a projection of the
global embedding."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ...common.streaming_nn import State, rope
from ..config import TimbreConfig


def slerp(a: torch.Tensor, b: torch.Tensor, t: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Spherical interpolation of unit vectors along the last dim; t broadcastable to a[..., :1]."""
    a, b = F.normalize(a, dim=-1), F.normalize(b, dim=-1)
    dot = (a * b).sum(-1, keepdim=True).clamp(-1 + eps, 1 - eps)
    om = torch.acos(dot)
    so = torch.sin(om).clamp_min(eps)
    return torch.sin((1 - t) * om) / so * a + torch.sin(t * om) / so * b


class GlobalTimbreMemory(nn.Module):
    def __init__(self, g_dim: int, d_kv: int, n_tokens: int, hidden: int) -> None:
        super().__init__()
        self.n, self.d = n_tokens, d_kv
        self.net = nn.Sequential(nn.Linear(g_dim, hidden), nn.GELU(), nn.Linear(hidden, 2 * n_tokens * d_kv))
        self.k_prior = nn.Parameter(torch.randn(n_tokens, d_kv) * 0.02)
        self.v_prior = nn.Parameter(torch.randn(n_tokens, d_kv) * 0.02)

    def forward(self, g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        k, v = self.net(g).split(self.n * self.d, -1)
        return k.view(-1, self.n, self.d) + self.k_prior, v.view(-1, self.n, self.d) + self.v_prior


class TimeVaryingTimbre(nn.Module):
    def __init__(self, cfg: TimbreConfig, content_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.gtm = GlobalTimbreMemory(cfg.g_dim, cfg.d_kv, cfg.num_tokens, cfg.gtm_hidden)
        self.q = nn.Conv1d(content_dim, cfg.d_kv, 1, bias=False)
        self.k = nn.Linear(cfg.d_kv, cfg.d_kv, bias=False)
        self.v = nn.Linear(cfg.d_kv, cfg.d_kv, bias=False)
        self.qp = nn.Linear(cfg.d_kv, cfg.d_kv, bias=False)
        self.o = nn.Linear(cfg.d_kv, cfg.d_kv, bias=False)
        self.post = nn.Conv1d(cfg.d_kv, cfg.d_out, 1)
        self.spk = nn.Linear(cfg.g_dim, cfg.d_out)
        self.gate = nn.Sequential(nn.Conv1d(content_dim, cfg.gate_hidden, 1), nn.GELU(),
                                  nn.Conv1d(cfg.gate_hidden, 1, 1), nn.Sigmoid())

    def _attend(self, x: torch.Tensor, g: torch.Tensor, pos0: int) -> torch.Tensor:
        B, _, T = x.shape
        h, dh = self.cfg.heads, self.cfg.d_kv // self.cfg.heads
        K, V = self.gtm(g)
        if self.training and self.cfg.token_dropout > 0:
            keep = (torch.rand(K.shape[1], device=x.device) > self.cfg.token_dropout).float()
            if keep.sum() == 0:
                keep[0] = 1
            K, V = K * keep[None, :, None], V * keep[None, :, None]
        q = self.qp(self.q(x).transpose(1, 2)).view(B, T, h, dh).transpose(1, 2)
        q = rope(q, torch.arange(pos0, pos0 + T, device=x.device))
        k = self.k(K).view(B, -1, h, dh).transpose(1, 2)
        v = self.v(V).view(B, -1, h, dh).transpose(1, 2)
        a = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(dh), -1) @ v
        return self.o(a.transpose(1, 2).reshape(B, T, -1)).transpose(1, 2)    # [B, d_kv, T]

    def forward(self, x: torch.Tensor, g: torch.Tensor, pos0: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """x content [B, C, T], g [B, g_dim] -> (tv timbre [B, d_out, T], gate alpha [B, 1, T])."""
        H = F.normalize(self.post(self._attend(x, g, pos0)), dim=1)
        gg = self.spk(F.normalize(g, dim=-1))[:, None].expand(-1, x.shape[-1], -1)
        alpha = self.gate(x)
        tv = slerp(H.transpose(1, 2), gg, alpha.transpose(1, 2)).transpose(1, 2)
        return tv, alpha

    def step(self, x: torch.Tensor, g: torch.Tensor, state: State | None) -> tuple[torch.Tensor, State]:
        pos = (state or {}).get("pos", 0)
        tv, _ = self.forward(x, g, pos)
        return tv, {"pos": pos + x.shape[-1]}
