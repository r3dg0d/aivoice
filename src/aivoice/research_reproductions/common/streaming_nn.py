"""Streaming-exact neural building blocks.

Every layer has two code paths that must agree to float precision:

* ``forward(x)``: whole sequence (training), causal by construction;
* ``step(x, state)``: incremental (live), returning ``(y, new_state)``.

The state is explicit (a dict of tensors) instead of hidden module attributes, so one model can
serve several streams and a reset is just ``state = None``. Tests in tests/test_reproductions.py
check ``cat(step(chunks)) == forward(cat(chunks))`` for arbitrary chunkings and verify strict
causality (changing future input never changes past output).

These are aivoice's own implementations of standard streaming techniques (causal padding, cached
left context, overlap state for transposed convolutions, KV caches with a bounded window). They are
shared by the paper reproductions so each reproduction differs only in what its paper specifies.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

State = dict[str, Any]


class CausalConv1d(nn.Module):
    """Causal 1-D convolution: output t depends on inputs <= t*stride + (stride-1) only.

    Full-sequence padding is ``(k-1)*dilation - (stride-1)`` on the left (the EnCodec/Mimi causal
    convention), so ``T`` input samples give ``ceil(T/stride)`` outputs when ``T`` is a multiple of
    the stride. ``step`` keeps exactly the input samples still needed by future outputs.
    """

    def __init__(self, cin: int, cout: int, kernel_size: int, stride: int = 1, dilation: int = 1,
                 groups: int = 1, bias: bool = True, weight_norm: bool = False) -> None:
        super().__init__()
        conv = nn.Conv1d(cin, cout, kernel_size, stride=stride, dilation=dilation, groups=groups, bias=bias)
        self.conv = nn.utils.parametrizations.weight_norm(conv) if weight_norm else conv
        self.k, self.s, self.d = kernel_size, stride, dilation
        self.eff_k = (kernel_size - 1) * dilation + 1
        self.pad = self.eff_k - stride  # left padding (may be negative only for k < stride, which we refuse)
        if self.pad < 0:
            raise ValueError("kernel_size must be >= stride for a causal strided convolution")

    @property
    def receptive_field(self) -> int:
        return self.eff_k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.pad, 0)))

    def init_state(self, x: torch.Tensor) -> State:
        return {"buf": x.new_zeros(x.shape[0], x.shape[1], self.pad)}

    def step(self, x: torch.Tensor, state: State | None) -> tuple[torch.Tensor, State]:
        if state is None:
            state = self.init_state(x)
        buf = torch.cat([state["buf"], x], dim=-1)
        n = (buf.shape[-1] - self.eff_k) // self.s + 1 if buf.shape[-1] >= self.eff_k else 0
        if n <= 0:
            return x.new_zeros(x.shape[0], self.conv.out_channels, 0), {"buf": buf}
        y = self.conv(buf[..., : (n - 1) * self.s + self.eff_k])
        return y, {"buf": buf[..., n * self.s :]}


class CausalConvTranspose1d(nn.Module):
    """Causal transposed convolution (upsampling by ``stride``): the right overhang is trimmed.

    ``step`` carries the overlap of ``kernel_size - stride`` samples that later inputs add to.
    """

    def __init__(self, cin: int, cout: int, kernel_size: int, stride: int, weight_norm: bool = False) -> None:
        super().__init__()
        if kernel_size < stride:
            raise ValueError("kernel_size must be >= stride")
        conv = nn.ConvTranspose1d(cin, cout, kernel_size, stride=stride, bias=False)
        self.conv = nn.utils.parametrizations.weight_norm(conv) if weight_norm else conv
        self.bias = nn.Parameter(torch.zeros(cout))
        self.k, self.s = kernel_size, stride
        self.ov = kernel_size - stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        y = y[..., : x.shape[-1] * self.s]
        return y + self.bias[None, :, None]

    def step(self, x: torch.Tensor, state: State | None) -> tuple[torch.Tensor, State]:
        cout = self.bias.shape[0]
        if state is None:
            state = {"ov": x.new_zeros(x.shape[0], cout, self.ov)}
        if x.shape[-1] == 0:
            return x.new_zeros(x.shape[0], cout, 0), state
        y = self.conv(x)  # length (n-1)*s + k
        if self.ov:
            y = y.clone()
            y[..., : self.ov] += state["ov"]
        n = x.shape[-1]
        out = y[..., : n * self.s] + self.bias[None, :, None]
        return out, {"ov": y[..., n * self.s :]}


class ChannelLayerNorm(nn.Module):
    """LayerNorm over channels of a [B, C, T] tensor (per frame: streaming-safe, unlike BatchNorm)."""

    def __init__(self, c: int) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


def rope(x: torch.Tensor, pos: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """Rotary embedding. x: [B, H, T, D] (D even); pos: [T] absolute positions."""
    d = x.shape[-1]
    inv = 1.0 / (base ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d))
    ang = pos.to(torch.float32)[:, None] * inv[None, :]  # [T, D/2]
    cos, sin = ang.cos()[None, None], ang.sin()[None, None]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2).to(x.dtype)


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention over a bounded window: frame t sees [t-context+1, t+right_context].

    ``right_context`` > 0 is *lookahead*: in streaming the output for frame t can only be produced
    once frame t+right_context has arrived, so ``step`` returns outputs delayed by that many frames
    (and the module reports it in ``lookahead_frames``). Positions use RoPE with absolute indices,
    so cached keys stay valid across chunks.
    """

    def __init__(self, dim: int, heads: int, context: int, right_context: int = 0, dropout: float = 0.0) -> None:
        super().__init__()
        assert dim % heads == 0
        self.h, self.dh = heads, dim // heads
        self.context, self.right = context, right_context
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout

    @property
    def lookahead_frames(self) -> int:
        return self.right

    def _mask(self, tq: torch.Tensor, tk: torch.Tensor) -> torch.Tensor:
        rel = tk[None, :] - tq[:, None]  # key - query
        return (rel <= self.right) & (rel > -self.context)

    def _split(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        return q, k, v

    def _attend(self, q, k, v, mask) -> torch.Tensor:
        att = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.dh)
        att = att.masked_fill(~mask[None, None], float("-inf"))
        w = torch.softmax(att, dim=-1)
        if self.training and self.dropout:
            w = F.dropout(w, self.dropout)
        y = torch.matmul(w, v)  # [B,H,T,dh]
        return y.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)

    def forward(self, x: torch.Tensor, pos0: int = 0) -> torch.Tensor:
        """x: [B, T, D]."""
        T = x.shape[1]
        pos = torch.arange(pos0, pos0 + T, device=x.device)
        q, k, v = self._split(x)
        q, k = rope(q, pos), rope(k, pos)
        return self.out(self._attend(q, k, v, self._mask(pos, pos)))

    def step(self, x: torch.Tensor, state: State | None) -> tuple[torch.Tensor, State]:
        """Incremental: x [B, n, D] are the next frames; returns outputs for frames that are now complete."""
        B, n, _ = x.shape
        if state is None:
            state = {"k": x.new_zeros(B, self.h, 0, self.dh), "v": x.new_zeros(B, self.h, 0, self.dh),
                     "q": x.new_zeros(B, self.h, 0, self.dh), "pos": 0, "qpos": 0}
        pos = torch.arange(state["pos"], state["pos"] + n, device=x.device)
        q, k, v = self._split(x)
        q, k = rope(q, pos), rope(k, pos)
        K = torch.cat([state["k"], k], dim=2)
        V = torch.cat([state["v"], v], dim=2)
        Q = torch.cat([state["q"], q], dim=2)  # queries waiting for their right context
        kpos = torch.arange(state["pos"] + n - K.shape[2], state["pos"] + n, device=x.device)
        qpos = torch.arange(state["qpos"], state["qpos"] + Q.shape[2], device=x.device)
        ready = int((qpos + self.right <= state["pos"] + n - 1).sum())  # queries whose lookahead has arrived
        y = self._attend(Q[:, :, :ready], K, V, self._mask(qpos[:ready], kpos)) if ready else x.new_zeros(B, 0, self.h * self.dh)
        keep = self.context - 1 + self.right  # keys any future query can still see
        new = {"k": K[:, :, -keep:] if keep > 0 else K[:, :, :0], "v": V[:, :, -keep:] if keep > 0 else V[:, :, :0],
               "q": Q[:, :, ready:], "pos": state["pos"] + n, "qpos": state["qpos"] + ready}
        return self.out(y) if ready else y, new


class TransformerBlock(nn.Module):
    """Pre-norm block: windowed causal MHSA + GELU MLP, with layer scale (stable deep stacks)."""

    def __init__(self, dim: int, heads: int, context: int, right_context: int = 0, ff_mult: int = 4,
                 layer_scale: float = 0.01) -> None:
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = CausalSelfAttention(dim, heads, context, right_context)
        self.ff = nn.Sequential(nn.Linear(dim, ff_mult * dim), nn.GELU(), nn.Linear(ff_mult * dim, dim))
        self.g1 = nn.Parameter(torch.full((dim,), layer_scale))
        self.g2 = nn.Parameter(torch.full((dim,), layer_scale))

    @property
    def lookahead_frames(self) -> int:
        return self.attn.right

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.g1 * self.attn(self.n1(x))
        return x + self.g2 * self.ff(self.n2(x))

    def step(self, x: torch.Tensor, state: State | None) -> tuple[torch.Tensor, State]:
        state = state or {"attn": None, "res": x.new_zeros(x.shape[0], 0, x.shape[2])}
        a, st = self.attn.step(self.n1(x), state["attn"])
        res = torch.cat([state["res"], x], dim=1)  # residual must wait as long as attention does
        r, res = res[:, : a.shape[1]], res[:, a.shape[1] :]
        y = r + self.g1 * a
        y = y + self.g2 * self.ff(self.n2(y))
        return y, {"attn": st, "res": res}


class Transformer(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, context: int, right_context: int = 0, ff_mult: int = 4,
                 layer_scale: float = 0.01) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(TransformerBlock(dim, heads, context, right_context, ff_mult, layer_scale)
                                    for _ in range(depth))

    @property
    def lookahead_frames(self) -> int:
        return sum(b.lookahead_frames for b in self.blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = b(x)
        return x

    def step(self, x: torch.Tensor, state: State | None) -> tuple[torch.Tensor, State]:
        state = state or {"blocks": [None] * len(self.blocks)}
        out = []
        for b, s in zip(self.blocks, state["blocks"]):
            x, s = b.step(x, s)
            out.append(s)
        return x, {"blocks": out}


def state_nbytes(state: Any) -> int:
    """Total bytes held in a (nested) streaming state: lets tests check caches stay bounded."""
    if isinstance(state, torch.Tensor):
        return state.numel() * state.element_size()
    if isinstance(state, dict):
        return sum(state_nbytes(v) for v in state.values())
    if isinstance(state, (list, tuple)):
        return sum(state_nbytes(v) for v in state)
    return 0


__all__ = ["CausalConv1d", "CausalConvTranspose1d", "CausalSelfAttention", "ChannelLayerNorm", "State",
           "Transformer", "TransformerBlock", "rope", "state_nbytes"]
