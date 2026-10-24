"""Causal, streaming-exact model blocks used by more than one reproduction.

* ``CausalSEANetEncoder``: waveform -> frames (strided causal CNN, SEANet/EnCodec-style layout).
* ``CausalHiFiGAN``: frames -> waveform (HiFi-GAN V1 layout with causal convs, causal transposed convs
  and per-stage global conditioning).
* ``AttentionPooling``: learnable attention pooling of frame features to one vector.
* ``StreamingSequential`` / ``ChannelsTransformer``: glue that gives every block ``forward`` and ``step``.

All of it is aivoice code written from the published descriptions of these standard architectures
(SEANet: Tagliasacchi et al. 2020 / EnCodec; HiFi-GAN: Kong et al. 2020).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .streaming_nn import CausalConv1d, CausalConvTranspose1d, State, Transformer


class Pointwise(nn.Module):
    """Marks a module as time-pointwise (activations, 1x1 ops): ``step`` = ``forward``."""

    def __init__(self, fn: nn.Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x)

    def step(self, x: torch.Tensor, state: State | None) -> tuple[torch.Tensor, State | None]:
        return self.fn(x), None


class StreamingSequential(nn.Module):
    def __init__(self, *mods: nn.Module) -> None:
        super().__init__()
        for m in mods:
            if not hasattr(m, "step"):
                raise TypeError(f"{type(m).__name__} has no step(); wrap pointwise modules in Pointwise")
        self.mods = nn.ModuleList(mods)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for m in self.mods:
            x = m(x)
        return x

    def step(self, x: torch.Tensor, state: State | None) -> tuple[torch.Tensor, State]:
        states = (state or {}).get("s") or [None] * len(self.mods)
        new = []
        for m, s in zip(self.mods, states):
            x, s = m.step(x, s)
            new.append(s)
        return x, {"s": new}


class ResidualUnit(nn.Module):
    """x + conv1x1(act(conv_k,d(act(x)))) with causal convs (stride 1, so output stays aligned)."""

    def __init__(self, ch: int, kernel: int = 3, dilation: int = 1, act: str = "elu") -> None:
        super().__init__()
        self.act = nn.ELU() if act == "elu" else nn.LeakyReLU(0.1)
        self.c1 = CausalConv1d(ch, ch // 2 if act == "elu" else ch, kernel, dilation=dilation)
        self.c2 = CausalConv1d(ch // 2 if act == "elu" else ch, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.c2(self.act(self.c1(self.act(x))))

    def step(self, x: torch.Tensor, state: State | None) -> tuple[torch.Tensor, State]:
        state = state or {"c1": None, "c2": None}
        y, s1 = self.c1.step(self.act(x), state["c1"])
        y, s2 = self.c2.step(self.act(y), state["c2"])
        return x + y, {"c1": s1, "c2": s2}


class CausalSEANetEncoder(nn.Module):
    """Waveform [B, 1, T] -> features [B, dim, T / prod(strides)]."""

    def __init__(self, dim: int = 512, n_filters: int = 32, strides: tuple[int, ...] = (2, 4, 5, 8),
                 n_residual: int = 1, kernel: int = 7, last_kernel: int = 3) -> None:
        super().__init__()
        self.strides = tuple(strides)
        self.hop = int(torch.tensor(strides).prod())
        layers: list[nn.Module] = [CausalConv1d(1, n_filters, kernel)]
        ch = n_filters
        for s in strides:
            for j in range(n_residual):
                layers.append(ResidualUnit(ch, 3, 2 ** j))
            layers += [Pointwise(nn.ELU()), CausalConv1d(ch, ch * 2, 2 * s, stride=s)]
            ch *= 2
        layers += [Pointwise(nn.ELU()), CausalConv1d(ch, dim, last_kernel)]
        self.net = StreamingSequential(*layers)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        return self.net(wav)

    def step(self, wav: torch.Tensor, state: State | None) -> tuple[torch.Tensor, State]:
        return self.net.step(wav, state)


class CausalSEANetDecoder(nn.Module):
    """Features [B, dim, T] -> waveform [B, 1, T * prod(ratios)] (SEANet/EnCodec decoder layout, causal)."""

    def __init__(self, dim: int = 512, n_filters: int = 64, ratios: tuple[int, ...] = (8, 5, 4, 2),
                 n_residual: int = 1, kernel: int = 7, last_kernel: int = 7) -> None:
        super().__init__()
        self.hop = int(torch.tensor(ratios).prod())
        ch = n_filters * 2 ** len(ratios)
        self.inp = CausalConv1d(dim, ch, kernel)
        self.ups = nn.ModuleList()
        self.res = nn.ModuleList()
        for r in ratios:
            self.ups.append(CausalConvTranspose1d(ch, ch // 2, 2 * r, r))
            ch //= 2
            self.res.append(StreamingSequential(*[ResidualUnit(ch, 3, 2 ** j) for j in range(n_residual)]))
        self.out = CausalConv1d(ch, 1, last_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inp(x)
        for up, res in zip(self.ups, self.res):
            x = res(up(F.elu(x)))
        return self.out(F.elu(x))

    def step(self, x: torch.Tensor, state: State | None) -> tuple[torch.Tensor, State]:
        st = state or {"inp": None, "ups": [None] * len(self.ups), "res": [None] * len(self.res), "out": None}
        x, s_in = self.inp.step(x, st["inp"])
        su, sr = [], []
        for up, res, a, b in zip(self.ups, self.res, st["ups"], st["res"]):
            x, a = up.step(F.elu(x), a)
            x, b = res.step(x, b)
            su.append(a)
            sr.append(b)
        y, s_out = self.out.step(F.elu(x), st["out"])
        return y, {"inp": s_in, "ups": su, "res": sr, "out": s_out}


class ChannelsTransformer(nn.Module):
    """[B, C, T] wrapper around the streaming Transformer (input/output projections optional)."""

    def __init__(self, dim_in: int, d_model: int, depth: int, heads: int, context: int,
                 right_context: int | list[int] = 0,
                 dim_out: int | None = None, ff_mult: int = 4, final_norm: bool = True) -> None:
        super().__init__()
        self.inp = nn.Linear(dim_in, d_model) if dim_in != d_model else nn.Identity()
        self.tf = Transformer(d_model, depth, heads, context, right_context, ff_mult)
        # final_norm=False matches Mimi's ProjectedTransformer (TVTSyn): no LayerNorm after the last block
        self.norm = nn.LayerNorm(d_model) if final_norm else nn.Identity()
        self.out = nn.Linear(d_model, dim_out) if dim_out and dim_out != d_model else nn.Identity()
        self.grad_checkpoint = False

    @property
    def lookahead_frames(self) -> int:
        return self.tf.lookahead_frames

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.inp(x.transpose(1, 2))
        for b in self.tf.blocks:
            h = checkpoint(b, h, use_reentrant=False) if self.grad_checkpoint and self.training else b(h)
        return self.out(self.norm(h)).transpose(1, 2)

    def step(self, x: torch.Tensor, state: State | None) -> tuple[torch.Tensor, State]:
        h, st = self.tf.step(self.inp(x.transpose(1, 2)), state)
        return self.out(self.norm(h)).transpose(1, 2), st


class MRFBlock(nn.Module):
    """HiFi-GAN ResBlock1: for each dilation, x += c2(lrelu(c1_d(lrelu(x))))."""

    def __init__(self, ch: int, kernel: int, dilations: tuple[int, ...] = (1, 3, 5)) -> None:
        super().__init__()
        self.c1 = nn.ModuleList(CausalConv1d(ch, ch, kernel, dilation=d) for d in dilations)
        self.c2 = nn.ModuleList(CausalConv1d(ch, ch, kernel) for _ in dilations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for a, b in zip(self.c1, self.c2):
            x = x + b(F.leaky_relu(a(F.leaky_relu(x, 0.1)), 0.1))
        return x

    def step(self, x: torch.Tensor, state: State | None) -> tuple[torch.Tensor, State]:
        st = (state or {}).get("s") or [[None, None] for _ in self.c1]
        new = []
        for (a, b), (sa, sb) in zip(zip(self.c1, self.c2), st):
            y, sa = a.step(F.leaky_relu(x, 0.1), sa)
            y, sb = b.step(F.leaky_relu(y, 0.1), sb)
            x = x + y
            new.append([sa, sb])
        return x, {"s": new}


class CausalHiFiGAN(nn.Module):
    """Frames [B, cin, T] (+ global condition [B, cond_dim]) -> waveform [B, 1, T * prod(rates)].

    Layout of HiFi-GAN V1 (upsample kernels 2*rate, MRF kernels 3/7/11 with dilations 1/3/5), with every
    convolution made causal and the transposed convolutions trimmed causally. The global condition is
    projected and added before every upsampling stage.
    """

    def __init__(self, cin: int, cond_dim: int = 0, ch: int = 512, rates: tuple[int, ...] = (8, 5, 4, 2),
                 mrf_kernels: tuple[int, ...] = (3, 7, 11), mrf_dilations: tuple[int, ...] = (1, 3, 5)) -> None:
        super().__init__()
        self.rates = tuple(rates)
        self.hop = int(torch.tensor(rates).prod())
        self.pre = CausalConv1d(cin, ch, 7)
        self.ups = nn.ModuleList()
        self.mrfs = nn.ModuleList()
        self.conds = nn.ModuleList()
        c = ch
        for r in rates:
            self.conds.append(nn.Linear(cond_dim, c) if cond_dim else nn.Identity())
            self.ups.append(CausalConvTranspose1d(c, c // 2, 2 * r, r))
            c //= 2
            self.mrfs.append(nn.ModuleList(MRFBlock(c, k, mrf_dilations) for k in mrf_kernels))
        self.post = CausalConv1d(c, 1, 7)
        self.cond_dim = cond_dim

    def _cond(self, i: int, x: torch.Tensor, g: torch.Tensor | None) -> torch.Tensor:
        return x + self.conds[i](g)[:, :, None] if self.cond_dim and g is not None else x

    def forward(self, x: torch.Tensor, g: torch.Tensor | None = None) -> torch.Tensor:
        x = self.pre(x)
        for i, (up, mrf) in enumerate(zip(self.ups, self.mrfs)):
            x = up(F.leaky_relu(self._cond(i, x, g), 0.1))
            x = sum(m(x) for m in mrf) / len(mrf)
        return torch.tanh(self.post(F.leaky_relu(x, 0.01)))

    def step(self, x: torch.Tensor, state: State | None, g: torch.Tensor | None = None) -> tuple[torch.Tensor, State]:
        st = state or {"pre": None, "up": [None] * len(self.ups), "mrf": [[None] * len(m) for m in self.mrfs],
                       "post": None}
        x, s_pre = self.pre.step(x, st["pre"])
        s_up, s_mrf = [], []
        for i, (up, mrf) in enumerate(zip(self.ups, self.mrfs)):
            x, su = up.step(F.leaky_relu(self._cond(i, x, g), 0.1), st["up"][i])
            outs, sm = [], []
            for m, s in zip(mrf, st["mrf"][i]):
                y, s = m.step(x, s)
                outs.append(y)
                sm.append(s)
            x = sum(outs) / len(outs)
            s_up.append(su)
            s_mrf.append(sm)
        y, s_post = self.post.step(F.leaky_relu(x, 0.01), st["post"])
        return torch.tanh(y), {"pre": s_pre, "up": s_up, "mrf": s_mrf, "post": s_post}


class AttentionPooling(nn.Module):
    """e = sum_t softmax_t(w^T tanh(W h_t + b)) h_t, then a linear projection (learnable attention pooling)."""

    def __init__(self, dim_in: int, dim_out: int, hidden: int = 256) -> None:
        super().__init__()
        self.att = nn.Sequential(nn.Linear(dim_in, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.proj = nn.Linear(dim_in, dim_out)

    def forward(self, h: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """h: [B, T, dim_in]; mask: [B, T] True = valid. Returns [B, dim_out]."""
        a = self.att(h).squeeze(-1)
        if mask is not None:
            a = a.masked_fill(~mask, float("-inf"))
        w = torch.softmax(a, dim=1)
        return self.proj(torch.einsum("bt,btd->bd", w, h))


def param_count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def set_grad_checkpoint(model: nn.Module, on: bool) -> int:
    """Enable activation checkpointing on every ChannelsTransformer; returns how many were switched."""
    n = 0
    for m in model.modules():
        if isinstance(m, ChannelsTransformer):
            m.grad_checkpoint = on
            n += 1
    return n


__all__ = ["AttentionPooling", "CausalHiFiGAN", "CausalSEANetDecoder", "CausalSEANetEncoder", "ChannelsTransformer", "MRFBlock", "Pointwise",
           "ResidualUnit", "StreamingSequential", "param_count", "set_grad_checkpoint"]

