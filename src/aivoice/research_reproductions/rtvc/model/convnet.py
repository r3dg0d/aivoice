"""Causal residual conv stack in the official RT-VC layout (conv-BN-ReLU, blocks of residual units with
dilations 1/3/9, output conv). Re-implemented by aivoice with explicit streaming state; BatchNorm is
used in eval mode at inference, where it is a per-frame affine map (streaming-safe)."""

from __future__ import annotations

import torch
from torch import nn

from ...common.streaming_nn import CausalConv1d, State


class _Unit(nn.Module):
    def __init__(self, ch: int, k: int, d: int) -> None:
        super().__init__()
        self.conv = CausalConv1d(ch, ch, k, dilation=d)
        self.bn = nn.BatchNorm1d(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.relu(self.bn(self.conv(x)))

    def step(self, x: torch.Tensor, state: State | None) -> tuple[torch.Tensor, State]:
        y, st = self.conv.step(x, state)
        return x + torch.relu(self.bn(y)), st


class CausalConvNet(nn.Module):
    def __init__(self, cin: int, hidden: int, cout: int, k: int = 3, n_blocks: int = 3) -> None:
        super().__init__()
        self.inp = CausalConv1d(cin, hidden, k)
        self.bn = nn.BatchNorm1d(hidden)
        self.units = nn.ModuleList(_Unit(hidden, k, d) for _ in range(n_blocks) for d in (1, 3, 9))
        self.out = CausalConv1d(hidden, cout, k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.bn(self.inp(x)))
        for u in self.units:
            x = u(x)
        return self.out(x)

    def step(self, x: torch.Tensor, state: State | None) -> tuple[torch.Tensor, State]:
        st = state or {"inp": None, "units": [None] * len(self.units), "out": None}
        x, s_in = self.inp.step(x, st["inp"])
        x = torch.relu(self.bn(x))
        su = []
        for u, s in zip(self.units, st["units"]):
            x, s = u.step(x, s)
            su.append(s)
        x, s_out = self.out.step(x, st["out"])
        return x, {"inp": s_in, "units": su, "out": s_out}


def mlp(cin: int, hidden: int, cout: int, expansion: int = 1, n_mid: int = 1) -> nn.Sequential:
    h = hidden * expansion
    layers: list[nn.Module] = [nn.Linear(cin, h), nn.ReLU()]
    for _ in range(n_mid):
        layers += [nn.Linear(h, h), nn.ReLU()]
    layers.append(nn.Linear(h, cout))
    return nn.Sequential(*layers)
