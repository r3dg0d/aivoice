"""Inference-time vector-field fusion for flow-matching decoders (experimental, off by default).

PFluxTTS (arXiv:2602.04160) fuses two decoders, one duration-guided and one
alignment-free, by combining their predicted velocity fields during ODE
sampling: v(x, t) = w(t) v_a(x, t) + (1 - w(t)) v_b(x, t). That is only
meaningful when both fields are defined on the same state space and the same
time parameterisation: same latent/feature type, dimension and frame rate,
same normalisation, same t convention (noise at t=0, data at t=1), and the same
conditioning frame alignment. A convex combination of two velocity fields on
one space is itself a valid velocity field, so the ODE is still well posed.

No engine shipped with aivoice exposes its velocity field with a compatible
spec (VoxCPM2 is an autoregressive LM with a local DiT; X-Voice and VoiceStar do
not share state spaces), so :func:`check_compatible` refuses every installed
pair. The module exists so a pair of compatible decoders can be fused and tested
the moment one is available, with the safety checks already in place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np


@dataclass(frozen=True)
class FieldSpec:
    state: str  # e.g. "mel-100@24kHz-hop256", "vae-latent-64@25Hz"
    dim: int
    frame_rate: float
    normalization: str
    time_convention: str = "noise@0->data@1"


class VectorField(Protocol):
    spec: FieldSpec

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray: ...


class IncompatibleFields(ValueError):
    pass


def check_compatible(a: FieldSpec, b: FieldSpec) -> None:
    for attr in ("state", "dim", "frame_rate", "normalization", "time_convention"):
        if getattr(a, attr) != getattr(b, attr):
            raise IncompatibleFields(f"cannot fuse: {attr} differs ({getattr(a, attr)!r} vs {getattr(b, attr)!r})")


def fused_field(a: VectorField, b: VectorField, weight: Callable[[float], float] | float = 0.5) -> Callable[[np.ndarray, float], np.ndarray]:
    check_compatible(a.spec, b.spec)
    w = weight if callable(weight) else (lambda t, c=float(weight): c)

    def v(x: np.ndarray, t: float) -> np.ndarray:
        wt = float(np.clip(w(t), 0.0, 1.0))
        return wt * a(x, t) + (1 - wt) * b(x, t)

    return v


def euler_sample(v: Callable[[np.ndarray, float], np.ndarray], x0: np.ndarray, steps: int = 32,
                 sway: float | None = None) -> np.ndarray:
    """Integrate dx/dt = v(x, t) from t=0 to 1. Optional sway sampling (F5-TTS) skews steps toward t=0."""
    ts = np.linspace(0.0, 1.0, steps + 1)
    if sway is not None:
        ts = ts + sway * (np.cos(np.pi / 2 * ts) - 1 + ts)
    x = np.asarray(x0, dtype=np.float64)
    for t0, t1 in zip(ts[:-1], ts[1:]):
        x = x + (t1 - t0) * v(x, float(t0))
    return x
