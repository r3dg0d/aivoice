"""Conservative output enhancement: only steps that measurably help are kept.

Modes:

``off``            nothing.
``auto``           DC removal, click repair and loudness normalisation, each applied only
                   if its own metric improves; then, when a speaker encoder is available,
                   the result is rejected outright if speaker similarity to the
                   unenhanced audio drops below ``min_identity``.
``high-fidelity``  needs a real enhancement / bandwidth-extension model. aivoice ships
                   none, so this mode raises instead of pretending (resampling is not
                   high-frequency reconstruction).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .loudness import integrated_loudness, normalize

MODES = ("off", "auto", "high-fidelity")


class EnhancementUnavailable(RuntimeError):
    pass


@dataclass
class EnhanceReport:
    mode: str
    applied: list[str] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)
    clicks_before: int = 0
    clicks_after: int = 0
    lufs_before: float | None = None
    lufs_after: float | None = None
    identity_similarity: float | None = None
    rejected: str | None = None


def detect_discontinuities(x: np.ndarray, sr: int, *, threshold: float = 20.0, min_abs: float = 0.05) -> np.ndarray:
    """Sample indices of impulsive discontinuities (clicks, hard splices).

    Uses the second difference against a robust local scale (median absolute
    second difference over 10 ms blocks, max of the block and its neighbours).
    threshold=20 was chosen on synthetic speech with injected impulses:
    0.08 false detections per clean clip, 90% recall (docs/research/experiments/G.md).
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size < 64:
        return np.zeros(0, dtype=int)
    d2 = np.abs(x[2:] - 2 * x[1:-1] + x[:-2])
    win = max(32, int(sr * 0.01))
    nb = int(np.ceil(d2.size / win))
    med = np.median(np.pad(d2, (0, nb * win - d2.size)).reshape(nb, win), axis=1)
    m3 = np.maximum.reduce([med, np.roll(med, 1), np.roll(med, -1)])
    scale = np.repeat(m3, win)[: d2.size] * 1.4826 + 1e-6
    idx = np.where((d2 > threshold * scale) & (d2 > min_abs))[0] + 1
    keep: list[int] = []
    for i in idx:
        if not keep or i - keep[-1] > win:
            keep.append(int(i))
    return np.array(keep, dtype=int)


def repair_clicks(x: np.ndarray, idx: np.ndarray, sr: int, *, width_ms: float = 1.0) -> np.ndarray:
    """Replace a few samples around each click with a cubic-smooth bridge."""
    y = np.asarray(x, dtype=np.float32).copy()
    w = max(2, int(sr * width_ms / 1000))
    for i in idx:
        a, b = max(0, i - w), min(y.size - 1, i + w + 1)
        if b - a < 3:
            continue
        t = np.linspace(0, 1, b - a + 1)
        s = t * t * (3 - 2 * t)
        y[a : b + 1] = (1 - s) * y[a] + s * y[b]
    return y


def remove_dc(x: np.ndarray) -> np.ndarray:
    return (np.asarray(x, dtype=np.float32) - np.float32(np.mean(x))).astype(np.float32)


def crossfade_concat(parts: list[np.ndarray], sr: int, *, fade_ms: float = 15.0, gaps_ms: list[float] | None = None) -> np.ndarray:
    """Join segments with equal-power crossfades (and optional silent gaps) so boundaries do not click."""
    if not parts:
        return np.zeros(0, np.float32)
    n_fade = int(sr * fade_ms / 1000)
    out = np.asarray(parts[0], np.float32)
    for k, nxt in enumerate(parts[1:]):
        nxt = np.asarray(nxt, np.float32)
        gap = int(sr * (gaps_ms[k] if gaps_ms and k < len(gaps_ms) else 0) / 1000)
        if gap > 0:
            # fade out, silence, fade in: no hard edges either side of the pause
            f = min(n_fade, out.size, nxt.size)
            if f:
                ramp = np.sin(np.linspace(0, np.pi / 2, f)).astype(np.float32)
                out = out.copy()
                out[-f:] *= ramp[::-1]
                nxt = nxt.copy()
                nxt[:f] *= ramp
            out = np.concatenate([out, np.zeros(gap, np.float32), nxt])
            continue
        f = min(n_fade, out.size, nxt.size)
        if f == 0:
            out = np.concatenate([out, nxt])
            continue
        t = np.linspace(0, np.pi / 2, f)
        fo, fi = np.cos(t).astype(np.float32), np.sin(t).astype(np.float32)
        mid = out[-f:] * fo + nxt[:f] * fi
        out = np.concatenate([out[:-f], mid, nxt[f:]])
    return out


def enhance(
    x: np.ndarray,
    sr: int,
    *,
    mode: str = "auto",
    target_lufs: float = -20.0,
    encoder: Any | None = None,
    min_identity: float = 0.95,
) -> tuple[np.ndarray, EnhanceReport]:
    if mode not in MODES:
        raise ValueError(f"enhance mode must be one of {MODES}")
    rep = EnhanceReport(mode=mode)
    x = np.asarray(x, dtype=np.float32)
    if mode == "off" or x.size == 0:
        return x, rep
    if mode == "high-fidelity":
        raise EnhancementUnavailable(
            "high-fidelity enhancement needs a neural enhancement/bandwidth-extension model; none is "
            "installed in aivoice. Use --enhance auto, or an engine with native high-rate output (VoxCPM2: 48 kHz)."
        )
    y = x
    if abs(float(np.mean(y))) > 1e-3:
        y = remove_dc(y)
        rep.applied.append("dc")
    else:
        rep.skipped["dc"] = "no measurable offset"
    clicks = detect_discontinuities(y, sr)
    rep.clicks_before = int(clicks.size)
    if clicks.size:
        fixed = repair_clicks(y, clicks, sr)
        after = detect_discontinuities(fixed, sr)
        if after.size < clicks.size:
            y = fixed
            rep.applied.append("declick")
            rep.clicks_after = int(after.size)
        else:
            rep.skipped["declick"] = "repair did not reduce detected discontinuities"
            rep.clicks_after = rep.clicks_before
    else:
        rep.skipped["declick"] = "none detected"
    norm, info = normalize(y, sr, target_lufs=target_lufs)
    rep.lufs_before = info["lufs_before"]
    if abs(info["gain_db"]) >= 0.5:
        y = norm
        rep.applied.append("loudness")
        rep.lufs_after = info["lufs_after"]
    else:
        rep.skipped["loudness"] = "already within 0.5 dB of target"
        rep.lufs_after = rep.lufs_before
    if encoder is not None and rep.applied:
        sim = encoder.similarity(x, sr, y, sr)
        rep.identity_similarity = sim
        if sim < min_identity:
            rep.rejected = f"speaker similarity to the unenhanced audio fell to {sim:.3f} (< {min_identity})"
            return x, rep
    return y, rep


def loudness_matched(parts: list[np.ndarray], sr: int, target_lufs: float = -20.0) -> list[np.ndarray]:
    """Bring segments to one loudness so long-form joins do not jump in level."""
    out = []
    for p in parts:
        if integrated_loudness(p, sr) == float("-inf"):
            out.append(np.asarray(p, np.float32))
        else:
            out.append(normalize(p, sr, target_lufs=target_lufs, max_gain_db=9.0)[0])
    return out
