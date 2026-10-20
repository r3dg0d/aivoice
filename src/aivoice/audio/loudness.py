"""ITU-R BS.1770-4 integrated loudness (K-weighting + gating) and conservative normalisation.

Filters use Brecht De Man's analog-prototype derivation of the standard's
high-shelf / high-pass pair, which reproduces the ITU coefficients at 48 kHz and
works at any rate (matches pyloudnorm's ``filter_class="DeMan"``).
"""

from __future__ import annotations

import math

import numpy as np


def _biquad_shelf(sr: int) -> tuple[np.ndarray, np.ndarray]:
    """BS.1770 stage-1 high shelf via B. De Man's bilinear design (exact ITU coefficients at 48 kHz)."""
    G, Q, fc = 3.999843853973347, 0.7071752369554196, 1681.974450955533
    K = math.tan(math.pi * fc / sr)
    Vh = 10 ** (G / 20.0)
    Vb = Vh**0.4996667741545416
    a0 = 1.0 + K / Q + K * K
    b = np.array([(Vh + Vb * K / Q + K * K), 2.0 * (K * K - Vh), (Vh - Vb * K / Q + K * K)]) / a0
    a = np.array([1.0, 2.0 * (K * K - 1.0) / a0, (1.0 - K / Q + K * K) / a0])
    return b, a


def _biquad_highpass(sr: int) -> tuple[np.ndarray, np.ndarray]:
    """BS.1770 stage-2 RLB high-pass (De Man design)."""
    Q, fc = 0.5003270373238773, 38.13547087602444
    K = math.tan(math.pi * fc / sr)
    d = 1.0 + K / Q + K * K
    return np.array([1.0, -2.0, 1.0]), np.array([1.0, 2.0 * (K * K - 1.0) / d, (1.0 - K / Q + K * K) / d])


def lfilter(b: np.ndarray, a: np.ndarray, x: np.ndarray) -> np.ndarray:
    try:
        from scipy.signal import lfilter as sp_lfilter  # type: ignore

        return sp_lfilter(b, a, x)
    except ImportError:
        pass
    y = np.zeros_like(x, dtype=np.float64)
    x1 = x2 = y1 = y2 = 0.0
    b0, b1, b2 = b
    _, a1, a2 = a
    for i, xi in enumerate(x.astype(np.float64)):
        yi = b0 * xi + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        x2, x1, y2, y1 = x1, xi, y1, yi
        y[i] = yi
    return y


def k_weight(x: np.ndarray, sr: int) -> np.ndarray:
    b, a = _biquad_shelf(sr)
    y = lfilter(b, a, x)
    b, a = _biquad_highpass(sr)
    return lfilter(b, a, y)


def integrated_loudness(x: np.ndarray, sr: int) -> float:
    """LUFS (mono). Returns -inf for silence or clips shorter than one 400 ms block."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    block = int(0.4 * sr)
    if x.size < block:
        return float("-inf")
    y = k_weight(x, sr)
    # block count as in pyloudnorm / BS.1770-4: round((T - T_g) / step) + 1
    n_blocks = int(np.round((x.size / sr - 0.4) / 0.1)) + 1
    z = np.array([np.mean(y[int(0.1 * j * sr) : int(0.1 * j * sr) + block] ** 2) for j in range(n_blocks)])
    with np.errstate(divide="ignore"):
        l = -0.691 + 10 * np.log10(z)
    z = z[l > -70.0]
    if z.size == 0:
        return float("-inf")
    rel = -0.691 + 10 * np.log10(np.mean(z)) - 10.0
    with np.errstate(divide="ignore"):
        z = z[(-0.691 + 10 * np.log10(z)) > rel]
    if z.size == 0:
        return float("-inf")
    return float(-0.691 + 10 * np.log10(np.mean(z)))


def peak_dbfs(x: np.ndarray) -> float:
    p = float(np.max(np.abs(x))) if x.size else 0.0
    return 20 * math.log10(p) if p > 0 else float("-inf")


def normalize(
    x: np.ndarray,
    sr: int,
    *,
    target_lufs: float = -23.0,
    max_gain_db: float = 12.0,
    peak_ceiling_dbfs: float = -1.0,
) -> tuple[np.ndarray, dict[str, float]]:
    """Gain toward target loudness, bounded so it never boosts noise wildly or clips."""
    x = np.asarray(x, dtype=np.float32)
    before = integrated_loudness(x, sr)
    if not math.isfinite(before):
        return x, {"lufs_before": before, "gain_db": 0.0, "lufs_after": before}
    gain = max(-max_gain_db, min(max_gain_db, target_lufs - before))
    pk = peak_dbfs(x)
    if math.isfinite(pk):
        gain = min(gain, peak_ceiling_dbfs - pk)
    y = (x * 10 ** (gain / 20)).astype(np.float32)
    return y, {"lufs_before": before, "gain_db": gain, "lufs_after": integrated_loudness(y, sr)}
