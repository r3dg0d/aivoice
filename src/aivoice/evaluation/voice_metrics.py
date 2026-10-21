"""Acoustic voice-attribute measurements used to evaluate modulation (and VC pitch/timbre behaviour).

* F0 statistics: median, 10-90 % range in semitones, voiced fraction, frame-to-frame jump rate
  (pitch continuity) - YIN-style per frame.
* Formants F1-F3 from LPC roots (autocorrelation method, Levinson-Durbin), median over voiced
  frames; formant dispersion and the apparent vocal-tract length it implies
  (``VTL = c / (2 * dispersion)``, Fitch 1997).
* Spectral centroid and long-term spectral tilt (dB/octave, 0.3-5 kHz regression).

These are descriptors, not perception: "lower formants" is measured, "sounds more masculine" is
not. LPC formant tracking is approximate (errors of 5-10 % are normal on real speech).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..audio.resample import resample
from ..modulation.dsp import _yin_frame

SPEED_OF_SOUND_CM_S = 35000.0


def f0_track(x: np.ndarray, sr: int, *, frame_ms: float = 40.0, hop_ms: float = 10.0,
             fmin: float = 60.0, fmax: float = 600.0) -> np.ndarray:
    x = np.asarray(x, np.float64)
    n = int(sr * frame_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    out = []
    for s in range(0, max(0, x.size - n), hop):
        f, ap = _yin_frame(x[s : s + n], sr, fmin, fmax)
        out.append(f if (f > 0 and ap < 0.25) else 0.0)
    return np.array(out)


def f0_stats(x: np.ndarray, sr: int) -> dict[str, Any]:
    f = f0_track(x, sr)
    v = f[f > 0]
    if v.size < 5:
        return {"f0_median_hz": None, "f0_range_st": None, "voiced_fraction": float(v.size / max(1, f.size)),
                "f0_jumps_per_s": None}
    cents = 1200 * np.log2(v)
    both = (f[1:] > 0) & (f[:-1] > 0)
    jumps = np.abs(1200 * np.log2(f[1:][both] / f[:-1][both])) > 300  # > 3 semitones in 10 ms
    return {
        "f0_median_hz": round(float(np.median(v)), 2),
        "f0_range_st": round(float((np.percentile(cents, 90) - np.percentile(cents, 10)) / 100), 2),
        "voiced_fraction": round(float(v.size / f.size), 3),
        "f0_jumps_per_s": round(float(jumps.sum() / (f.size * 0.01)), 3),
    }


def _lpc(frame: np.ndarray, order: int) -> np.ndarray | None:
    r = np.correlate(frame, frame, "full")[frame.size - 1 : frame.size + order]
    if r[0] <= 1e-12:
        return None
    a = np.zeros(order + 1)
    a[0] = 1.0
    e = r[0]
    for i in range(1, order + 1):
        k = -(r[i] + np.dot(a[1:i], r[i - 1 : 0 : -1])) / e
        a[1:i] = a[1:i] + k * a[i - 1 : 0 : -1]
        a[i] = k
        e *= 1 - k * k
        if e <= 0:
            return None
    return a


def formants(x: np.ndarray, sr: int, *, n_formants: int = 3) -> dict[str, Any]:
    """Median F1..Fn (Hz) over voiced frames, by LPC at 10 kHz (order 12)."""
    fs = 10000
    w = resample(np.asarray(x, np.float32), sr, fs).astype(np.float64)
    if w.size < fs // 5:
        return {"formants_hz": None}
    w = np.append(w[0], w[1:] - 0.97 * w[:-1])  # pre-emphasis
    n, hop = int(0.025 * fs), int(0.010 * fs)
    f0 = f0_track(resample(np.asarray(x, np.float32), sr, 16000), 16000, frame_ms=25, hop_ms=10)
    win = np.hamming(n)
    tracks: list[list[float]] = []
    for i, s in enumerate(range(0, w.size - n, hop)):
        if i >= f0.size or f0[i] <= 0:
            continue
        a = _lpc(w[s : s + n] * win, 12)
        if a is None:
            continue
        roots = np.roots(a)
        roots = roots[np.imag(roots) > 0.01]
        ang = np.angle(roots) * fs / (2 * np.pi)
        bw = -0.5 * fs / np.pi * np.log(np.abs(roots))
        cand = sorted(f for f, b in zip(ang, bw) if 150 < f < 4800 and b < 500)
        if len(cand) >= n_formants:
            tracks.append(cand[:n_formants])
    if len(tracks) < 5:
        return {"formants_hz": None}
    arr = np.array(tracks)
    med = np.median(arr, axis=0)
    disp = float(np.mean(np.diff(np.concatenate([[0.0], med])))) if n_formants else None
    # dispersion over F1..Fn: (Fn - F1) / (n - 1) is the usual definition
    disp = float((med[-1] - med[0]) / (n_formants - 1)) if n_formants > 1 else None
    return {
        "formants_hz": [round(float(v), 1) for v in med],
        "formant_dispersion_hz": round(disp, 1) if disp else None,
        "apparent_vtl_cm": round(SPEED_OF_SOUND_CM_S / (2 * disp), 2) if disp else None,
        "frames": int(arr.shape[0]),
    }


def spectral_shape(x: np.ndarray, sr: int) -> dict[str, Any]:
    x = np.asarray(x, np.float64)
    n = 2048
    if x.size < n:
        return {"spectral_centroid_hz": None, "spectral_tilt_db_per_oct": None}
    frames = np.lib.stride_tricks.sliding_window_view(x, n)[:: n // 2] * np.hanning(n)
    p = np.mean(np.abs(np.fft.rfft(frames, axis=1)) ** 2, axis=0)
    f = np.fft.rfftfreq(n, 1 / sr)
    centroid = float(np.sum(f * p) / max(np.sum(p), 1e-20))
    band = (f >= 300) & (f <= 5000)
    slope = float(np.polyfit(np.log2(f[band]), 10 * np.log10(p[band] + 1e-20), 1)[0]) if band.sum() > 4 else None
    return {"spectral_centroid_hz": round(centroid, 1), "spectral_tilt_db_per_oct": round(slope, 2) if slope is not None else None}


def voice_profile(x: np.ndarray, sr: int) -> dict[str, Any]:
    return {**f0_stats(x, sr), **formants(x, sr), **spectral_shape(x, sr)}


def semitones(a: float | None, b: float | None) -> float | None:
    if not a or not b:
        return None
    return round(12 * math.log2(b / a), 2)


__all__ = ["f0_stats", "f0_track", "formants", "semitones", "spectral_shape", "voice_profile"]
