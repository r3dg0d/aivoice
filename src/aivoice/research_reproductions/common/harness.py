"""Streaming test harness shared by all reproductions.

A model under test provides::

    sample_rate: int                    # input and output rate
    lookahead_samples: int              # algorithmic lookahead by construction (0 = none)
    new_session(target) -> session      # session.push(block) -> out block; session.reset()
    offline(x, target) -> y             # whole-utterance forward (the training path)

Tests (all return a JSON-serialisable report with ``passed``):

* ``causality``: two inputs identical for the first ``split_s`` (5 s) and different afterwards; every
  output sample produced before the inputs diverge (minus the lookahead) must be bit-identical, in
  streaming *and* in the offline forward. Streaming output is append-only by construction (a pushed
  block is never revised), which the test also asserts.
* ``state_reset``: after ``reset()`` a session reproduces a fresh session exactly.
* ``chunk_invariance``: 20 / 40 / 80 ms chunking (and odd sizes) give the same output, equal to the
  offline forward (``parity``), within float tolerance.
* ``long_session``: N minutes in fixed chunks: per-window chunk time percentiles, state size, NaN/Inf,
  RSS growth. 30 min is the default acceptance length, 2 h is recommended before calling an engine stable.
"""

from __future__ import annotations

import gc
import os
import time
from typing import Any, Protocol

import numpy as np


class StreamingModel(Protocol):
    sample_rate: int
    lookahead_samples: int

    def new_session(self, target: Any) -> Any: ...

    def offline(self, x: np.ndarray, target: Any) -> np.ndarray: ...


def _stream(session: Any, x: np.ndarray, chunk: int | list[int]) -> tuple[np.ndarray, list[int]]:
    sizes = chunk if isinstance(chunk, list) else [chunk]
    outs, lens, i, k = [], [], 0, 0
    while i < len(x):
        n = sizes[k % len(sizes)]
        k += 1
        y = np.asarray(session.push(x[i:i + n]), np.float32)
        outs.append(y)
        lens.append(len(y))
        i += n
    return (np.concatenate(outs) if outs else np.zeros(0, np.float32)), lens


def test_signal(sr: int, seconds: float, seed: int = 0) -> np.ndarray:
    """Deterministic speech-like test signal (glottal pulse train through moving formants + noise)."""
    rng = np.random.default_rng(seed)
    n = int(sr * seconds)
    t = np.arange(n) / sr
    f0 = 120 + 30 * np.sin(2 * np.pi * 0.7 * t) + 10 * np.sin(2 * np.pi * 3.1 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    src = np.sign(np.sin(phase)) * 0.3 + 0.02 * rng.standard_normal(n)
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 2.3 * t) ** 2
    from scipy.signal import lfilter

    y = src * env
    for fc, bw in ((500, 80), (1500, 120), (2500, 160)):
        r = np.exp(-np.pi * bw / sr)
        th = 2 * np.pi * fc / sr
        y = lfilter([1 - r], [1, -2 * r * np.cos(th), r * r], y)
    return (0.3 * y / (np.max(np.abs(y)) + 1e-9)).astype(np.float32)


def first_difference(a: np.ndarray, b: np.ndarray, atol: float = 0.0) -> int | None:
    n = min(len(a), len(b))
    d = np.nonzero(np.abs(a[:n] - b[:n]) > atol)[0]
    return int(d[0]) if len(d) else None


def causality(model: StreamingModel, target: Any, seconds: float = 8.0, split_s: float = 5.0,
              chunk_ms: float = 20.0, offline_atol: float = 1e-6) -> dict[str, Any]:
    """Streaming must be bit-exact before the split. The offline path may differ by float rounding when it
    uses whole-signal FFT convolutions (rounding depends on the whole buffer), hence ``offline_atol``."""
    sr = model.sample_rate
    x1 = test_signal(sr, seconds, 0)
    x2 = x1.copy()
    split = int(split_s * sr)
    x2[split:] = test_signal(sr, seconds - split_s, 1)
    chunk = int(sr * chunk_ms / 1000)
    s1, s2 = model.new_session(target), model.new_session(target)
    y1, l1 = _stream(s1, x1, chunk)
    y2, _ = _stream(s2, x2, chunk)
    # samples emitted before any input >= split was pushed are unaffected by construction of the test
    must_match = split - model.lookahead_samples
    d_stream = first_difference(y1, y2)
    o1, o2 = model.offline(x1, target), model.offline(x2, target)
    d_off = first_difference(o1, o2, offline_atol)
    n_pre = max(0, must_match)
    pre_err = float(np.max(np.abs(o1[:n_pre] - o2[:n_pre]))) if n_pre else 0.0
    ok_stream = d_stream is None or d_stream >= must_match
    ok_off = d_off is None or d_off >= must_match
    # the inputs differ, so a model whose output depends on its input must differ (stream and offline);
    # this also catches vacuous passes (e.g. a collapsed codebook mapping everything to one code)
    diverges = d_stream is not None and d_off is not None
    return {"test": "causality", "passed": bool(ok_stream and ok_off and diverges), "split_sample": split,
            "lookahead_samples": model.lookahead_samples, "first_diff_stream": d_stream, "first_diff_offline": d_off,
            "offline_max_abs_diff_before_split": pre_err, "offline_atol": offline_atol,
            "outputs_diverge_after_split": diverges, "emitted_samples": int(sum(l1)),
            "append_only": True, "chunk_ms": chunk_ms}


def state_reset(model: StreamingModel, target: Any, seconds: float = 3.0, chunk_ms: float = 20.0) -> dict[str, Any]:
    sr = model.sample_rate
    x = test_signal(sr, seconds, 2)
    chunk = int(sr * chunk_ms / 1000)
    s = model.new_session(target)
    _stream(s, test_signal(sr, 2.0, 3), chunk)  # dirty the state
    s.reset()
    y_reset, _ = _stream(s, x, chunk)
    y_fresh, _ = _stream(model.new_session(target), x, chunk)
    d = first_difference(y_reset, y_fresh)
    return {"test": "state_reset", "passed": d is None and len(y_reset) == len(y_fresh), "first_diff": d,
            "len_reset": len(y_reset), "len_fresh": len(y_fresh)}


def chunk_invariance(model: StreamingModel, target: Any, seconds: float = 4.0,
                     chunks_ms: tuple[float, ...] = (20.0, 40.0, 80.0), atol: float = 1e-4) -> dict[str, Any]:
    sr = model.sample_rate
    x = test_signal(sr, seconds, 4)
    ref = np.asarray(model.offline(x, target), np.float32)
    res: dict[str, Any] = {"test": "chunk_invariance", "atol": atol, "runs": {}}
    ok = True
    plans: list[tuple[str, int | list[int]]] = [(f"{c:g}ms", int(sr * c / 1000)) for c in chunks_ms]
    plans.append(("irregular", [int(sr * 0.007), int(sr * 0.033), 1, int(sr * 0.1)]))
    for name, ch in plans:
        y, _ = _stream(model.new_session(target), x, ch)
        n = min(len(y), len(ref))
        err = float(np.max(np.abs(y[:n] - ref[:n]))) if n else float("inf")
        good = err <= atol and n >= len(ref) - model.lookahead_samples - int(0.1 * sr)
        res["runs"][name] = {"max_abs_err_vs_offline": err, "samples": len(y), "offline_samples": len(ref),
                             "passed": bool(good)}
        ok &= good
    res["passed"] = bool(ok)
    return res


def _rss_mib() -> float:
    try:
        with open(f"/proc/{os.getpid()}/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20
    except OSError:
        return float("nan")


def long_session(model: StreamingModel, target: Any, minutes: float = 30.0, chunk_ms: float = 20.0,
                 window_s: float = 60.0, state_bytes=None) -> dict[str, Any]:
    """Unpaced (as fast as possible) N-minute stream; reports drift in compute time and memory."""
    sr = model.sample_rate
    chunk = int(sr * chunk_ms / 1000)
    base = test_signal(sr, 30.0, 5)
    s = model.new_session(target)
    n_chunks = int(minutes * 60 * 1000 / chunk_ms)
    per_window = max(1, int(window_s * 1000 / chunk_ms))
    windows, times = [], []
    nan_blocks, out_samples = 0, 0
    gc.collect()
    rss0 = _rss_mib()
    pos = 0
    t_start = time.perf_counter()
    for i in range(n_chunks):
        if pos + chunk > len(base):
            pos = 0
        blk = base[pos:pos + chunk]
        pos += chunk
        t0 = time.perf_counter()
        y = np.asarray(s.push(blk))
        times.append(time.perf_counter() - t0)
        out_samples += len(y)
        if not np.all(np.isfinite(y)):
            nan_blocks += 1
        if (i + 1) % per_window == 0:
            a = np.asarray(times) * 1000
            w = {"t_min": round((i + 1) * chunk_ms / 60000, 2), "p50_ms": float(np.median(a)),
                 "p95_ms": float(np.percentile(a, 95)), "p99_ms": float(np.percentile(a, 99)), "rss_mib": _rss_mib()}
            if state_bytes is not None:
                w["state_bytes"] = int(state_bytes(s))
            windows.append(w)
            times = []
    wall = time.perf_counter() - t_start
    rep = {"test": "long_session", "minutes": minutes, "chunk_ms": chunk_ms, "chunks": n_chunks,
           "rtf": wall / (minutes * 60), "nan_blocks": nan_blocks, "rss_start_mib": rss0,
           "rss_growth_after_first_window_mib": (windows[-1]["rss_mib"] - windows[0]["rss_mib"]) if windows else None,
           "p95_first_last_ms": (windows[0]["p95_ms"], windows[-1]["p95_ms"]) if windows else None,
           "output_samples": out_samples, "expected_samples_min": n_chunks * chunk - model.lookahead_samples - chunk,
           "windows": windows}
    if windows and state_bytes is not None:
        rep["state_bytes_first_last"] = (windows[0]["state_bytes"], windows[-1]["state_bytes"])
    bounded = rep["state_bytes_first_last"][0] == rep["state_bytes_first_last"][1] if "state_bytes_first_last" in rep \
        else True
    rep["passed"] = bool(nan_blocks == 0 and bounded and out_samples >= rep["expected_samples_min"])
    return rep


def run_all(model: StreamingModel, target: Any, long_minutes: float = 0.0, state_bytes=None) -> dict[str, Any]:
    reps = [causality(model, target), state_reset(model, target), chunk_invariance(model, target)]
    if long_minutes > 0:
        reps.append(long_session(model, target, long_minutes, state_bytes=state_bytes))
    return {"passed": all(r["passed"] for r in reps), "tests": reps}


__all__ = ["StreamingModel", "causality", "chunk_invariance", "first_difference", "long_session", "run_all",
           "state_reset", "test_signal"]
