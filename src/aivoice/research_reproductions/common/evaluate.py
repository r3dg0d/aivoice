"""Evaluation of a reproduction: streaming-harness tests plus conversion metrics (all MEASURED BY AIVOICE).

Conversion metrics, for converted audio given the source and target references:

* target similarity: GE2E cosine to the target reference (the speaker encoder aivoice already ships);
* source leakage: GE2E cosine to the *source* speaker (lower is better: timbre that leaked through);
* F0: log-F0 Pearson correlation with the source (intonation kept), voicing agreement, median shift
  in semitones relative to the target median (register moved to the target);
* clicks / F0 jumps from the existing voice metrics.

GE2E similarity is evidence, not proof of identity, and none of these replaces listening.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .provenance import MEASURED


def f0_metrics(converted: np.ndarray, source: np.ndarray, target: np.ndarray, sr: int) -> dict[str, Any]:
    from aivoice.evaluation.voice_metrics import f0_stats, f0_track

    fc, fs = f0_track(converted, sr), f0_track(source, sr)
    n = min(len(fc), len(fs))
    fc, fs = fc[:n], fs[:n]
    both = (fc > 0) & (fs > 0)
    corr = float(np.corrcoef(np.log(fc[both]), np.log(fs[both]))[0, 1]) if both.sum() > 10 else None
    vda = float(np.mean((fc > 0) == (fs > 0))) if n else None
    tm = f0_stats(target, sr)["f0_median_hz"]
    cm = float(np.median(fc[fc > 0])) if (fc > 0).sum() > 5 else None
    st = 12 * np.log2(cm / tm) if cm and tm else None
    return {"logf0_corr_with_source": None if corr is None else round(corr, 4),
            "voicing_agreement": None if vda is None else round(vda, 4),
            "median_offset_from_target_st": None if st is None else round(float(st), 2),
            "converted": f0_stats(converted, sr)}


def conversion_metrics(converted: np.ndarray, source: np.ndarray, target: np.ndarray, sr: int) -> dict[str, Any]:
    from aivoice.evaluation.metrics import speaker_similarity

    return {"label": MEASURED,
            "target_similarity": speaker_similarity(converted, sr, target, sr),
            "source_leakage": speaker_similarity(converted, sr, source, sr),
            "f0": f0_metrics(converted, source, target, sr)}


def streaming_report(model: Any, target: Any, long_minutes: float = 0.0) -> dict[str, Any]:
    from . import harness

    t0 = time.time()
    # exactness tests check the streaming algorithm, so run them in full fp32: cuDNN convolutions default to
    # TF32 on Ampere+, whose ~1e-4 rounding differs between chunk shapes and failed the 1e-4 chunk-invariance
    # tolerance on GPU while the same checkpoint was exact (3e-7) on CPU. Tolerances are unchanged.
    import torch

    prev = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    try:
        rep = harness.run_all(model, target, long_minutes,
                              state_bytes=lambda s: s.state_bytes() if hasattr(s, "state_bytes") else 0)
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = prev
    rep["tf32"] = False
    rep["label"] = MEASURED
    rep["seconds"] = round(time.time() - t0, 1)
    return rep


def compute_report(model: Any, target: Any, seconds: float = 10.0, chunk_ms: float = 20.0) -> dict[str, Any]:
    """Real-time factor and per-chunk latency of the streaming path on this machine."""
    from .harness import test_signal

    sr = model.sample_rate
    x = test_signal(sr, seconds, 11)
    n = int(sr * chunk_ms / 1000)
    s = model.new_session(target)
    times = []
    for i in range(0, len(x), n):
        t = time.perf_counter()
        s.push(x[i:i + n])
        times.append(time.perf_counter() - t)
    a = np.asarray(times[5:] or times) * 1000
    return {"label": MEASURED, "chunk_ms": chunk_ms, "rtf": float(sum(times) / seconds), "chunk_p50_ms": float(np.median(a)),
            "chunk_p95_ms": float(np.percentile(a, 95)), "algorithmic_lookahead_ms": 1000 * model.lookahead_samples / sr}


__all__ = ["compute_report", "conversion_metrics", "f0_metrics", "streaming_report"]
