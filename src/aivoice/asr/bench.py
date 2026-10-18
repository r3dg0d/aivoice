"""ASR measurement: word error rate and latency on identical clips for every backend."""

from __future__ import annotations

import resource
import statistics
import time
from typing import Iterable

import numpy as np

from ..audio.resample import resample
from ..stabilize import norm
from .base import ASRBackend, BenchmarkResult, streaming_view, to_mono_f32


def word_errors(ref: str, hyp: str) -> tuple[int, int]:
    """(edit distance in words, reference length) after case/punctuation normalisation."""
    r = [w for w in (norm(x) for x in ref.split()) if w]
    h = [w for w in (norm(x) for x in hyp.split()) if w]
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
            prev = cur
    return d[len(h)], len(r)


def wer(refs: list[str], hyps: list[str]) -> float | None:
    errs = total = 0
    for r, h in zip(refs, hyps):
        e, n = word_errors(r, h)
        errs += e
        total += n
    return errs / total if total else None


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarize(values: list[float]) -> dict[str, float | None]:
    return {
        "n": len(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "mean": statistics.fmean(values) if values else None,
        "max": max(values) if values else None,
    }


def _rss_mib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def run_asr_benchmark(
    backend: ASRBackend,
    clips: Iterable[tuple[np.ndarray, int, str | None]],
    *,
    chunk_ms: int = 80,
) -> BenchmarkResult:
    """Feed each clip in ``chunk_ms`` pieces as fast as possible.

    * final latency = time from the last chunk being handed over to the final text
      (what the speaker waits for after they stop talking);
    * first partial = compute time until the first partial appears;
    * RTF = total compute / total audio.
    """
    t_load = time.perf_counter()
    if not backend.loaded:
        backend.load_model()
    load_ms = (time.perf_counter() - t_load) * 1000
    view = streaming_view(backend)
    res = BenchmarkResult(backend=view.get_capabilities().name, clips=0, audio_s=0.0, load_ms=load_ms)
    refs, hyps = [], []
    compute = 0.0
    for audio, sr, ref in clips:
        x = to_mono_f32(audio)
        if sr != 16000:
            x = resample(x, sr, 16000)
        step = int(16000 * chunk_ms / 1000)
        view.reset_stream()
        first = None
        t0 = time.perf_counter()
        for i in range(0, x.size, step):
            ups = view.transcribe_stream(x[i : i + step], 16000)
            if ups and first is None:
                first = (time.perf_counter() - t0) * 1000
        t_end = time.perf_counter()
        final = view.finalize_utterance()
        t_fin = time.perf_counter()
        compute += t_fin - t0
        res.final_latency_ms.append((t_fin - t_end) * 1000)
        if first is not None:
            res.first_partial_ms.append(first)
        res.clips += 1
        res.audio_s += x.size / 16000
        res.hypotheses.append(final.text)
        if ref is not None:
            refs.append(ref)
            hyps.append(final.text)
    res.rtf = compute / res.audio_s if res.audio_s else None
    res.wer = wer(refs, hyps) if refs else None
    res.peak_rss_mib = _rss_mib()
    return res
