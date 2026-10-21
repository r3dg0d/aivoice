"""Long-session streaming stability (experiment O).

Streams N minutes of looped rights-free speech (with silences) through the live per-block path and
records, per window: block-processing p50/p95/max, resident memory, output backlog, NaN blocks,
fallbacks, output level. Two modes:

* ``paced=False`` - as fast as the CPU allows (tests state growth / leaks over many blocks);
* ``paced=True``  - real-time device clock: :class:`LiveRunner` with a real-time ``ArrayCapture``
  and ``ClockDriver``, so capture overflows, playback underruns and clock drift are real.
"""

from __future__ import annotations

import json
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np


def _rss_mib() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _material(sr: int) -> np.ndarray:
    from ..audio.resample import resample
    from ..evaluation import synthetic_data as sd

    parts = []
    for i, spk in enumerate(sd.SPEAKERS):
        x, s = sd.speak(" ".join(sd.HARVARD[i * 3 : i * 3 + 3]), spk)
        parts += [resample(x, s, sr), np.zeros(int(sr * (0.4 + 0.3 * i)), np.float32)]
    return np.concatenate(parts).astype(np.float32)


def run_long_session(minutes: float, engine_factory, *, label: str, paced: bool = False, window_s: float = 60.0,
                     post_preset: str | None = None, post_strength: float = 0.0, out: Path | None = None) -> dict[str, Any]:
    from .live import PIPE_SR, LiveConfig, LivePipeline

    eng = engine_factory()
    cfg = LiveConfig(mode="modulation" if eng.capabilities().task == "modulation" else "voice_conversion",
                     engine=eng.capabilities().name, post_preset=post_preset, post_strength=post_strength, fallback=["dsp"])
    mat = _material(PIPE_SR)
    total = int(minutes * 60 * PIPE_SR)
    block = int(PIPE_SR * cfg.block_ms / 1000)
    windows: list[dict[str, Any]] = []
    t_start = time.perf_counter()
    rss0 = _rss_mib()
    if not paced:
        pipe = LivePipeline(cfg, eng)
        pipe.start()
        pos, w_proc, w_level, w_t0 = 0, [], [], time.perf_counter()
        next_w = int(window_s * PIPE_SR)
        while pos < total:
            i = pos % mat.size
            b = mat[i : i + block]
            if b.size < block:
                b = np.concatenate([b, mat[: block - b.size]])
            t0 = time.perf_counter()
            y = pipe.feed(b)
            w_proc.append((time.perf_counter() - t0) * 1000)
            w_level.append(float(np.sqrt(np.mean(y * y))))
            pos += block
            if pos >= next_w:
                a = np.asarray(w_proc)
                windows.append({"t_audio_min": round(pos / PIPE_SR / 60, 2), "block_p50_ms": round(float(np.percentile(a, 50)), 3),
                                "block_p95_ms": round(float(np.percentile(a, 95)), 3), "block_max_ms": round(float(a.max()), 3),
                                "rss_mib": round(_rss_mib(), 1), "backlog_samples": int(pipe._fifo.size),
                                "nan_blocks": pipe.stats.nan_blocks, "fallbacks": len(pipe.stats.fallbacks),
                                "out_rms": round(float(np.mean(w_level)), 5), "wall_s": round(time.perf_counter() - w_t0, 2)})
                w_proc, w_level, w_t0 = [], [], time.perf_counter()
                next_w += int(window_s * PIPE_SR)
        engine_ms = pipe.stats.window(10**9)
        eng.unload()
        extra = {"engine_ms_all": engine_ms}
    else:
        from ..audio.capture import ArrayCapture
        from ..audio.playback import ClockDriver
        from .live import LiveRunner

        reps = int(np.ceil(total / mat.size))
        audio = np.tile(mat, reps)[:total]
        cap = ArrayCapture(audio, PIPE_SR, target_rate=PIPE_SR, block_ms=int(cfg.block_ms), realtime=True, tail_silence_ms=200)
        pipe = LivePipeline(cfg, eng)
        runner = LiveRunner(pipe, capture=cap, driver_kind="clock")
        runner.start()
        runner.driver.stop()  # replace the auto driver with a real-time clock that keeps nothing
        runner.driver = ClockDriver(runner.player, block_ms=10, realtime=True, keep=False)
        runner.driver.start()
        t_next = time.monotonic() + window_s
        while not runner._stop.is_set():
            runner._stop.wait(1.0)
            if time.monotonic() >= t_next or runner._stop.is_set():
                t_next += window_s
                w = pipe.stats.window(int(window_s * 1000 / cfg.block_ms))
                windows.append({"t_wall_min": round((time.perf_counter() - t_start) / 60, 2),
                                "block_p50_ms": w["block_ms"].get("p50"), "block_p95_ms": w["block_ms"].get("p95"),
                                "block_max_ms": w["block_ms"].get("max"), "rss_mib": round(_rss_mib(), 1),
                                "underruns": runner.player.underruns, "dropped_blocks": cap.dropped,
                                "output_buffered_ms": round(runner.player.buffered_ms(), 1),
                                "trimmed_ms": round(getattr(runner.player, "trimmed", 0) * 1000 / PIPE_SR, 1), "backlog_samples": int(pipe._fifo.size),
                                "nan_blocks": pipe.stats.nan_blocks, "fallbacks": len(pipe.stats.fallbacks)})
        runner.stop()
        extra = {"underruns": runner.player.underruns, "dropped_blocks": cap.dropped,
                 "trimmed_ms": round(getattr(runner.player, "trimmed", 0) * 1000 / PIPE_SR, 1)}
    rss = [w["rss_mib"] for w in windows]
    p95 = [w["block_p95_ms"] for w in windows if w.get("block_p95_ms") is not None]
    res = {
        "label": label, "paced": paced, "minutes_audio": minutes, "wall_s": round(time.perf_counter() - t_start, 1),
        "windows": windows, "rss_start_mib": round(rss0, 1), "rss_end_mib": rss[-1] if rss else None,
        "rss_growth_mib_after_first_window": round(rss[-1] - rss[0], 1) if len(rss) > 1 else None,
        "block_p95_first_last_ms": [p95[0], p95[-1]] if p95 else None,
        "latency_drift_p95_ms": round(p95[-1] - p95[0], 3) if len(p95) > 1 else None,
        "nan_blocks": windows[-1]["nan_blocks"] if windows else None, "fallbacks": windows[-1]["fallbacks"] if windows else None,
        **extra,
    }
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=2) + "\n")
    return res


__all__ = ["run_long_session"]
