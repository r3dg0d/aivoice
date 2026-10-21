"""`aivoice benchmark vc`: every live engine on the same sources, targets and machine.

Each source is streamed through exactly the live per-block path (:meth:`LivePipeline.run_array`:
48 kHz blocks, resamplers, engine chunks), so timing and artifacts are those of live use.

Per engine x target x source:
  * timing: per-chunk compute p50/p95/p99, RTF, declared algorithmic latency, *measured* delay
    (energy-envelope cross-correlation of output vs input), CPU seconds, RSS, worker VRAM peak
  * identity: GE2E cosine to the target reference (target similarity), to the source
    (source-speaker leakage) and the source-target baseline
  * content: WER/CER when an ASR backend is installed (else reported as not measured)
  * artifacts: discontinuity (click) count, F0 jumps per second, NaN blocks, dropouts
Listening files are loudness-matched to -23 LUFS so no engine wins by being louder.
"""

from __future__ import annotations

import csv
import json
import platform
import resource
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..audio.loudness import integrated_loudness, normalize
from ..audio.resample import resample
from .base import TASK_MOD, StreamOptions, VoiceTarget
from .live import PIPE_SR, LiveConfig, LivePipeline


@dataclass
class Clip:
    name: str
    audio: np.ndarray
    sr: int
    text: str | None = None


def measured_delay_ms(src: np.ndarray, out: np.ndarray, sr: int, *, max_ms: float = 2000.0) -> float | None:
    """Delay of *out* relative to *src* from 10 ms log-energy envelopes (works when the voice changes)."""
    hop = sr // 100
    def env(x):
        n = x.size // hop
        if n < 10:
            return None
        e = np.log(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-8)
        return (e - e.mean()) / (e.std() + 1e-8)
    a, b = env(src), env(out)
    if a is None or b is None:
        return None
    lags = range(0, min(int(max_ms / 10), b.size - 5))
    best, score = 0, -np.inf
    for lag in lags:
        n = min(a.size, b.size - lag)
        if n < 20:
            break
        c = float(np.dot(a[:n], b[lag : lag + n]) / n)
        if c > score:
            best, score = lag, c
    if score < 0.3:
        return None
    # refine at sample resolution around the envelope lag
    return best * 10.0


def _cpu_s() -> float:
    r, c = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return r.ru_utime + r.ru_stime + c.ru_utime + c.ru_stime


def synthetic_clips(n_sources: int = 2, n_targets: int = 2) -> tuple[list[Clip], list[Clip]]:
    """Rights-free espeak-ng speakers: sources and targets are different synthetic voices."""
    from ..evaluation import synthetic_data as sd

    spk = list(sd.SPEAKERS)
    srcs, tgts = [], []
    for s in spk[:n_sources]:
        text = " ".join(sd.HARVARD[4:7])
        x, sr = sd.speak(text, s)
        srcs.append(Clip(s, x, sr, text))
    for s in spk[2 : 2 + n_targets] or spk[:n_targets]:
        x, sr = sd.speak(" ".join(sd.HARVARD[:3]), s)
        tgts.append(Clip(s, x, sr, None))
    return srcs, tgts


def run_vc_benchmark(engines: list[str], sources: list[Clip], targets: list[VoiceTarget], out_dir: Path, *,
                     quality: str = "balanced", modulation_preset: str = "masculine-natural", asr: Any | None = None,
                     record_results: bool = True, label: str = "synthetic speech; imitates no real person") -> dict[str, Any]:
    from ..audio.enhancement import detect_discontinuities
    from ..audio_io import read_wav
    from ..evaluation.metrics import char_error_rate, word_error_rate, write_labeled_wav
    from ..evaluation.voice_metrics import f0_stats
    from ..research.speaker_identity import get_encoder
    from .engines import DspEngine, get_vc_engine
    from .router import record

    enc = get_encoder()
    stamp = time.strftime("%Y-%m-%d_%H%M%S")
    root = out_dir / stamp
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    skipped: dict[str, str] = {}
    manifest: dict[str, Any] = {"created": stamp, "machine": platform.node(), "quality": quality, "label": label,
                                "loudness_lufs": -23.0, "files": []}

    def save(path: Path, x: np.ndarray, sr: int, what: str) -> str:
        info: dict[str, float] = {}
        y = x
        if x.size and np.isfinite(integrated_loudness(x, sr)):
            y, info = normalize(x, sr, target_lufs=-23.0, max_gain_db=30.0)
        write_labeled_wav(path, np.clip(y, -1, 1), sr, comment=f"aivoice benchmark: {what}; {label}")
        manifest["files"].append({"path": str(path.relative_to(root)), "what": what,
                                  "lufs": round(info.get("lufs_after", float("nan")), 2) if info else None})
        return str(path.relative_to(root))

    tclips: dict[str, tuple[np.ndarray, int]] = {}
    for t in targets:
        if t.reference:
            a, sr = read_wav(t.reference)
            tclips[t.name] = (a, sr)
            save(root / "targets" / f"{t.name}.wav", a, sr, f"target reference {t.name}")
    for s in sources:
        save(root / "sources" / f"{s.name}.wav", s.audio, s.sr, f"source {s.name}")

    for name in engines:
        try:
            probe = get_vc_engine(name)
            st = probe.status()
        except Exception as e:  # noqa: BLE001
            skipped[name] = str(e)
            continue
        if not st.state.runnable:
            skipped[name] = f"{st.state.value}: {st.detail}"
            continue
        is_mod = st.caps.task == TASK_MOD
        for t in ([VoiceTarget("self")] if is_mod else targets):
            for s in sources:
                if is_mod and name == "dsp":
                    from ..modulation.presets import get_preset

                    eng = DspEngine(get_preset(modulation_preset).params())
                else:
                    eng = get_vc_engine(name)
                opts = StreamOptions(quality=quality)
                eng.options = opts
                cfg = LiveConfig(mode="modulation" if is_mod else "voice_conversion", engine=name, quality=quality,
                                 fallback=[], stream_options=opts)
                row: dict[str, Any] = {"engine": name, "task": st.caps.task, "target": t.name, "source": s.name,
                                       "quality": quality, "state": st.state.value}
                cpu0, t0 = _cpu_s(), time.perf_counter()
                try:
                    pipe = LivePipeline(cfg, eng, target=t)
                    pipe.start()
                    load_s = time.perf_counter() - t0
                    t1 = time.perf_counter()
                    y = pipe.run_array(s.audio, s.sr)
                    wall = time.perf_counter() - t1
                    caps = eng.capabilities()
                    worker = eng.stats() if hasattr(eng, "stats") and callable(getattr(eng, "stats")) else {}
                    timing = eng.timing_summary()
                    eng.unload()
                except Exception as e:  # noqa: BLE001
                    row["error"] = f"{type(e).__name__}: {e}"[:500]
                    rows.append(row)
                    try:
                        eng.unload()
                    except Exception:  # noqa: BLE001
                        pass
                    continue
                if pipe.stats.fallbacks:
                    row["error"] = f"engine failed during the run: {pipe.stats.fallbacks[0]['reason']}"
                    rows.append(row)
                    continue
                src48 = resample(s.audio, s.sr, PIPE_SR) if s.sr != PIPE_SR else s.audio
                dur = src48.size / PIPE_SR
                d = measured_delay_ms(src48, y, PIPE_SR)
                trimmed = y[int((d or 0) * PIPE_SR / 1000) :][: src48.size]
                row.update({
                    "load_s": round(load_s, 2), "audio_s": round(dur, 2), "wall_s": round(wall, 3),
                    "rtf": round(eng.total_ms / 1000 / dur, 4) if eng.total_chunks else None,
                    "pipeline_rtf": round(sum(pipe.stats.proc_ms) / 1000 / dur, 4),
                    "chunk_p50_ms": timing.get("p50_ms"), "chunk_p95_ms": timing.get("p95_ms"),
                    "chunk_p99_ms": timing.get("p99_ms"), "chunk_max_ms": timing.get("max_ms"), "chunks": timing.get("chunks"),
                    "block_p95_ms": pipe.stats.window()["block_ms"].get("p95"),
                    "algorithmic_latency_ms": caps.algorithmic_latency_ms,
                    "pipeline_extra_latency_ms": round(pipe._pipeline_extra_ms(), 2), "measured_delay_ms": d,
                    "cpu_s": round(_cpu_s() - cpu0, 3), "rss_max_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
                    "vram_peak_mib": worker.get("vram_peak_mib"), "gpu": worker.get("gpu"),
                    "clicks": int(detect_discontinuities(trimmed, PIPE_SR).size),
                    "f0_jumps_per_s": f0_stats(trimmed, PIPE_SR).get("f0_jumps_per_s"),
                    "nan_blocks": pipe.stats.nan_blocks,
                    "silent_output": bool(np.sqrt(np.mean(trimmed**2)) < 1e-4) if trimmed.size else True,
                })
                if enc is not None and trimmed.size > PIPE_SR // 2:
                    try:
                        row["source_similarity"] = round(enc.similarity(trimmed, PIPE_SR, s.audio, s.sr), 4)
                        if not is_mod and t.name in tclips:
                            ta, tsr = tclips[t.name]
                            row["target_similarity"] = round(enc.similarity(trimmed, PIPE_SR, ta, tsr), 4)
                            row["source_target_baseline"] = round(enc.similarity(s.audio, s.sr, ta, tsr), 4)
                        row["similarity_encoder"] = enc.name
                    except ValueError as e:
                        row["similarity_error"] = str(e)
                if asr is not None and s.text:
                    hyp = asr.transcribe(resample(trimmed, PIPE_SR, 16000), 16000).text
                    row.update({"wer": word_error_rate(s.text, hyp), "cer": char_error_rate(s.text, hyp), "asr": asr.get_capabilities().name})
                row["file"] = save(root / name / f"{t.name}__{s.name}.wav", trimmed, PIPE_SR,
                                   f"{name} ({quality}) {s.name} -> {t.name}")
                rows.append(row)

    summary: dict[str, dict[str, Any]] = {}
    for name in {r["engine"] for r in rows if "error" not in r}:
        rs = [r for r in rows if r["engine"] == name and "error" not in r]

        def mean(k):
            v = [r[k] for r in rs if isinstance(r.get(k), (int, float))]
            return round(float(np.mean(v)), 4) if v else None

        def worst(k):
            v = [r[k] for r in rs if isinstance(r.get(k), (int, float))]
            return round(float(np.max(v)), 3) if v else None

        summary[name] = {"task": rs[0]["task"], "n": len(rs), "rtf": mean("rtf"), "chunk_p50_ms": mean("chunk_p50_ms"),
                         "chunk_p95_ms": worst("chunk_p95_ms"), "chunk_p99_ms": worst("chunk_p99_ms"),
                         "algorithmic_latency_ms": mean("algorithmic_latency_ms"), "measured_delay_ms": mean("measured_delay_ms"),
                         "target_similarity": mean("target_similarity"), "source_similarity": mean("source_similarity"),
                         "source_target_baseline": mean("source_target_baseline"), "wer": mean("wer"), "cer": mean("cer"),
                         "clicks": mean("clicks"), "f0_jumps_per_s": mean("f0_jumps_per_s"), "vram_peak_mib": worst("vram_peak_mib"),
                         "cpu_s": mean("cpu_s")}
        if record_results and rs[0]["task"] != TASK_MOD:
            record(name, quality, summary[name])
    result = {"summary": summary, "rows": rows, "skipped": skipped, "out_dir": str(root),
              "notes": ["GE2E cosine is similarity under one encoder, not proof of identity",
                        "WER/CER need an installed ASR backend" if asr is None else "ASR WER is an intelligibility proxy",
                        "measured_delay_ms: energy-envelope alignment at 10 ms resolution (excludes device/PipeWire buffers)"]}
    (root / "results.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    keys = sorted({k for r in rows for k in r})
    with open(root / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return result


def format_summary(result: dict[str, Any]) -> str:
    cols = [("engine", 15), ("task", 17), ("rtf", 7), ("p50", 7), ("p95", 7), ("p99", 7), ("alg ms", 7), ("delay", 7),
            ("tgt sim", 8), ("src sim", 8), ("wer", 6), ("clicks", 7)]
    lines = [" ".join(f"{c:<{w}}" for c, w in cols)]
    for name, s in sorted(result["summary"].items()):
        vals = [name, s["task"], s["rtf"], s["chunk_p50_ms"], s["chunk_p95_ms"], s["chunk_p99_ms"], s["algorithmic_latency_ms"],
                s["measured_delay_ms"], s["target_similarity"], s["source_similarity"], s["wer"], s["clicks"]]
        lines.append(" ".join(f"{('-' if v is None else v)!s:<{w}}" for v, (_, w) in zip(vals, cols)))
    for k, v in result["skipped"].items():
        lines.append(f"{k:<15} skipped: {v[:150]}")
    for r in result["rows"]:
        if "error" in r:
            lines.append(f"{r['engine']:<15} error ({r['target']} <- {r['source']}): {r['error'][:150]}")
    lines.append(f"\nfiles: {result['out_dir']}")
    return "\n".join(lines)


__all__ = ["Clip", "format_summary", "measured_delay_ms", "run_vc_benchmark", "synthetic_clips"]
