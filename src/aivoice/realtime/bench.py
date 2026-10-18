"""Benchmark suite: ASR backends on identical clips, TTS time-to-first-audio, and the full pipeline.

Test audio: a directory of ``name.wav`` + ``name.txt`` (reference transcript)
pairs, or - when espeak-ng is installed - a generated set with known text.
Generated speech is robotic; it is fine for latency, only indicative for WER.
Scenario variants are derived from each clip: quiet (-20 dB), noise (white
noise at 10 dB SNR), rapid (faster speaking rate), long (clips joined),
interruptions (clips separated by 150 ms gaps).

Everything reported is measured on this machine at run time. Pipeline runs
with ``mock=True`` use a scripted ASR and a tone TTS: those numbers describe
aivoice's own scheduling overhead, never model speed, and are labelled so.
"""

from __future__ import annotations

import json
import os
import resource
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from ..asr.bench import run_asr_benchmark, summarize
from ..audio.resample import resample
from ..hardware import detect, fingerprint, query_nvidia_smi
from ..paths import benchmarks_dir

SENTENCES = {
    "short": ["Turn left at the next corner.", "That's actually pretty funny.", "Can you hear me now?"],
    "long": [
        "We should meet at seven thirty tomorrow evening, because the train from the city arrives a little late.",
        "Honestly, I did not expect the second level to be that difficult, but the boss fight was worth it.",
    ],
    "rapid": ["Okay okay I get it, just give me one second and I will be right there."],
}


def load_clip_dir(path: Path) -> list[dict[str, Any]]:
    from ..audio_io import read_wav

    clips = []
    for wav in sorted(Path(path).glob("*.wav")):
        txt = wav.with_suffix(".txt")
        pcm, sr = read_wav(wav)
        clips.append({"name": wav.stem, "audio": pcm, "sr": sr, "text": txt.read_text().strip() if txt.is_file() else None, "scenario": "user"})
    return clips


def espeak_clips(rate_wpm: int = 165) -> list[dict[str, Any]]:
    exe = shutil.which("espeak-ng")
    if not exe:
        return []
    from ..audio_io import read_wav

    out = []
    with tempfile.TemporaryDirectory(prefix="aivoice-bench-") as td:
        for scen, sents in SENTENCES.items():
            for i, s in enumerate(sents):
                f = Path(td) / f"{scen}{i}.wav"
                speed = rate_wpm + (80 if scen == "rapid" else 0)
                subprocess.run([exe, "-s", str(speed), "-w", str(f), s], check=True, capture_output=True, timeout=30)
                pcm, sr = read_wav(f)
                out.append({"name": f"{scen}{i}", "audio": resample(pcm, sr, 16000), "sr": 16000, "text": s, "scenario": scen})
    return out


def derive_scenarios(clips: list[dict[str, Any]], seed: int = 0) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    out = list(clips)
    base = [c for c in clips if c["scenario"] in ("short", "user")][:3] or clips[:3]
    for c in base:
        x = c["audio"].astype(np.float32)
        out.append({**c, "name": c["name"] + "-quiet", "audio": x * 0.1, "scenario": "quiet"})
        p = float(np.mean(x * x)) + 1e-9
        noise = rng.normal(0, np.sqrt(p / 10), x.size).astype(np.float32)
        out.append({**c, "name": c["name"] + "-noise", "audio": x + noise, "scenario": "noise"})
    if len(clips) >= 2 and all(c["text"] for c in clips[:3]):
        gap = np.zeros(int(0.15 * 16000), np.float32)
        parts = [c["audio"] if c["sr"] == 16000 else resample(c["audio"], c["sr"], 16000) for c in clips[:3]]
        joined = np.concatenate([p for x in parts for p in (x, gap)])
        out.append({"name": "interruptions", "audio": joined, "sr": 16000, "text": " ".join(c["text"] for c in clips[:3]), "scenario": "interruptions"})
    return out


class ResourceSampler:
    """Samples process CPU time, peak RSS and GPU memory (nvidia-smi) while a benchmark runs."""

    def __init__(self, interval: float = 0.25) -> None:
        self.interval = interval
        self.gpu_peak_mib: int | None = None
        self._stop = threading.Event()
        self._t = None

    def __enter__(self):
        self.wall0 = time.monotonic()
        self.cpu0 = time.process_time()
        if shutil.which("nvidia-smi"):
            self._t = threading.Thread(target=self._run, daemon=True)
            self._t.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            for g in query_nvidia_smi():
                used = g.total_mib - g.free_mib
                self.gpu_peak_mib = max(self.gpu_peak_mib or 0, used)
            time.sleep(self.interval)

    def __exit__(self, *exc):
        self._stop.set()
        wall = max(1e-6, time.monotonic() - self.wall0)
        self.cpu_util = (time.process_time() - self.cpu0) / wall / (os.cpu_count() or 1)
        self.peak_rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "cpu_util_of_all_cores": round(self.cpu_util, 4),
            "peak_rss_mib": round(self.peak_rss_mib, 1),
            "gpu_mem_used_peak_mib": self.gpu_peak_mib,
        }


def bench_asr(backends: dict[str, Any], clips: list[dict[str, Any]], *, record: bool = True) -> dict[str, Any]:
    from ..asr import backend_config
    from ..asr.selector import ASRSelector, Measurement

    results: dict[str, Any] = {}
    for name, be in backends.items():
        try:
            with ResourceSampler() as rs:
                r = run_asr_benchmark(be, [(c["audio"], c["sr"], c["text"]) for c in clips])
            by_scen: dict[str, list[tuple[str, str]]] = {}
            for c, h in zip(clips, r.hypotheses):
                if c["text"]:
                    by_scen.setdefault(c["scenario"], []).append((c["text"], h))
            from ..asr.bench import wer

            results[name] = {
                "streaming": be.get_capabilities().streaming_method if hasattr(be, "get_capabilities") else None,
                "clips": r.clips,
                "audio_s": round(r.audio_s, 2),
                "load_ms": r.load_ms,
                "final_latency_ms": summarize(r.final_latency_ms),
                "first_partial_ms": summarize(r.first_partial_ms),
                "rtf": r.rtf,
                "wer": r.wer,
                "wer_by_scenario": {k: wer([a for a, _ in v], [b for _, b in v]) for k, v in by_scen.items()},
                "resources": rs.to_dict(),
                "samples": [{"ref": c["text"], "hyp": h} for c, h in list(zip(clips, r.hypotheses))[:5]],
            }
            if record and r.final_latency_ms:
                ASRSelector().record(
                    Measurement(
                        backend=name,
                        final_p50_ms=summarize(r.final_latency_ms)["p50"] or 0.0,
                        final_p95_ms=summarize(r.final_latency_ms)["p95"],
                        rtf=r.rtf,
                        wer=r.wer,
                        rss_mib=rs.peak_rss_mib,
                        clips=r.clips,
                        measured_at=time.time(),
                        config=backend_config(name),
                    )
                )
        except Exception as e:  # noqa: BLE001
            results[name] = {"error": str(e)}
        finally:
            try:
                be.unload_model()
            except Exception:  # noqa: BLE001
                pass
    return results


def bench_tts(tts: Any, sentences: list[str] | None = None, voice: Any | None = None) -> dict[str, Any]:
    sentences = sentences or [s for v in SENTENCES.values() for s in v]
    ttfa, total, rtf = [], [], []
    errors = []
    with ResourceSampler() as rs:
        for s in sentences:
            t0 = time.perf_counter()
            first = None
            n = 0
            sr = 1
            try:
                for ch in tts.synthesize(s, voice=voice):
                    if first is None:
                        first = (time.perf_counter() - t0) * 1000
                    n += ch.pcm.size
                    sr = ch.sample_rate
            except Exception as e:  # noqa: BLE001
                errors.append(str(e))
                continue
            dt = (time.perf_counter() - t0) * 1000
            if first is not None:
                ttfa.append(first)
            total.append(dt)
            if n:
                rtf.append(dt / 1000 / (n / sr))
    caps = tts.capabilities()
    return {
        "backend": caps.name,
        "streaming": caps.streaming,
        "remote": caps.remote,
        "ttfa_ms": summarize(ttfa),
        "total_ms": summarize(total),
        "rtf": summarize(rtf),
        "errors": errors,
        "resources": rs.to_dict(),
    }


def bench_pipeline(clips: list[dict[str, Any]], *, cfg: dict[str, Any] | None = None, mock: bool = False) -> dict[str, Any]:
    from .. import settings as st
    from ..asr.mock import ScriptedASR
    from ..audio.capture import ArrayCapture
    from ..tts.mock import ToneTTS
    from .build import build_engine

    cfg = cfg or st.load()
    reports = []
    with ResourceSampler() as rs:
        for c in clips:
            kw: dict[str, Any] = {}
            if mock:
                kw["asr_override"] = ScriptedASR([c["text"] or "test"], words_per_s=3.0)
                kw["tts_override"] = ToneTTS(ttfa_ms=0)
            eng, _ = build_engine(cfg, output="null", write_status=False, **kw)
            eng.capture = ArrayCapture(c["audio"], c["sr"], realtime=True)
            eng.start(warmup=True)
            eng.wait_idle(timeout=max(30.0, c["audio"].size / c["sr"] * 4))
            eng.stop()
            rep = eng.timelines.report()
            rep["clip"] = c["name"]
            rep["scenario"] = c["scenario"]
            rep["underruns"] = eng.player.underruns if eng.player else None
            rep["dropped_capture_blocks"] = getattr(eng.capture, "dropped", 0)
            rep["dropped_segments"] = eng.dropped_segments
            rep["timelines"] = [t.to_dict() for t in eng.timelines.all()]
            reports.append(rep)
    keys = [k for k in reports[0] if isinstance(reports[0][k], dict) and "p50" in reports[0][k]] if reports else []
    agg = {}
    for k in keys:
        vals = [t["metrics"][k] for r in reports for t in r["timelines"] if t["metrics"].get(k) is not None]
        agg[k] = summarize(vals)
    return {
        "mock": mock,
        "note": "mock run: scripted ASR + tone TTS; measures aivoice overhead only" if mock else "configured backends",
        "aggregate": agg,
        "underruns": sum(r["underruns"] or 0 for r in reports),
        "dropped_capture_blocks": sum(r["dropped_capture_blocks"] for r in reports),
        "dropped_segments": sum(r["dropped_segments"] for r in reports),
        "resources": rs.to_dict(),
        "clips": reports,
    }


def save_report(kind: str, data: dict[str, Any]) -> tuple[Path, Path]:
    hw = detect()
    data = {
        "kind": kind,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "fingerprint": fingerprint(hw),
        "hardware": {
            "cpu": hw.cpu_model,
            "threads": hw.cpu_threads,
            "ram_mib": hw.ram_total_mib,
            "gpus": [asdict(g) for g in hw.gpus],
            "runtimes": {k: v for k, v in hw.runtimes.items() if v},
        },
        **data,
    }
    d = benchmarks_dir()
    d.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    j = d / f"{stamp}-{kind}.json"
    t = d / f"{stamp}-{kind}.txt"
    j.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")
    t.write_text(format_bench(data) + "\n", encoding="utf-8")
    return j, t


def _fmt(s: dict | None) -> str:
    if not s or s.get("p50") is None:
        return "-"
    return f"p50 {s['p50']:.0f} / p90 {s['p90']:.0f} / p95 {s['p95']:.0f} ms (n={s['n']})"


def format_bench(data: dict[str, Any]) -> str:
    lines = [f"aivoice benchmark: {data.get('kind')}  {data.get('created', '')}", f"hardware: {data.get('hardware', {}).get('cpu')}"]
    gpus = data.get("hardware", {}).get("gpus") or []
    lines.append("gpu: " + (", ".join(g["name"] for g in gpus) if gpus else "none"))
    if "asr" in data:
        lines.append("\nASR (identical clips)")
        for name, r in data["asr"].items():
            if "error" in r:
                lines.append(f"  {name:<18} ERROR {r['error']}")
                continue
            w = f"{r['wer']:.1%}" if r.get("wer") is not None else "n/a"
            rtf = f"{r['rtf']:.3f}" if r.get("rtf") is not None else "n/a"
            lines.append(f"  {name:<18} [{r['streaming']}] WER {w}  RTF {rtf}  final {_fmt(r['final_latency_ms'])}")
            lines.append(f"  {'':<18} first partial {_fmt(r['first_partial_ms'])}  RSS {r['resources']['peak_rss_mib']} MiB")
    if "tts" in data:
        r = data["tts"]
        lines.append(f"\nTTS {r['backend']} (streaming={r['streaming']}, remote={r['remote']})")
        lines.append(f"  time to first audio {_fmt(r['ttfa_ms'])}")
        lines.append(f"  errors {len(r['errors'])}")
    if "pipeline" in data:
        p = data["pipeline"]
        lines.append(f"\nPipeline ({p['note']})")
        for k, v in p["aggregate"].items():
            lines.append(f"  {k:<34} {_fmt(v)}")
        lines.append(f"  underruns {p['underruns']}  dropped capture blocks {p['dropped_capture_blocks']}  dropped segments {p['dropped_segments']}")
        lines.append(f"  resources {p['resources']}")
    return "\n".join(lines)
