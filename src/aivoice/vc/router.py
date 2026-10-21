"""Pick a live VC engine from what is installed here and what was *measured* here.

Hard filters: runnable now, the voice has what the engine needs (reference audio, an RVC .pth),
hardware, latency budget of the priority. Ranking uses ``aivoice benchmark vc`` results for
this machine's hardware fingerprint. Paper numbers are never used for ranking.

Without measurements the order is aivoice's previous behaviour (RVC when the voice has a
trained model, else MeanVC2) followed by the newer engines; it is printed as unmeasured so
nobody mistakes it for a quality ranking. X-VC becomes the ``auto`` choice only once local
benchmarks show it ahead.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..hardware import fingerprint
from ..paths import benchmarks_dir
from .base import TASK_VC, EngineState, EngineStatusVC, VoiceTarget

PRIORITIES = {
    # name: (description, max algorithmic+p95 latency ms or None)
    "balanced": ("similarity and intelligibility, then latency", 250.0),
    "latency": ("lowest measured end-to-end latency", None),
    "gaming": ("lowest latency within a 150 ms budget", 150.0),
    "similarity": ("highest measured target-speaker similarity", None),
    "naturalness": ("lowest measured WER/CER (intelligibility proxy; no naturalness predictor is installed)", None),
    "streaming-quality": ("best similarity within 250 ms", 250.0),
    "low-vram": ("least measured GPU memory", None),
    "cpu": ("engines that run without a GPU", None),
}
UNMEASURED_ORDER = ("rvc", "meanvc2", "xvc")  # never includes experimental engines


def _store() -> Path:
    return benchmarks_dir() / "vc_measurements.json"


def record(engine: str, quality: str, metrics: dict[str, Any], *, path: Path | None = None) -> None:
    p = path or _store()
    try:
        data = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        data = {}
    data[f"{fingerprint()}::{engine}::{quality}"] = {"engine": engine, "quality": quality, "measured_at": time.time(), **metrics}
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2) + "\n")


def measurements(*, path: Path | None = None, quality: str | None = None) -> dict[str, dict[str, Any]]:
    p = path or _store()
    try:
        data = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    fp = fingerprint()
    out: dict[str, dict[str, Any]] = {}
    for k, v in data.items():
        if not k.startswith(fp + "::"):
            continue
        if quality and v.get("quality") != quality:
            continue
        prev = out.get(v["engine"])
        if prev is None or v.get("measured_at", 0) > prev.get("measured_at", 0):
            out[v["engine"]] = v
    return out


@dataclass
class VCDecision:
    engine: str | None
    priority: str
    measured: bool
    reasons: list[str] = field(default_factory=list)
    rejected: dict[str, str] = field(default_factory=dict)
    candidates: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _latency(m: dict[str, Any]) -> float | None:
    a, p95 = m.get("algorithmic_latency_ms"), m.get("chunk_p95_ms")
    if a is None and p95 is None:
        return m.get("measured_delay_ms")
    return float(m.get("measured_delay_ms") or a or 0.0) + float(p95 or 0.0)


def _score(priority: str, m: dict[str, Any]) -> float | None:
    sim, wer, lat, vram = m.get("target_similarity"), m.get("wer"), _latency(m), m.get("vram_peak_mib")
    if m.get("rtf") is not None and m["rtf"] >= 1.0:
        return None  # not real time on this machine
    if priority in ("latency", "gaming"):
        return -lat if lat is not None else None
    if priority in ("similarity", "streaming-quality"):
        return sim
    if priority == "naturalness":
        return -wer if wer is not None else None
    if priority == "low-vram":
        return -vram if vram is not None else None
    if priority in ("balanced", "cpu"):
        if sim is None and wer is None:
            return None
        return (sim or 0.0) - (wer or 0.0) - 0.0005 * (lat or 0.0)
    return None


def route_vc(priority: str = "balanced", *, target: VoiceTarget | None = None, statuses: list[EngineStatusVC] | None = None,
             measured: dict[str, dict[str, Any]] | None = None) -> VCDecision:
    if priority not in PRIORITIES:
        raise ValueError(f"priority must be one of {', '.join(PRIORITIES)}")
    if statuses is None:
        from .engines import all_vc_status

        statuses = all_vc_status()
    m = measured if measured is not None else measurements()
    dec = VCDecision(None, priority, False)
    budget = PRIORITIES[priority][1]
    ok: list[EngineStatusVC] = []
    for s in statuses:
        c = s.caps
        if c.task != TASK_VC:
            continue  # modulation engines are not voice conversion
        if s.state == EngineState.EXPERIMENTAL:
            dec.rejected[s.name] = "experimental: only used when chosen explicitly"
            continue
        if not s.ready:
            dec.rejected[s.name] = f"{s.state.value}: {s.detail}"
            continue
        if c.requires_model_file and (target is None or target.model_file is None):
            dec.rejected[s.name] = "voice has no trained model file (RVC .pth)"
            continue
        if c.requires_reference and (target is None or target.reference is None):
            dec.rejected[s.name] = "voice has no reference audio"
            continue
        if priority == "cpu" and "cpu" not in c.devices:
            dec.rejected[s.name] = "needs a GPU"
            continue
        lat = _latency(m.get(s.name, {})) if s.name in m else c.algorithmic_latency_ms
        if budget is not None and lat is not None and lat > budget:
            dec.rejected[s.name] = f"latency {lat:.0f} ms > {budget:.0f} ms budget for '{priority}'"
            continue
        ok.append(s)
    dec.candidates = [s.name for s in ok]
    if not ok:
        dec.reasons.append("no installed voice-conversion engine can serve this voice (see `aivoice engines list`)")
        return dec
    scored = [(s, _score(priority, m.get(s.name, {}))) for s in ok]
    have = [(s, v) for s, v in scored if v is not None]
    if have:
        best, v = max(have, key=lambda sv: sv[1])
        dec.engine, dec.measured = best.name, True
        dec.reasons.append(f"best measured '{priority}' score on this machine ({v:.4g}; `aivoice benchmark vc`)")
        missing = [s.name for s, val in scored if val is None]
        if missing:
            dec.reasons.append(f"not measured here (neither better nor worse): {', '.join(missing)}")
        return dec
    order = {n: i for i, n in enumerate(UNMEASURED_ORDER)}
    best = min(ok, key=lambda s: order.get(s.name, 99))
    dec.engine = best.name
    dec.reasons.append("no local VC measurements: aivoice's fixed order (RVC for trained voices, then MeanVC2, then "
                       "X-VC), NOT a quality ranking. Run `aivoice benchmark vc` to route by measured results.")
    return dec


__all__ = ["PRIORITIES", "UNMEASURED_ORDER", "VCDecision", "measurements", "record", "route_vc"]
