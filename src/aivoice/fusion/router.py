"""Adaptive engine router.

Hard filters first (task, installed, capabilities the request needs, memory),
then a ranking by the requested priority using **measurements taken on this
machine** (``aivoice benchmark research`` / ``engines benchmark``). Without
measurements the router says so and falls back to a fixed, documented
preference order: it never claims an unmeasured engine is better.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..engines.base import TASK_TTS, TASK_VC, EngineCapabilities, SpeakerRef, SynthesisOptions
from ..hardware import fingerprint
from ..paths import benchmarks_dir

PRIORITIES = ("balanced", "latency", "similarity", "naturalness", "multilingual", "expressiveness")
# Used only when nothing has been measured; documented in docs/model-backends.md.
UNMEASURED_ORDER = ("voxcpm2", "xvoice", "voicestar", "fish-server", "fish-api", "meanvc2", "espeak")


def _store() -> Path:
    return benchmarks_dir() / "engine_measurements.json"


def record_measurement(engine: str, metrics: dict[str, Any], *, path: Path | None = None) -> None:
    p = path or _store()
    try:
        data = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        data = {}
    key = f"{fingerprint()}::{engine}"
    data[key] = {"engine": engine, "measured_at": time.time(), **metrics}
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2) + "\n")


def measurements(*, path: Path | None = None) -> dict[str, dict[str, Any]]:
    p = path or _store()
    try:
        data = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    fp = fingerprint()
    return {v["engine"]: v for k, v in data.items() if k.startswith(fp + "::")}


@dataclass
class Decision:
    engine: str | None
    priority: str
    measured: bool
    reasons: list[str] = field(default_factory=list)
    rejected: dict[str, str] = field(default_factory=dict)
    candidates: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _requirements(caps: EngineCapabilities, task: str, speaker: SpeakerRef | None, o: SynthesisOptions,
                  streaming: bool) -> str | None:
    if task not in caps.tasks:
        return f"does not do {task}"
    if speaker is not None and speaker.audio is not None and not caps.zero_shot:
        return "cannot clone from reference audio"
    if speaker is not None and caps.requires_prompt_transcript and not speaker.transcript:
        return "needs the reference transcript (profile has none)"
    if o.style_prompt and not caps.style_prompt:
        return "no style prompt support"
    if o.style_reference and not caps.style_reference:
        return "no separate style reference support"
    if o.target_duration_s is not None and not caps.duration_control:
        return "no duration control"
    if o.language and caps.languages and o.language.lower() not in caps.languages:
        return f"language {o.language} not listed"
    if streaming and not caps.streaming:
        return "not streaming"
    if caps.remote:
        from .. import settings

        if not settings.has_remote_consent(caps.name):
            return "remote engine not allowed (aivoice experimental remote allow)"
    return None


def _score(priority: str, m: dict[str, Any]) -> float | None:
    sim, wer, ttfa, rtf = m.get("speaker_similarity"), m.get("wer"), m.get("first_audio_ms"), m.get("rtf")
    if priority == "latency":
        if ttfa is not None:
            return -float(ttfa)
        return -float(rtf) * 1000 if rtf is not None else None
    if priority == "similarity":
        return sim
    if priority == "naturalness":
        # No validated objective naturalness metric is installed; intelligibility is the measurable proxy.
        return -wer if wer is not None else None
    if priority in ("balanced", "multilingual", "expressiveness"):
        if sim is None and wer is None:
            return None
        return (sim or 0.0) - (wer or 0.0) - 0.0001 * (ttfa or 0.0)
    return None


def route(
    task: str = TASK_TTS,
    *,
    priority: str = "balanced",
    speaker: SpeakerRef | None = None,
    options: SynthesisOptions | None = None,
    streaming: bool = False,
    statuses: list[Any] | None = None,
    measured: dict[str, dict[str, Any]] | None = None,
    free_vram_mib: int | None = None,
) -> Decision:
    if priority not in PRIORITIES:
        raise ValueError(f"priority must be one of {PRIORITIES}")
    o = options or SynthesisOptions()
    if statuses is None:
        from ..engines.registry import all_status

        statuses = all_status()
    if free_vram_mib is None:
        from ..hardware import detect

        gpus = detect().gpus
        free_vram_mib = max((g.free_mib for g in gpus), default=0)
    dec = Decision(None, priority, False)
    ok: list[Any] = []
    for s in statuses:
        if not s.ready:
            dec.rejected[s.name] = f"unavailable: {s.unavailable}"
            continue
        why = _requirements(s.caps, task, speaker, o, streaming)
        if why:
            dec.rejected[s.name] = why
            continue
        if s.caps.est_vram_mib and "cpu" not in s.caps.devices and free_vram_mib < s.caps.est_vram_mib:
            dec.rejected[s.name] = f"needs ~{s.caps.est_vram_mib} MiB GPU memory, {free_vram_mib} MiB free"
            continue
        if priority == "multilingual" and not s.caps.cross_lingual:
            dec.rejected[s.name] = "not cross-lingual"
            continue
        if priority == "expressiveness" and not (s.caps.style_prompt or s.caps.emotion_tags or s.caps.style_reference):
            dec.rejected[s.name] = "no style/emotion controls"
            continue
        ok.append(s)
    dec.candidates = [s.name for s in ok]
    if not ok:
        dec.reasons.append("no installed engine satisfies the request")
        return dec
    m = measured if measured is not None else measurements()
    scored = [(s, _score(priority, m.get(s.name, {}))) for s in ok]
    with_scores = [(s, v) for s, v in scored if v is not None]
    if with_scores:
        best, v = max(with_scores, key=lambda sv: sv[1])
        dec.engine, dec.measured = best.name, True
        dec.reasons.append(f"best measured {priority} score on this machine ({v:.4g})")
        unmeasured = [s.name for s, val in scored if val is None]
        if unmeasured:
            dec.reasons.append(f"not measured yet (not considered better or worse): {', '.join(unmeasured)}")
        return dec
    order = {n: i for i, n in enumerate(UNMEASURED_ORDER)}
    best = min(ok, key=lambda s: order.get(s.name, 99))
    dec.engine = best.name
    dec.reasons.append(
        "no measurements on this machine; picked by aivoice's fixed preference order (not a quality claim). "
        "Run `aivoice benchmark research` to route by measured results."
    )
    if task == TASK_VC:
        dec.reasons.append("voice conversion: MeanVC2 is the only VC engine")
    return dec
