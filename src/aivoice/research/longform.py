"""Long-form synthesis: segmentation, stable identity across segments, natural joins.

Informed by VoiceStar (Findings of ACL 2026), which shows autoregressive TTS
drifting on long inputs and addresses it inside the model (PM-RoPE,
continuation-prompt training). For engines without that, aivoice works around
the model:

* split at sentence/clause boundaries into segments within the engine's comfortable
  length (never mid-word);
* condition **every** segment on the same prepared reference (no chaining on
  previous outputs, which compounds drift);
* give each segment a duration plan where the engine supports it;
* join with punctuation-dependent pauses and equal-power crossfades, at one loudness;
* report per-segment speaker similarity to the reference (consistency) when an
  encoder is available, and flag outliers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..audio.enhancement import crossfade_concat, loudness_matched
from ..audio.resample import resample

PAUSE_MS = {".": 380, "!": 380, "?": 420, ";": 300, ":": 280, ",": 160, "\n\n": 650}
_ABBR = re.compile(r"\b(?:Mr|Mrs|Ms|Dr|St|vs|etc|e\.g|i\.e|No)\.$", re.I)


@dataclass
class Segment:
    text: str
    pause_after_ms: int


def split_text(text: str, *, max_chars: int = 220, min_chars: int = 30) -> list[Segment]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    segs: list[Segment] = []
    for pi, para in enumerate(paragraphs):
        sentences: list[str] = []
        buf = ""
        for tok in re.split(r"(?<=[.!?])\s+", para):
            buf = f"{buf} {tok}".strip() if buf else tok
            if _ABBR.search(buf):
                continue
            sentences.append(buf)
            buf = ""
        if buf:
            sentences.append(buf)
        merged: list[str] = []
        for s in sentences:
            if merged and len(merged[-1]) < min_chars and len(merged[-1]) + len(s) + 1 <= max_chars:
                merged[-1] = merged[-1] + " " + s
            else:
                merged.append(s)
        for s in merged:
            for piece in _split_long(s, max_chars):
                end = piece.rstrip()[-1:] if piece.rstrip() else "."
                segs.append(Segment(piece, PAUSE_MS.get(end, 250)))
        if segs and pi < len(paragraphs) - 1:
            segs[-1].pause_after_ms = PAUSE_MS["\n\n"]
    if segs:
        segs[-1].pause_after_ms = 0
    return segs


def _split_long(s: str, max_chars: int) -> list[str]:
    if len(s) <= max_chars:
        return [s]
    out, buf = [], ""
    for clause in re.split(r"(?<=[,;:])\s+", s):
        if len(buf) + len(clause) + 1 <= max_chars:
            buf = f"{buf} {clause}".strip()
            continue
        if buf:
            out.append(buf)
        while len(clause) > max_chars:
            cut = clause.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            out.append(clause[:cut])
            clause = clause[cut:].strip()
        buf = clause
    if buf:
        out.append(buf)
    return out


@dataclass
class LongformResult:
    audio: np.ndarray
    sample_rate: int
    segments: list[dict[str, Any]] = field(default_factory=list)
    consistency_min: float | None = None
    outliers: list[int] = field(default_factory=list)


def synthesize_longform(
    engine: Any,
    text: str,
    speaker: Any,
    options: Any,
    *,
    max_chars: int = 220,
    encoder: Any | None = None,
    reference_audio: tuple[np.ndarray, int] | None = None,
    plan_durations: bool = True,
    outlier_margin: float = 0.08,
) -> LongformResult:
    from dataclasses import replace

    segs = split_text(text, max_chars=max_chars)
    caps = engine.capabilities()
    parts: list[np.ndarray] = []
    info: list[dict[str, Any]] = []
    sr_out: int | None = None
    for i, seg in enumerate(segs):
        o = options
        if plan_durations and caps.duration_control and speaker is not None and getattr(speaker, "audio", None) and options.target_duration_s is None:
            from .duration import plan_duration

            plan = plan_duration(seg.text, speaker.audio, transcript=speaker.transcript, language=options.language or "en",
                                 speed=options.speed or 1.0)
            o = replace(options, target_duration_s=plan.seconds)
        r = engine.synthesize(seg.text, speaker, o)
        if sr_out is None:
            sr_out = r.sample_rate
        audio = r.audio if r.sample_rate == sr_out else resample(r.audio, r.sample_rate, sr_out)
        parts.append(audio)
        info.append({"index": i, "text": seg.text, "seconds": round(audio.size / sr_out, 3),
                     "pause_after_ms": seg.pause_after_ms, "target_duration_s": o.target_duration_s})
    sr_out = sr_out or 16000
    parts = loudness_matched(parts, sr_out)
    audio = crossfade_concat(parts, sr_out, gaps_ms=[s.pause_after_ms for s in segs[:-1]])
    res = LongformResult(audio, sr_out, info)
    if encoder is not None and reference_audio is not None and parts:
        ref = encoder.embed(*reference_audio)
        sims = []
        for p, d in zip(parts, info):
            try:
                s = float(np.dot(encoder.embed(p, sr_out), ref))
            except ValueError:
                s = None
            d["similarity_to_reference"] = s
            if s is not None:
                sims.append(s)
        if sims:
            res.consistency_min = min(sims)
            med = float(np.median(sims))
            res.outliers = [d["index"] for d in info if d.get("similarity_to_reference") is not None
                            and d["similarity_to_reference"] < med - outlier_margin]
    return res
