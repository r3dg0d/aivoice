"""Per-utterance timing records and latency reports (``time.monotonic`` throughout)."""

from __future__ import annotations

import itertools
import threading
from dataclasses import asdict, dataclass, field
from typing import Any

from ..asr.bench import summarize

_ids = itertools.count(1)


def new_uid() -> str:
    return f"u{next(_ids):05d}"


@dataclass
class SegmentTiming:
    seq: int
    text: str
    final: bool
    t_commit: float
    t_llm_start: float | None = None
    t_llm_end: float | None = None
    llm_fallback: str | None = None
    tts_text: str | None = None
    t_tts_request: float | None = None
    t_tts_first_chunk: float | None = None
    t_first_played: float | None = None
    t_done: float | None = None
    cancelled: str | None = None  # reason
    error: str | None = None
    audio_s: float = 0.0


@dataclass
class UtteranceTimeline:
    uid: str
    asr_backend: str
    t_speech_start: float  # first speech block (VAD start), monotonic
    t_endpoint: float | None = None  # VAD declared end of speech
    silence_ms: float = 0.0  # hangover included in t_endpoint
    t_first_partial: float | None = None
    t_final: float | None = None
    partials: list[tuple[float, int, str]] = field(default_factory=list)
    final_text: str = ""
    asr_final_ms: float | None = None
    segments: list[SegmentTiming] = field(default_factory=list)
    queue_depth: int = 0
    forced_split: bool = False
    errors: list[str] = field(default_factory=list)
    cancelled: bool = False

    # derived ------------------------------------------------------------
    def first_played(self) -> float | None:
        ts = [s.t_first_played for s in self.segments if s.t_first_played is not None]
        return min(ts) if ts else None

    def speech_end(self) -> float | None:
        return self.t_endpoint - self.silence_ms / 1000 if self.t_endpoint is not None else None

    def metrics(self) -> dict[str, float | None]:
        fp = self.first_played()

        def ms(a, b):
            return (a - b) * 1000 if a is not None and b is not None else None

        first_seg = self.segments[0] if self.segments else None
        done = [s.t_done for s in self.segments if s.t_done]
        return {
            "speech_end_to_first_audio_ms": ms(fp, self.speech_end()),
            "endpoint_to_first_audio_ms": ms(fp, self.t_endpoint),
            "stable_to_first_audio_ms": ms(fp, first_seg.t_commit) if first_seg else None,
            "tts_request_to_first_audio_ms": ms(first_seg.t_tts_first_chunk, first_seg.t_tts_request) if first_seg else None,
            "capture_to_playback_ms": ms(fp, self.t_speech_start),
            "asr_final_ms": self.asr_final_ms,
            "llm_ms": ms(first_seg.t_llm_end, first_seg.t_llm_start) if first_seg else None,
            "full_utterance_ms": ms(max(done), self.t_speech_start) if done else None,
            "first_partial_ms": ms(self.t_first_partial, self.t_speech_start),
        }

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["metrics"] = self.metrics()
        return d


class TimelineStore:
    def __init__(self, keep: int = 500) -> None:
        self._lock = threading.Lock()
        self._items: dict[str, UtteranceTimeline] = {}
        self.keep = keep

    def add(self, t: UtteranceTimeline) -> UtteranceTimeline:
        with self._lock:
            self._items[t.uid] = t
            while len(self._items) > self.keep:
                self._items.pop(next(iter(self._items)))
        return t

    def get(self, uid: str) -> UtteranceTimeline | None:
        with self._lock:
            return self._items.get(uid)

    def all(self) -> list[UtteranceTimeline]:
        with self._lock:
            return list(self._items.values())

    def report(self) -> dict[str, Any]:
        items = [t for t in self.all() if t.segments]
        keys = [
            "speech_end_to_first_audio_ms",
            "endpoint_to_first_audio_ms",
            "stable_to_first_audio_ms",
            "tts_request_to_first_audio_ms",
            "capture_to_playback_ms",
            "asr_final_ms",
            "llm_ms",
            "full_utterance_ms",
            "first_partial_ms",
        ]
        out: dict[str, Any] = {"utterances": len(self.all()), "spoken": len(items)}
        for k in keys:
            vals = [v for v in (t.metrics()[k] for t in items) if v is not None]
            out[k] = summarize(vals)
        out["cancelled_segments"] = sum(1 for t in self.all() for s in t.segments if s.cancelled)
        out["errors"] = sum(len(t.errors) + sum(1 for s in t.segments if s.error) for t in self.all())
        return out


def format_report(r: dict[str, Any]) -> str:
    lines = [f"utterances {r['utterances']}  spoken {r['spoken']}  cancelled segments {r['cancelled_segments']}  errors {r['errors']}"]
    lines.append(f"{'metric':<34} {'n':>4} {'p50':>8} {'p90':>8} {'p95':>8}")
    for k, v in r.items():
        if isinstance(v, dict) and "p50" in v:
            def f(x):
                return f"{x:8.0f}" if x is not None else "       -"

            lines.append(f"{k:<34} {v['n']:>4} {f(v['p50'])} {f(v['p90'])} {f(v['p95'])}")
    return "\n".join(lines)
