"""Streaming voice-conversion / modulation engine interface.

Three different systems share the live audio path but are different tasks:

* **direct VC** (``task="voice_conversion"``): source speech -> target speaker (X-VC, RVC, MeanVC2 ...)
* **modulation** (``task="modulation"``): the *same* speaker with altered attributes (DSP engine)
* **speech-to-speech** (ASR -> LLM -> TTS): lives in :mod:`aivoice.realtime`, not here

A :class:`StreamingVCEngine` is loaded once, keeps its model resident, and converts a
continuous stream chunk by chunk with persistent state (codec / vocoder / F0 / crossfade caches).
Capabilities are facts about the integration, not paper claims; every status is the result
of an actual check on this machine.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

TASK_VC = "voice_conversion"
TASK_MOD = "modulation"


class EngineState(str, Enum):
    READY = "READY"  # runtime + weights present, can start now
    INSTALLED = "INSTALLED"  # runtime present; readiness depends on a per-voice asset (e.g. an RVC .pth)
    MISSING_WEIGHTS = "MISSING_WEIGHTS"
    MISSING_RUNTIME = "MISSING_RUNTIME"
    UNSUPPORTED_HARDWARE = "UNSUPPORTED_HARDWARE"
    EXPERIMENTAL = "EXPERIMENTAL"  # runnable, but high latency / restricted use; never chosen by `auto`
    RESEARCH_ONLY = "RESEARCH_ONLY"  # paper only, no public implementation
    BLOCKED_LICENSE = "BLOCKED_LICENSE"
    BLOCKED_CODE = "BLOCKED_CODE"  # code/weights published incompletely

    @property
    def runnable(self) -> bool:
        return self in (EngineState.READY, EngineState.INSTALLED, EngineState.EXPERIMENTAL)


class VCError(RuntimeError):
    pass


class VCUnavailable(VCError):
    pass


@dataclass(frozen=True)
class VCCapabilities:
    name: str
    display_name: str
    task: str = TASK_VC
    streaming: bool = True
    zero_shot: bool = False
    requires_reference: bool = True  # target reference audio
    requires_model_file: bool = False  # per-voice trained model (RVC .pth)
    supports_cross_lingual: bool | None = None  # None = not verified
    supports_pitch_control: bool = False
    supports_index: bool = False
    supports_style: bool = False
    sample_rate: int | None = None  # rate the engine consumes and produces
    chunk_ms: float | None = None  # hop between outputs
    lookahead_ms: float | None = None  # future context the model waits for
    algorithmic_latency_ms: float | None = None  # chunk + lookahead + smoothing (by construction, not measured)
    estimated_vram_mib: int | None = None
    devices: tuple[str, ...] = ("cuda",)
    license: str = ""
    code: str = ""
    weights: str = ""
    paper: str = ""
    evidence: str = ""  # peer-reviewed | official-implementation | community-implementation | unvalidated
    install: str = ""
    notes: str = ""
    facts: dict[str, str] = field(default_factory=dict, hash=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EngineStatusVC:
    name: str
    state: EngineState
    caps: VCCapabilities
    detail: str = ""

    @property
    def ready(self) -> bool:
        return self.state in (EngineState.READY, EngineState.INSTALLED)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "state": self.state.value, "detail": self.detail, "ready": self.ready,
                "runnable": self.state.runnable, **{"caps": self.caps.to_dict()}}


@dataclass
class VoiceTarget:
    """What a VC engine needs to convert into one voice."""

    name: str
    reference: Path | None = None
    model_file: Path | None = None  # RVC .pth
    index_file: Path | None = None  # RVC .index
    extra: dict[str, Any] = field(default_factory=dict)  # per-engine (StyleStream style dir, rt.json settings)


@dataclass
class StreamOptions:
    pitch_semitones: float = 0.0
    index_rate: float | None = None
    protect: float | None = None
    chunk_ms: float | None = None
    quality: str = "balanced"  # lowest-latency | balanced | best-quality
    device: str = "cuda"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkTiming:
    process_ms: float
    in_samples: int
    out_samples: int


@runtime_checkable
class StreamingVCEngine(Protocol):
    def capabilities(self) -> VCCapabilities: ...

    def status(self) -> EngineStatusVC: ...

    def load(self) -> None:
        """Load models once (idempotent). Raises VCUnavailable with an install hint."""

    def prepare_reference(self, target: VoiceTarget | None) -> None:
        """Compute target conditioning once per voice (speaker embedding, codec prompt ...)."""

    def start_stream(self, options: StreamOptions) -> None:
        """Reset streaming caches for a new continuous stream."""

    def process_chunk(self, pcm: np.ndarray) -> np.ndarray:
        """float32 mono at capabilities().sample_rate in, converted audio out (may be empty while priming)."""

    def flush(self) -> np.ndarray:
        """Drain buffered audio at end of stream."""

    def reset_stream(self) -> None: ...

    def unload(self) -> None: ...


class BaseEngine:
    """Shared plumbing: timing stats and a default offline ``convert_file`` that streams the file."""

    name = "base"

    TIMING_WINDOW = 20000  # recent chunks kept for percentiles; totals are running sums (bounded memory)

    def __init__(self) -> None:
        self.timings: deque[ChunkTiming] = deque(maxlen=self.TIMING_WINDOW)
        self.total_ms = 0.0
        self.total_chunks = 0
        self.loaded = False
        self.target: VoiceTarget | None = None
        self.options = StreamOptions()

    # subclasses implement capabilities/status/load/_process/_flush

    def process_chunk(self, pcm: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        out = self._process(np.asarray(pcm, np.float32).reshape(-1))
        out = np.asarray(out, np.float32).reshape(-1)
        ms = (time.perf_counter() - t0) * 1000
        self.timings.append(ChunkTiming(ms, int(np.size(pcm)), int(out.size)))
        self.total_ms += ms
        self.total_chunks += 1
        return out

    def _process(self, pcm: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def flush(self) -> np.ndarray:
        return np.zeros(0, np.float32)

    def prepare_reference(self, target: VoiceTarget | None) -> None:
        caps = self.capabilities()  # type: ignore[attr-defined]
        if caps.requires_reference and (target is None or target.reference is None):
            raise VCError(f"{caps.name} needs target reference audio (a voice)")
        if caps.requires_model_file and (target is None or target.model_file is None):
            raise VCError(f"{caps.name} needs a per-voice model file (RVC .pth) for this voice")
        self.target = target

    def start_stream(self, options: StreamOptions) -> None:
        self.options = options
        self.timings = deque(maxlen=self.TIMING_WINDOW)
        self.total_ms, self.total_chunks = 0.0, 0

    def reset_stream(self) -> None:
        self.start_stream(self.options)

    def unload(self) -> None:
        self.loaded = False

    def convert_array(self, x: np.ndarray, sr: int, *, chunk_ms: float | None = None) -> tuple[np.ndarray, int]:
        """Stream a whole clip through the live path (same code as live), return (audio, rate)."""
        from ..audio.resample import resample

        caps = self.capabilities()  # type: ignore[attr-defined]
        rate = caps.sample_rate or sr
        y = resample(np.asarray(x, np.float32), sr, rate) if sr != rate else np.asarray(x, np.float32)
        step = int(rate * (chunk_ms or caps.chunk_ms or 20) / 1000)
        parts = [self.process_chunk(y[i : i + step]) for i in range(0, y.size, step)]
        parts.append(self.flush())
        out = np.concatenate([p for p in parts if p.size]) if any(p.size for p in parts) else np.zeros(0, np.float32)
        return out, rate

    def timing_summary(self) -> dict[str, Any]:
        if not self.timings:
            return {}
        ms = np.array([t.process_ms for t in self.timings])
        return {"chunks": int(ms.size), "mean_ms": round(float(ms.mean()), 3), "p50_ms": round(float(np.percentile(ms, 50)), 3),
                "p95_ms": round(float(np.percentile(ms, 95)), 3), "p99_ms": round(float(np.percentile(ms, 99)), 3),
                "max_ms": round(float(ms.max()), 3)}


__all__ = [
    "BaseEngine", "ChunkTiming", "EngineState", "EngineStatusVC", "StreamOptions", "StreamingVCEngine",
    "TASK_MOD", "TASK_VC", "VCCapabilities", "VCError", "VCUnavailable", "VoiceTarget",
]
