"""Shared ASR interface.

A backend that decodes incrementally from cached state sets
``capabilities.native_streaming = True`` and implements ``transcribe_stream``.
A backend that can only decode a whole clip does not: it leaves
``transcribe_stream`` raising, and :func:`streaming_view` wraps it in
:class:`~aivoice.asr.chunked.ChunkedStreamingAdapter`, which is labelled as
chunked re-decoding everywhere it is reported. Nothing pretends to stream.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import numpy as np


class ASRError(RuntimeError):
    pass


class StreamingNotSupported(ASRError):
    pass


@dataclass
class Word:
    text: str
    start: float | None = None
    end: float | None = None
    confidence: float | None = None


@dataclass
class TranscriptUpdate:
    text: str
    is_final: bool
    revision: int = 0
    backend: str = ""
    words: list[Word] | None = None
    latency_ms: float | None = None  # compute time of the pass that produced it
    audio_s: float | None = None  # audio covered so far
    language: str | None = None


@dataclass
class ASRCapabilities:
    name: str
    native_streaming: bool
    partials: bool
    timestamps: bool
    confidence: bool
    languages: list[str]
    devices: list[str]
    sample_rate: int = 16000
    streaming_method: str = "native"  # native | chunked-redecode | none
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkResult:
    backend: str
    clips: int
    audio_s: float
    final_latency_ms: list[float] = field(default_factory=list)  # end-of-audio -> final text
    first_partial_ms: list[float] = field(default_factory=list)
    rtf: float | None = None
    wer: float | None = None
    load_ms: float | None = None
    peak_rss_mib: float | None = None
    hypotheses: list[str] = field(default_factory=list)


class ASRBackend(ABC):
    name: str = "asr"

    def initialize(self) -> None:
        """Check dependencies without loading weights. Raise ASRError with an install hint."""

    @abstractmethod
    def load_model(self) -> None: ...

    @abstractmethod
    def unload_model(self) -> None: ...

    @property
    def loaded(self) -> bool:
        return False

    @abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> TranscriptUpdate:
        """Whole-clip transcription (final)."""

    def transcribe_stream(self, chunk: np.ndarray, sample_rate: int = 16000) -> list[TranscriptUpdate]:
        """Feed audio of the current utterance; return any new partial updates."""
        raise StreamingNotSupported(f"{self.name} does not support native streaming")

    def reset_stream(self) -> None:
        """Drop the current utterance's state."""

    def finalize_utterance(self) -> TranscriptUpdate:
        raise StreamingNotSupported(f"{self.name} does not support native streaming")

    @abstractmethod
    def get_capabilities(self) -> ASRCapabilities: ...

    def benchmark(self, clips: Iterable[tuple[np.ndarray, int, str | None]], *, chunk_ms: int = 80) -> BenchmarkResult:
        from .bench import run_asr_benchmark

        return run_asr_benchmark(self, clips, chunk_ms=chunk_ms)

    def shutdown(self) -> None:
        try:
            self.unload_model()
        except Exception:  # noqa: BLE001
            pass


def streaming_view(backend: ASRBackend, **adapter_kw: Any) -> ASRBackend:
    """The backend itself if it streams natively, else a clearly labelled chunked adapter."""
    if backend.get_capabilities().native_streaming:
        return backend
    from .chunked import ChunkedStreamingAdapter

    return ChunkedStreamingAdapter(backend, **adapter_kw)


def to_mono_f32(audio: np.ndarray) -> np.ndarray:
    a = np.asarray(audio)
    if a.ndim > 1:
        a = a.mean(axis=1)
    if a.dtype == np.int16:
        return a.astype(np.float32) / 32768.0
    return a.astype(np.float32, copy=False)


def timed(fn, *a, **kw):
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    return out, (time.perf_counter() - t0) * 1000
