"""TTS backend interface."""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Iterator

import numpy as np


class TTSError(RuntimeError):
    pass


class RemoteNotAllowed(TTSError):
    pass


@dataclass
class AudioChunk:
    pcm: np.ndarray  # mono float32
    sample_rate: int


@dataclass
class TTSCapabilities:
    name: str
    streaming: bool  # audio arrives incrementally (verified for this transport)
    dialect: str  # expression dialect: drama3 | s2 | s1 | none
    remote: bool  # text / reference audio leave this machine
    destination: str | None = None
    voice_kinds: tuple[str, ...] = ()
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TTSBackend(ABC):
    name = "tts"

    @abstractmethod
    def capabilities(self) -> TTSCapabilities: ...

    def health(self) -> tuple[bool, str]:
        return True, "ok"

    def warmup(self, voice: Any | None = None) -> float | None:
        """Synthesize a short phrase to prime caches. Returns ms to first audio."""
        import time

        t0 = time.perf_counter()
        for _ in self.synthesize("Hello.", voice=voice):
            return (time.perf_counter() - t0) * 1000
        return None

    @abstractmethod
    def synthesize(
        self,
        text: str,
        *,
        voice: Any | None = None,
        cancel: threading.Event | None = None,
    ) -> Iterator[AudioChunk]: ...

    def close(self) -> None:
        pass
