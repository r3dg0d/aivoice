"""Tone generator TTS for tests and pipeline benchmarks (no model, simulated timing)."""

from __future__ import annotations

import time
from typing import Iterator

import numpy as np

from .base import AudioChunk, TTSBackend, TTSCapabilities, TTSError


class ToneTTS(TTSBackend):
    def __init__(
        self,
        *,
        sample_rate: int = 24000,
        ttfa_ms: float = 0.0,
        rtf: float = 0.0,
        chunk_ms: int = 100,
        words_per_s: float = 3.0,
        streaming: bool = True,
        dialect: str = "drama3",
        fail: bool = False,
    ) -> None:
        self.sample_rate = sample_rate
        self.ttfa_ms = ttfa_ms
        self.rtf = rtf
        self.chunk_ms = chunk_ms
        self.words_per_s = words_per_s
        self.streaming = streaming
        self.dialect = dialect
        self.fail = fail
        self.name = "tone"
        self.requests: list[str] = []

    def capabilities(self) -> TTSCapabilities:
        return TTSCapabilities(self.name, self.streaming, self.dialect, False, notes="synthetic tones (mock)")

    def synthesize(self, text, *, voice=None, cancel=None) -> Iterator[AudioChunk]:
        self.requests.append(text)
        if self.fail:
            raise TTSError("mock failure")
        dur = max(0.2, len(text.split()) / self.words_per_s)
        n = int(dur * self.sample_rate)
        t = np.arange(n) / self.sample_rate
        audio = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        if self.ttfa_ms:
            time.sleep(self.ttfa_ms / 1000)
        step = int(self.sample_rate * self.chunk_ms / 1000) if self.streaming else n
        for i in range(0, n, step):
            if cancel is not None and cancel.is_set():
                return
            block = audio[i : i + step]
            if self.rtf and i:
                time.sleep(block.size / self.sample_rate * self.rtf)
            yield AudioChunk(block, self.sample_rate)
