"""Deterministic ASR stand-in for tests and pipeline benchmarks without model weights.

It "recognises" a known script in proportion to the audio fed so far and can
inject a revision (a wrong word that is corrected one pass later), which is the
behaviour the transcript stabilizer has to cope with. Its timing is simulated
(``cost_ms`` per pass), so pipeline numbers measured with it describe aivoice's
own overhead, never model speed.
"""

from __future__ import annotations

import time

import numpy as np

from .base import ASRBackend, ASRCapabilities, TranscriptUpdate


class ScriptedASR(ASRBackend):
    def __init__(
        self,
        scripts: list[str],
        *,
        words_per_s: float = 2.8,
        native_streaming: bool = True,
        cost_ms: float = 0.0,
        final_cost_ms: float = 0.0,
        revise_every: int = 0,
        name: str = "scripted",
    ) -> None:
        self.scripts = list(scripts)
        self.words_per_s = words_per_s
        self._native = native_streaming
        self.cost_ms = cost_ms
        self.final_cost_ms = final_cost_ms
        self.revise_every = revise_every
        self.name = name
        self._i = 0
        self._samples = 0
        self._rev = 0
        self._loaded = False
        self.calls = 0

    def load_model(self) -> None:
        self._loaded = True

    def unload_model(self) -> None:
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    def get_capabilities(self) -> ASRCapabilities:
        return ASRCapabilities(
            name=self.name,
            native_streaming=self._native,
            partials=self._native,
            timestamps=False,
            confidence=False,
            languages=["en"],
            devices=["cpu"],
            streaming_method="native" if self._native else "none",
            notes="scripted mock (no model)",
        )

    def _script(self) -> str:
        return self.scripts[self._i % len(self.scripts)] if self.scripts else ""

    def _partial(self) -> str:
        words = self._script().split()
        n = min(len(words), int(self._samples / 16000 * self.words_per_s))
        out = words[:n]
        if self.revise_every and n and self._rev % self.revise_every == 1:
            out = out[:-1] + [out[-1] + "x"]  # a wrong last word, fixed next pass
        return " ".join(out)

    def _sleep(self, ms: float) -> None:
        if ms > 0:
            time.sleep(ms / 1000)

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> TranscriptUpdate:
        self.calls += 1
        self._sleep(self.final_cost_ms)
        self._samples = int(np.asarray(audio).size * 16000 / sample_rate)
        text = self._partial()
        return TranscriptUpdate(text=text, is_final=True, backend=self.name, latency_ms=self.final_cost_ms)

    def transcribe_stream(self, chunk: np.ndarray, sample_rate: int = 16000) -> list[TranscriptUpdate]:
        if not self._native:
            return super().transcribe_stream(chunk, sample_rate)
        before = self._partial()
        self._samples += int(np.asarray(chunk).size * 16000 / sample_rate)
        self._rev += 1
        now = self._partial()
        if now == before or not now:
            return []
        self._sleep(self.cost_ms)
        return [TranscriptUpdate(text=now, is_final=False, revision=self._rev, backend=self.name, latency_ms=self.cost_ms)]

    def finalize_utterance(self) -> TranscriptUpdate:
        if not self._native:
            return super().finalize_utterance()
        self._sleep(self.final_cost_ms)
        text = self._script()
        self.reset_stream()
        self._i += 1
        return TranscriptUpdate(text=text, is_final=True, revision=self._rev + 1, backend=self.name, latency_ms=self.final_cost_ms)

    def reset_stream(self) -> None:
        if not self._native and self._samples:
            self._i += 1  # whole-clip mode: the next utterance gets the next script
        self._samples = 0
        self._rev = 0
