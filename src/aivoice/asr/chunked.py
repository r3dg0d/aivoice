"""Chunked re-decoding adapter for whole-clip ASR models (e.g. Parakeet TDT).

This is *not* streaming inference. Every ``interval_ms`` of new audio it
re-decodes the utterance so far (bounded by ``max_window_s``) and reports the
result as a partial. Partials therefore cost a full decode each, and their text
can change anywhere, which is why the engine runs them through the transcript
stabilizer before anything is spoken. The interval stretches automatically when
a decode takes longer than the audio it covers, so a slow machine degrades to
fewer partials instead of falling ever further behind.
"""

from __future__ import annotations

import time

import numpy as np

from ..audio.resample import resample
from .base import ASRBackend, ASRCapabilities, TranscriptUpdate, to_mono_f32


class ChunkedStreamingAdapter(ASRBackend):
    def __init__(
        self,
        inner: ASRBackend,
        *,
        interval_ms: int = 480,
        max_window_s: float = 20.0,
        min_audio_ms: int = 320,
        final_window_s: float = 60.0,
    ) -> None:
        self.inner = inner
        self.name = f"{inner.name}+chunked"
        self.interval_ms = interval_ms
        self.max_window_s = max_window_s
        self.min_audio_ms = min_audio_ms
        self.final_window_s = final_window_s
        self._buf: list[np.ndarray] = []
        self._samples = 0
        self._since = 0
        self._revision = 0
        self._last_cost_ms = 0.0
        self._last_text = ""
        self._sr = inner.get_capabilities().sample_rate

    def initialize(self) -> None:
        self.inner.initialize()

    def load_model(self) -> None:
        self.inner.load_model()

    def unload_model(self) -> None:
        self.inner.unload_model()

    @property
    def loaded(self) -> bool:
        return self.inner.loaded

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> TranscriptUpdate:
        return self.inner.transcribe(audio, sample_rate)

    def get_capabilities(self) -> ASRCapabilities:
        c = self.inner.get_capabilities()
        return ASRCapabilities(
            name=self.name,
            native_streaming=False,
            partials=True,
            timestamps=c.timestamps,
            confidence=c.confidence,
            languages=c.languages,
            devices=c.devices,
            sample_rate=c.sample_rate,
            streaming_method="chunked-redecode",
            notes=f"chunked re-decoding every {self.interval_ms} ms (not native streaming). {c.notes}".strip(),
        )

    def _audio(self, window_s: float | None = None) -> np.ndarray:
        a = np.concatenate(self._buf) if self._buf else np.zeros(0, np.float32)
        max_n = int((window_s or self.max_window_s) * self._sr)
        return a[-max_n:] if a.size > max_n else a

    def transcribe_stream(self, chunk: np.ndarray, sample_rate: int = 16000) -> list[TranscriptUpdate]:
        x = to_mono_f32(chunk)
        if sample_rate != self._sr:
            x = resample(x, sample_rate, self._sr)
        self._buf.append(x)
        self._samples += x.size
        self._since += x.size
        interval = max(self.interval_ms, self._last_cost_ms * 1.25)
        if self._since * 1000 / self._sr < interval or self._samples * 1000 / self._sr < self.min_audio_ms:
            return []
        self._since = 0
        t0 = time.perf_counter()
        res = self.inner.transcribe(self._audio(), self._sr)
        self._last_cost_ms = (time.perf_counter() - t0) * 1000
        if res.text == self._last_text:
            return []
        self._last_text = res.text
        self._revision += 1
        return [
            TranscriptUpdate(
                text=res.text,
                is_final=False,
                revision=self._revision,
                backend=self.name,
                words=res.words,
                latency_ms=self._last_cost_ms,
                audio_s=self._samples / self._sr,
            )
        ]

    def finalize_utterance(self) -> TranscriptUpdate:
        t0 = time.perf_counter()
        # One decode of the whole utterance (partials are bounded to max_window_s for speed; the final used to be
        # too, which dropped everything older than 20 s from long utterances).
        audio = self._audio(self.final_window_s)
        res = self.inner.transcribe(audio, self._sr) if audio.size else TranscriptUpdate("", True)
        text = res.text if res.text.strip() else self._last_text  # never finalize emptier than the last partial
        out = TranscriptUpdate(
            text=text,
            is_final=True,
            revision=self._revision + 1,
            backend=self.name,
            words=res.words,
            latency_ms=(time.perf_counter() - t0) * 1000,
            audio_s=self._samples / self._sr,
        )
        self.reset_stream()
        return out

    def reset_stream(self) -> None:
        self.inner.reset_stream()
        self._buf.clear()
        self._samples = self._since = self._revision = 0
        self._last_text = ""
