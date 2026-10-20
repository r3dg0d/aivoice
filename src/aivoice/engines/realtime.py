"""Use any TTS engine in the experimental real-time pipeline (tts.backend = "engine:<name>")."""

from __future__ import annotations

from typing import Any, Iterator

from ..tts.base import AudioChunk, TTSBackend, TTSCapabilities, TTSError
from .base import EngineError, SpeakerRef, SynthesisOptions


class EngineTTSBackend(TTSBackend):
    def __init__(self, engine: Any, speaker: SpeakerRef | None = None, options: SynthesisOptions | None = None) -> None:
        self.engine = engine
        self.speaker = speaker
        self.options = options or SynthesisOptions()
        self.name = f"engine:{engine.capabilities().name}"

    def capabilities(self) -> TTSCapabilities:
        c = self.engine.capabilities()
        return TTSCapabilities(
            name=self.name,
            streaming=c.streaming,
            dialect="none",  # engines get plain text; style travels in SynthesisOptions
            remote=c.remote,
            destination=None,
            voice_kinds=(),
            notes=c.notes,
        )

    def health(self) -> tuple[bool, str]:
        why = self.engine.available()
        return (why is None, why or "ok")

    def synthesize(self, text, *, voice=None, cancel=None) -> Iterator[AudioChunk]:
        try:
            if self.engine.capabilities().streaming and hasattr(self.engine, "synthesize_stream"):
                for pcm, sr in self.engine.synthesize_stream(text, self.speaker, self.options):
                    if cancel is not None and cancel.is_set():
                        return
                    yield AudioChunk(pcm, sr)
                return
            r = self.engine.synthesize(text, self.speaker, self.options)
            if cancel is None or not cancel.is_set():
                yield AudioChunk(r.audio, r.sample_rate)
        except EngineError as e:
            raise TTSError(str(e)) from e

    def close(self) -> None:
        self.engine.unload()
