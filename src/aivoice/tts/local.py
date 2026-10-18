"""Local fallback TTS: the same engines `voices import-rvc` already uses (command, Piper, espeak-ng).

Not streaming: each engine renders a whole file, which is then played. It
exists so experimental mode keeps speaking when the Fish server is down, never
as a silent switch to the cloud.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Iterator

from .base import AudioChunk, TTSBackend, TTSCapabilities, TTSError


class LocalEngineTTS(TTSBackend):
    def __init__(self, engine_name: str | None = None) -> None:
        from ..rvc.synth import default_tts_engines

        engines = [e for e in default_tts_engines() if e.unavailable_reason() is None]
        if engine_name:
            engines = [e for e in engines if e.name == engine_name]
        if not engines:
            raise TTSError(
                "no local TTS engine available: install espeak-ng, or set AIVOICE_PIPER_MODEL / AIVOICE_TTS_CMD"
            )
        self.engine = engines[0]
        self.name = f"local:{self.engine.name}"

    def capabilities(self) -> TTSCapabilities:
        return TTSCapabilities(
            name=self.name,
            streaming=False,
            dialect="none",
            remote=False,
            voice_kinds=(),
            notes="whole-utterance rendering; ignores voice profiles and expression tags",
        )

    def synthesize(self, text, *, voice=None, cancel=None) -> Iterator[AudioChunk]:
        from ..audio_io import read_wav
        from .expression_strip import strip_tags

        clean = strip_tags(text)
        if not clean.strip():
            return
        with tempfile.TemporaryDirectory(prefix="aivoice-tts-") as td:
            out = Path(td) / "out.wav"
            try:
                self.engine.synthesize(clean, out)
            except Exception as e:  # noqa: BLE001
                raise TTSError(f"{self.name} failed: {e}") from e
            if cancel is not None and cancel.is_set():
                return
            pcm, sr = read_wav(out)
        yield AudioChunk(pcm=pcm, sample_rate=sr)
