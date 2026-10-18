"""Moonshine Voice streaming ASR (moonshine-voice package, C++/ONNX runtime on CPU).

Native streaming: audio is appended to a ``Stream`` whose encoder state is
cached by the runtime, and the runtime emits LineStarted / LineTextChanged /
LineCompleted events as it decodes. Verified against moonshine-voice 0.1.5.

Model files are not on the Hugging Face Hub in the runtime's format; the
runtime's own downloader fetches ``.ort`` files from download.moonshine.ai.
aivoice only lets it do that when downloads were explicitly allowed
(``aivoice models download moonshine-small --yes`` or
``asr.allow_model_download``); otherwise it loads from an already registered path.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from ..audio.resample import resample
from .base import ASRBackend, ASRCapabilities, ASRError, TranscriptUpdate, Word, to_mono_f32

SIZES = ("tiny", "small", "medium")


def _mv():
    try:
        import moonshine_voice as mv  # type: ignore
    except ImportError as e:
        raise ASRError("moonshine-voice is not installed: pip install 'aivoice[moonshine]'") from e
    return mv


class MoonshineBackend(ASRBackend):
    def __init__(
        self,
        size: str = "small",
        *,
        language: str = "en",
        model_path: str | None = None,
        update_interval: float = 0.2,
        allow_download: bool = False,
        options: dict[str, Any] | None = None,
    ) -> None:
        if size not in SIZES:
            raise ASRError(f"moonshine size must be one of {SIZES}")
        self.size = size
        self.name = f"moonshine-{size}"
        self.language = language
        self.model_path = model_path
        self.update_interval = update_interval
        self.allow_download = allow_download
        self.options = options or {}
        self._tr = None
        self._stream = None
        self._lines: dict[int, str] = {}
        self._order: list[int] = []
        self._revision = 0
        self._new: list[TranscriptUpdate] = []
        self._samples = 0
        self._last_words: list[Word] | None = None

    def initialize(self) -> None:
        _mv()

    def _arch(self):
        mv = _mv()
        return getattr(mv.ModelArch, f"{self.size.upper()}_STREAMING")

    def load_model(self) -> None:
        if self._tr is not None:
            return
        mv = _mv()
        arch = self._arch()
        path = self.model_path
        if path is None:
            if not self.allow_download:
                raise ASRError(
                    f"{self.name} is not registered locally. Download it explicitly: "
                    f"`aivoice models download {self.name} --yes`"
                )
            path, arch = mv.get_model_for_language(self.language, arch)
        try:
            self._tr = mv.Transcriber(model_path=str(path), model_arch=arch, update_interval=self.update_interval, options=self.options or None)
        except Exception as e:  # noqa: BLE001
            raise ASRError(f"moonshine failed to load {path}: {e}") from e

    def unload_model(self) -> None:
        self._close_stream()
        if self._tr is not None:
            self._tr.close()
            self._tr = None

    @property
    def loaded(self) -> bool:
        return self._tr is not None

    def get_capabilities(self) -> ASRCapabilities:
        return ASRCapabilities(
            name=self.name,
            native_streaming=True,
            partials=True,
            timestamps=True,
            confidence=True,
            languages=[self.language],
            devices=["cpu"],
            sample_rate=16000,
            streaming_method="native",
            notes="Moonshine Voice streaming model with cached encoder state (CPU).",
        )

    # --- streaming -------------------------------------------------------

    def _listener(self, event) -> None:
        line = event.line
        kind = type(event).__name__
        if kind == "Error":
            return
        lid = int(line.line_id)
        if lid not in self._lines:
            self._order.append(lid)
        self._lines[lid] = line.text or ""
        if getattr(line, "words", None):
            self._last_words = [Word(w.word, w.start, w.end, w.confidence) for w in line.words]
        if kind in ("LineTextChanged", "LineStarted", "LineCompleted"):
            self._revision += 1
            self._new.append(
                TranscriptUpdate(
                    text=self._text(),
                    is_final=False,
                    revision=self._revision,
                    backend=self.name,
                    words=self._last_words,
                    latency_ms=float(getattr(line, "last_transcription_latency_ms", 0) or 0),
                    audio_s=self._samples / 16000,
                    language=self.language,
                )
            )

    def _text(self) -> str:
        return " ".join(t.strip() for t in (self._lines[i] for i in self._order) if t.strip())

    def _ensure_stream(self):
        if self._tr is None:
            self.load_model()
        if self._stream is None:
            self._stream = self._tr.create_stream(update_interval=self.update_interval)
            self._stream.add_listener(self._listener)
            self._stream.start()
        return self._stream

    def transcribe_stream(self, chunk: np.ndarray, sample_rate: int = 16000) -> list[TranscriptUpdate]:
        s = self._ensure_stream()
        x = to_mono_f32(chunk)
        self._samples += int(x.size * 16000 / sample_rate)
        s.add_audio(x.tolist(), sample_rate)
        out, self._new = self._new, []
        return out

    def finalize_utterance(self) -> TranscriptUpdate:
        t0 = time.perf_counter()
        if self._stream is not None:
            self._stream.stop()  # flushes: remaining lines complete
        text = self._text()
        upd = TranscriptUpdate(
            text=text,
            is_final=True,
            revision=self._revision + 1,
            backend=self.name,
            words=self._last_words,
            latency_ms=(time.perf_counter() - t0) * 1000,
            audio_s=self._samples / 16000,
            language=self.language,
        )
        self.reset_stream()
        return upd

    def _close_stream(self) -> None:
        if self._stream is not None:
            try:
                self._stream.close()
            except Exception:  # noqa: BLE001
                pass
            self._stream = None

    def reset_stream(self) -> None:
        self._close_stream()
        self._lines.clear()
        self._order.clear()
        self._new.clear()
        self._revision = 0
        self._samples = 0
        self._last_words = None

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> TranscriptUpdate:
        if self._tr is None:
            self.load_model()
        x = to_mono_f32(audio)
        if sample_rate != 16000:
            x = resample(x, sample_rate, 16000)
        t0 = time.perf_counter()
        tr = self._tr.transcribe_without_streaming(x.tolist(), 16000)
        text = " ".join(line.text.strip() for line in tr.lines if line.text.strip())
        return TranscriptUpdate(text=text, is_final=True, backend=self.name, latency_ms=(time.perf_counter() - t0) * 1000, audio_s=x.size / 16000)
