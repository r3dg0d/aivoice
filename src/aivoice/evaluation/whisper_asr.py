"""Whisper (faster-whisper / CTranslate2) as an *evaluation* ASR for WER/CER.

Not a real-time backend: it transcribes whole clips. Used by the research
benchmark when ``AIVOICE_EVAL_ASR=whisper`` or when no Moonshine/Parakeet
backend is registered. ``pip install faster-whisper``; the model is fetched by
faster-whisper from the Hugging Face Hub on first use.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..asr.base import ASRCapabilities

DEFAULT_MODEL = "large-v3"


@dataclass
class _Result:
    text: str


class WhisperEvalASR:
    def __init__(self, model: str | None = None, device: str | None = None) -> None:
        self.model_name = model or os.environ.get("AIVOICE_EVAL_WHISPER_MODEL", DEFAULT_MODEL)
        self.device = device or os.environ.get("AIVOICE_EVAL_WHISPER_DEVICE")
        self.model: Any = None
        self.compute_type = ""

    def get_capabilities(self) -> ASRCapabilities:
        return ASRCapabilities(name=f"whisper-{self.model_name} ({self.device or 'auto'}, {self.compute_type or '?'})",
                               native_streaming=False, partials=False, timestamps=False, confidence=False,
                               languages=["multilingual"], devices=["cuda", "cpu"], streaming_method="none",
                               notes="evaluation only: whole-clip transcription")

    def load_model(self) -> None:
        from faster_whisper import WhisperModel  # type: ignore

        tries = [(self.device, "float16" if self.device == "cuda" else "int8")] if self.device else [("cuda", "float16"), ("cpu", "int8")]
        last: Exception | None = None
        for dev, ct in tries:
            try:
                self.model = WhisperModel(self.model_name, device=dev, compute_type=ct)
                self.device, self.compute_type = dev, ct
                return
            except Exception as e:  # noqa: BLE001 - CUDA libraries missing -> try CPU
                last = e
        raise RuntimeError(f"faster-whisper could not load {self.model_name}: {last}")

    def transcribe(self, x: np.ndarray, sr: int, language: str | None = "en") -> _Result:
        from ..audio.resample import resample

        if self.model is None:
            self.load_model()
        a = np.asarray(x, np.float32)
        if sr != 16000:
            a = resample(a, sr, 16000)
        try:
            text = self._run(a, language)
        except RuntimeError as e:
            # CTranslate2 loads cuBLAS/cuDNN lazily: a missing library shows up here, not in load_model().
            if self.device != "cuda" or not any(k in str(e).lower() for k in ("cublas", "cudnn", "cuda")):
                raise
            self.device = "cpu"
            self.load_model()
            text = self._run(a, language)
        return _Result(text)

    def _run(self, a: np.ndarray, language: str | None) -> str:
        segs, _ = self.model.transcribe(a, language=language, beam_size=5, vad_filter=False,
                                        condition_on_previous_text=False)
        return " ".join(s.text.strip() for s in segs).strip()
