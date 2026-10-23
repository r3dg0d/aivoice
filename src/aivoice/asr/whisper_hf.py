"""Whisper (openai/whisper-small.en) through transformers: an offline, GPU, whole-clip ASR backend.

Used for WER in `aivoice benchmark vc`. Weights are safetensors at a pinned Hub revision (Apache-2.0); they
download once with your permission (`aivoice models` is not involved: transformers caches them in the HF cache).
Not a streaming backend: the live speech-to-speech mode keeps Moonshine/Parakeet.
"""

from __future__ import annotations

import time

import numpy as np

from .base import ASRBackend, ASRCapabilities, ASRError, TranscriptUpdate, to_mono_f32

REPO = "openai/whisper-small.en"
REVISION = "e8727524f962ee844a7319d92be39ac1bd25655a"


class WhisperHFBackend(ASRBackend):
    name = "whisper-small-en"

    def __init__(self, *, device: str | None = None, allow_download: bool = False) -> None:
        self.device = device
        self.allow_download = allow_download
        self._model = self._proc = None

    def initialize(self) -> None:
        import importlib.util

        if importlib.util.find_spec("transformers") is None or importlib.util.find_spec("torch") is None:
            raise ASRError("whisper-small-en needs torch + transformers (pip install 'aivoice[repro]')")

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load_model(self) -> None:
        self.initialize()
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        kw = {"revision": REVISION, "local_files_only": not self.allow_download}
        try:
            self._proc = WhisperProcessor.from_pretrained(REPO, **kw)
            self._model = WhisperForConditionalGeneration.from_pretrained(
                REPO, dtype=torch.float16 if dev == "cuda" else torch.float32, use_safetensors=True, **kw).to(dev).eval()
        except OSError as e:
            raise ASRError(f"{REPO}@{REVISION[:7]} is not in the local Hugging Face cache; allow the one-time "
                           "download (~970 MB) with allow_download=True / `aivoice benchmark vc --asr-download`") from e
        self.device = dev

    def unload_model(self) -> None:
        self._model = self._proc = None

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> TranscriptUpdate:
        import torch

        if self._model is None:
            self.load_model()
        x = to_mono_f32(audio)
        if sample_rate != 16000:
            from ..audio.resample import resample

            x = resample(x, sample_rate, 16000)
        t0 = time.perf_counter()
        texts = []
        for i in range(0, max(1, x.size), 16000 * 28):  # Whisper windows are 30 s
            feats = self._proc(x[i:i + 16000 * 28], sampling_rate=16000, return_tensors="pt").input_features
            feats = feats.to(self.device, self._model.dtype)
            with torch.no_grad():
                ids = self._model.generate(feats, max_new_tokens=220)
            texts.append(self._proc.batch_decode(ids, skip_special_tokens=True)[0].strip())
        return TranscriptUpdate(text=" ".join(t for t in texts if t), is_final=True, backend=self.name,
                                latency_ms=(time.perf_counter() - t0) * 1000, audio_s=x.size / 16000, language="en")

    def get_capabilities(self) -> ASRCapabilities:
        return ASRCapabilities(name=self.name, native_streaming=False, partials=False, timestamps=False, confidence=False,
                               languages=["en"], devices=["cuda", "cpu"], streaming_method="none",
                               notes=f"{REPO} @ {REVISION[:7]} (Apache-2.0, safetensors); offline evaluation backend")
