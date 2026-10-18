"""NVIDIA Parakeet TDT 0.6B v3 via NeMo.

The v3 checkpoint is an offline (full-context) model: there is no cache-aware
streaming mode for it in NeMo. This backend therefore implements whole-clip
``transcribe`` only and reports ``native_streaming=False``; the engine streams
it through :class:`~aivoice.asr.chunked.ChunkedStreamingAdapter` (re-decoding),
and says so in ``experimental status``.

NeMo's ``ASRModel.transcribe`` accepts numpy arrays (verified in nemo_toolkit
3.0.0), so audio never round-trips through temp files.
"""

from __future__ import annotations

import inspect
import time
from pathlib import Path

import numpy as np

from ..audio.resample import resample
from .base import ASRBackend, ASRCapabilities, ASRError, TranscriptUpdate, Word, to_mono_f32

REPO_ID = "nvidia/parakeet-tdt-0.6b-v3"
LANGUAGES = [
    "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de", "el", "hu", "it",
    "lv", "lt", "mt", "pl", "pt", "ro", "sk", "sl", "es", "sv", "ru", "uk",
]


class ParakeetBackend(ASRBackend):
    name = "parakeet"

    def __init__(
        self,
        *,
        model: str = REPO_ID,
        nemo_path: str | None = None,
        device: str = "cuda",
        half: bool = True,
        timestamps: bool = False,
    ) -> None:
        self.model_id = model
        self.nemo_path = nemo_path
        self.device = device
        self.half = half
        self.timestamps = timestamps
        self._m = None
        self._transcribe_kw: dict = {}

    def initialize(self) -> None:
        import importlib.util

        if importlib.util.find_spec("nemo") is None:
            raise ASRError("NeMo is not installed: pip install 'nemo_toolkit[asr]' (plus a CUDA torch build)")

    def load_model(self) -> None:
        if self._m is not None:
            return
        self.initialize()
        import torch  # type: ignore
        from nemo.collections.asr.models import ASRModel  # type: ignore

        dev = self.device
        if dev.startswith("cuda") and not torch.cuda.is_available():
            raise ASRError("CUDA requested for Parakeet but torch reports no GPU; use --device cpu")
        if self.nemo_path:
            m = ASRModel.restore_from(self.nemo_path, map_location=dev)
        else:
            # Resolves through the HF cache; only reaches the network when not cached.
            m = ASRModel.from_pretrained(model_name=self.model_id, map_location=dev)
        m.eval()
        if self.half and dev.startswith("cuda"):
            m = m.half()
        self._m = m
        params = inspect.signature(m.transcribe).parameters
        self._transcribe_kw = {"batch_size": 1}
        if "verbose" in params:
            self._transcribe_kw["verbose"] = False
        if "timestamps" in params:
            self._transcribe_kw["timestamps"] = self.timestamps

    def unload_model(self) -> None:
        if self._m is None:
            return
        self._m = None
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    @property
    def loaded(self) -> bool:
        return self._m is not None

    def get_capabilities(self) -> ASRCapabilities:
        return ASRCapabilities(
            name=self.name,
            native_streaming=False,
            partials=False,
            timestamps=True,
            confidence=False,
            languages=LANGUAGES,
            devices=["cuda", "cpu"],
            sample_rate=16000,
            streaming_method="none",
            notes="Offline TDT model; streamed only via chunked re-decoding.",
        )

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> TranscriptUpdate:
        if self._m is None:
            self.load_model()
        x = to_mono_f32(audio)
        if sample_rate != 16000:
            x = resample(x, sample_rate, 16000)
        if x.size < 1600:
            return TranscriptUpdate("", True, backend=self.name, audio_s=x.size / 16000)
        import torch  # type: ignore

        t0 = time.perf_counter()
        with torch.inference_mode():
            out = self._m.transcribe([x], **self._transcribe_kw)
        if isinstance(out, tuple):  # some NeMo versions return (best, all)
            out = out[0]
        hyp = out[0] if out else ""
        text = getattr(hyp, "text", hyp) if not isinstance(hyp, str) else hyp
        words = None
        ts = getattr(hyp, "timestamp", None)
        if isinstance(ts, dict) and ts.get("word"):
            words = [Word(w.get("word", ""), w.get("start"), w.get("end")) for w in ts["word"]]
        return TranscriptUpdate(
            text=str(text).strip(),
            is_final=True,
            backend=self.name,
            words=words,
            latency_ms=(time.perf_counter() - t0) * 1000,
            audio_s=x.size / 16000,
        )


def find_nemo_file(snapshot: str | Path) -> str | None:
    files = sorted(Path(snapshot).glob("*.nemo"))
    return str(files[0]) if files else None
