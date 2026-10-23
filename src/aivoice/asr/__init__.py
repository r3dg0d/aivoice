"""Speech recognition backends behind one interface."""

from __future__ import annotations

import importlib.util
from typing import Any

from .base import ASRBackend, ASRCapabilities, ASRError, TranscriptUpdate, streaming_view

BACKENDS = ("moonshine-tiny", "moonshine-small", "moonshine-medium", "parakeet", "whisper-small-en")


def available_backends() -> dict[str, str | None]:
    """name -> None when usable, else why not (runtime missing)."""
    out: dict[str, str | None] = {}
    ms = None if importlib.util.find_spec("moonshine_voice") else "pip install 'aivoice[moonshine]'"
    for size in ("tiny", "small", "medium"):
        out[f"moonshine-{size}"] = ms
    out["parakeet"] = None if importlib.util.find_spec("nemo") else "pip install 'nemo_toolkit[asr]' (+ CUDA torch)"
    hf = importlib.util.find_spec("transformers") and importlib.util.find_spec("torch")
    out["whisper-small-en"] = None if hf else "pip install 'aivoice[repro]' (transformers + torch)"
    return out


def backend_config(name: str, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """The settings that change a backend's speed/accuracy (part of the benchmark cache key)."""
    cfg = cfg or {}
    # Must describe what create_backend() actually builds, or measurements recorded by the benchmark are never
    # found again (Parakeet's device comes from asr.parakeet_device, not the CPU-only asr.device).
    if name == "parakeet":
        return {"device": cfg.get("parakeet_device") or ("cuda" if _has_gpu() else "cpu"), "half": True}
    if name == "whisper-small-en":
        return {"language": "en", "device": "cuda" if _has_gpu() else "cpu"}
    return {"language": cfg.get("language", "en"), "device": "cpu"}


def create_backend(name: str, *, settings: dict[str, Any] | None = None, registry: Any | None = None) -> ASRBackend:
    """Build (not load) a backend. ``registry`` is a HuggingFaceModelManager for local paths."""
    s = settings or {}
    if name.startswith("moonshine-"):
        from .moonshine import MoonshineBackend

        size = name.split("-", 1)[1]
        entry = registry.get_entry(name) if registry else None
        return MoonshineBackend(
            size,
            language=s.get("language", "en"),
            model_path=entry["path"] if entry else None,
            allow_download=bool(s.get("allow_model_download")),
        )
    if name == "parakeet":
        from .parakeet import REPO_ID, ParakeetBackend, find_nemo_file

        entry = registry.get_entry(REPO_ID) if registry else None
        nemo = find_nemo_file(entry["path"]) if entry else None
        dev = s.get("parakeet_device") or ("cuda" if _has_gpu() else "cpu")
        return ParakeetBackend(nemo_path=nemo, device=dev)
    if name == "whisper-small-en":
        from .whisper_hf import WhisperHFBackend

        return WhisperHFBackend(allow_download=bool(s.get("allow_model_download")))
    if name == "scripted":
        from .mock import ScriptedASR

        return ScriptedASR(s.get("scripts") or ["hello world"])
    raise ASRError(f"unknown ASR backend {name!r}; choose from {', '.join(BACKENDS)}")


def _has_gpu() -> bool:
    from ..hardware import query_nvidia_smi

    return bool(query_nvidia_smi())


__all__ = [
    "ASRBackend",
    "ASRCapabilities",
    "ASRError",
    "BACKENDS",
    "TranscriptUpdate",
    "available_backends",
    "create_backend",
    "streaming_view",
]
