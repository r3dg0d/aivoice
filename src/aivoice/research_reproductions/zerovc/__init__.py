"""Zero-VC paper reproduction (aivoice code; NOT an official implementation, no official weights exist)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..common.base import Reproduction
from ..common.teachers import TeacherUnavailable, teacher_dir
from . import config as C
from .info import INFO


class ZeroVCReproduction(Reproduction):
    info = INFO
    package_dir = Path(__file__).parent
    stages = {"content": "causal content student (variant A: w2v-BERT 2.0 distillation; B: HuBERT k-means CE)",
              "vc": "timbre pooling + causal HiFi-GAN decoder with MPD/MSD, anonymised-source reconstruction"}
    presets = tuple(C.PRESETS)

    def config(self, preset: str = "paper", stage: str | None = None, overrides: dict[str, Any] | None = None):
        variant = (overrides or {}).get("variant", "A")
        cfg = C.PRESETS[preset](variant, stage or "vc")
        for k, v in (overrides or {}).items():
            if k != "variant":
                _set(cfg, k, v)
        return cfg

    def recipe(self, stage: str, config, manifest, **kw: Any):
        from .train.trainer import ZeroVCRecipe

        config.stage = stage
        return ZeroVCRecipe(config, manifest, **kw)

    def prepare(self, config, dataset: str, out: str, **kw: Any) -> dict[str, Any]:
        from .train.preprocess import prepare

        return prepare(config, dataset, out, **kw)

    def load_streaming(self, checkpoint, device: str = "cpu", allow_untrained: bool = False, **kw: Any):
        from .streaming.inference import ZeroVCStreaming

        return ZeroVCStreaming.from_checkpoint(checkpoint, device, allow_untrained, kw.get("teacher_dirs"))

    def requirements(self) -> list[dict[str, Any]]:
        out = []
        for name, why in (("wavlm-large", "timbre teacher (stage vc, and per-voice enrollment at inference)"),
                          ("w2v-bert-2.0", "content teacher, variant A (training only)"),
                          ("hubert-base", "content teacher, variant B (training only)")):
            try:
                out.append({"name": name, "why": why, "ok": True, "path": str(teacher_dir(name))})
            except TeacherUnavailable as e:
                out.append({"name": name, "why": why, "ok": False, "hint": str(e)})
        out.append({"name": "LibriTTS", "why": "training data (paper: LibriTTS, >= 4 s, 16 kHz)", "ok": None,
                    "hint": "pass --dataset libritts:/path/to/LibriTTS"})
        return out


def _set(cfg: Any, dotted: str, value: Any) -> None:
    obj = cfg
    parts = dotted.split(".")
    for p in parts[:-1]:
        obj = getattr(obj, p)
    if not hasattr(obj, parts[-1]):
        raise KeyError(f"unknown config key {dotted!r}")
    old = getattr(obj, parts[-1])
    setattr(obj, parts[-1], type(old)(value) if old is not None and not isinstance(old, tuple) else value)


REPRODUCTION = ZeroVCReproduction()
