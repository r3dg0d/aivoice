"""Zero-VC paper reproduction (aivoice code; NOT an official implementation, no official weights exist)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..common.base import Reproduction
from . import config as C
from .info import INFO


class ZeroVCReproduction(Reproduction):
    info = INFO
    package_dir = Path(__file__).parent
    stages = {"content": "causal content student (variant A: w2v-BERT 2.0 distillation; B: HuBERT k-means CE)",
              "vc": "timbre pooling + causal HiFi-GAN decoder with MPD/MSD, anonymised-source reconstruction"}
    presets = tuple(C.PRESETS)
    inference_stages = ("vc",)

    def config(self, preset: str = "paper", stage: str | None = None, overrides: dict[str, Any] | None = None):
        from ..common.base import apply_overrides

        variant = (overrides or {}).get("variant", "A")
        return apply_overrides(C.PRESETS[preset](variant, stage or "vc"), overrides)

    def recipe(self, stage: str, config, manifest, **kw: Any):
        from .train.trainer import ZeroVCRecipe

        config.stage = stage
        return ZeroVCRecipe(config, manifest, **{k: v for k, v in kw.items() if k in (
            "val_manifest", "content_checkpoint", "teacher_dirs", "cache_dir", "teacher_cache", "timbre_cache",
            "datasets", "kmeans_centroids", "fixed_crops")})

    def prepare(self, config, dataset: str, out: str, **kw: Any) -> dict[str, Any]:
        from .train.preprocess import prepare

        return prepare(config, dataset, out, **kw)

    def load_streaming(self, checkpoint, device: str = "cpu", allow_untrained: bool = False, **kw: Any):
        from .streaming.inference import ZeroVCStreaming

        return ZeroVCStreaming.from_checkpoint(checkpoint, device, allow_untrained, kw.get("teacher_dirs"))

    def requirements(self) -> list[dict[str, Any]]:
        from ..common.teacher_paths import TeacherUnavailable, teacher_dir

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


REPRODUCTION = ZeroVCReproduction()
