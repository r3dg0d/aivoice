"""TVTSyn paper reproduction (aivoice code), verified against the TVTSyn backbone in the official VOSSA code."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..common.base import Reproduction, apply_overrides
from . import config as C
from .info import INFO


class TVTSynReproduction(Reproduction):
    info = INFO
    package_dir = Path(__file__).parent
    stages = {"content": "PASR content encoder + factorised VQ on HuBERT-L9 k-means-200 codes",
              "synth": "time-varying timbre (GTM + Slerp) + decoder with pitch/energy predictors, MPD/MSD"}
    presets = tuple(C.PRESETS)
    inference_stages = ("synth",)

    def config(self, preset: str = "paper", stage: str | None = None, overrides: dict[str, Any] | None = None):
        return apply_overrides(C.PRESETS[preset](stage or "synth"), overrides)

    def recipe(self, stage: str, config, manifest, **kw: Any):
        from .train.trainer import TVTSynRecipe

        config.stage = stage
        return TVTSynRecipe(config, manifest, **{k: v for k, v in kw.items() if k in (
            "val_manifest", "content_checkpoint", "cache_dir", "code_cache", "spk_cache", "datasets", "fixed_crops")})

    def prepare(self, config, dataset: str, out: str, **kw: Any) -> dict[str, Any]:
        from .train.preprocess import prepare

        return prepare(config, dataset, out, **kw)

    def load_streaming(self, checkpoint, device: str = "cpu", allow_untrained: bool = False, **kw: Any):
        from .streaming.inference import TVTSynStreaming

        return TVTSynStreaming.from_checkpoint(checkpoint, device, allow_untrained, kw.get("teacher_dirs"))

    def requirements(self) -> list[dict[str, Any]]:
        from ..common.teacher_paths import TeacherUnavailable, teacher_dir

        out = []
        for name, why in (("hubert-base", "k-means pseudo-labels (training only)"),
                          ("xvector", "512-d global speaker embedding (training + enrollment)"),
                          ("ecapa", "192-d global speaker embedding (training + enrollment)")):
            try:
                out.append({"name": name, "why": why, "ok": True, "path": str(teacher_dir(name))})
            except TeacherUnavailable as e:
                out.append({"name": name, "why": why, "ok": False, "hint": str(e)})
        return out


REPRODUCTION = TVTSynReproduction()
