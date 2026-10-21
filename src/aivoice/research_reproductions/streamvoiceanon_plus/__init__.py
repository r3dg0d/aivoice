"""StreamVoiceAnon+ fine-tuning extension of the official StreamVoiceAnon model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..common.base import Reproduction, apply_overrides
from . import config as C
from .info import INFO


class PlusReproduction(Reproduction):
    info = INFO
    package_dir = Path(__file__).parent
    stages = {"finetune": "official base + SEP tokens + emotion2vec distillation on (neutral prompt, emotional source)"}
    presets = tuple(C.PRESETS)

    def config(self, preset: str = "paper", stage: str | None = None, overrides: dict[str, Any] | None = None):
        return apply_overrides(C.PRESETS[preset](), overrides)

    def recipe(self, stage: str, config, manifest, **kw: Any):
        from .train.trainer import PlusRecipe

        return PlusRecipe(config, manifest, **{k: v for k, v in kw.items() if k in (
            "val_manifest", "caches", "repo_dir", "trust_pickle", "datasets", "fixed_crops")})

    def prepare(self, config, dataset: str, out: str, **kw: Any) -> dict[str, Any]:
        from .train.preprocess import prepare

        return prepare(config, dataset, out, **kw)

    def load_streaming(self, checkpoint, device: str = "cpu", allow_untrained: bool = False, **kw: Any):
        raise NotImplementedError("StreamVoiceAnon+ runs through the official StreamVoiceAnon streaming path "
                                  "(CUDA): `aivoice reproduce export streamvoiceanon-plus` writes an official-format "
                                  "checkpoint + sep.pt for engine `streamvoiceanon`")

    def requirements(self) -> list[dict[str, Any]]:
        from .model.official import repo_dir

        out = []
        try:
            out.append({"name": "StreamVoiceAnon checkout", "why": "official base model code", "ok": True,
                        "path": str(repo_dir())})
        except FileNotFoundError as e:
            out.append({"name": "StreamVoiceAnon checkout", "why": "official base model code", "ok": False, "hint": str(e)})
        try:
            import funasr  # noqa: F401

            out.append({"name": "emotion2vec (funasr)", "why": "frame-level emotion teacher (training only)", "ok": True})
        except ImportError:
            out.append({"name": "emotion2vec (funasr)", "why": "frame-level emotion teacher (training only)", "ok": False,
                        "hint": "pip install funasr; emotion2vec/emotion2vec_plus_base on disk"})
        return out


REPRODUCTION = PlusReproduction()
