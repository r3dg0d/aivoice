"""RT-VC: aivoice reconstruction around the official (unlicensed, weight-less) model layout."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..common.base import Reproduction
from ..common.teachers import TeacherUnavailable, teacher_dir
from . import config as C
from .info import INFO


class RTVCReproduction(Reproduction):
    info = INFO
    package_dir = Path(__file__).parent
    stages = {"pitch": "causal pitch / periodicity / loudness tracker (log-mel, 200 Hz)",
              "ema": "causal articulatory (EMA) inverter (MFCC, 200 Hz) on SPARC pseudo-labels",
              "vocoder": "speaker encoder + DDSP vocoder, multi-scale spectral loss"}
    presets = tuple(C.PRESETS)

    def config(self, preset: str = "paper", stage: str | None = None, overrides: dict[str, Any] | None = None):
        from ..common.base import apply_overrides

        return apply_overrides(C.PRESETS[preset](stage or "vocoder"), overrides)

    def recipe(self, stage: str, config, manifest, **kw: Any):
        from .train.trainer import RTVCRecipe

        config.stage = stage
        return RTVCRecipe(config, manifest, **{k: v for k, v in kw.items() if k in (
            "val_manifest", "cache_dir", "ema_cache", "pitch_cache", "teacher_dirs", "datasets", "fixed_crops")})

    def prepare(self, config, dataset: str, out: str, **kw: Any) -> dict[str, Any]:
        from .train.preprocess import prepare

        return prepare(config, dataset, out, **kw)

    def load_streaming(self, checkpoint, device: str = "cpu", allow_untrained: bool = False, **kw: Any):
        from .streaming.inference import RTVCStreaming

        return RTVCStreaming.from_checkpoint(checkpoint, device, allow_untrained, kw.get("teacher_dirs"))

    def requirements(self) -> list[dict[str, Any]]:
        out = []
        try:
            out.append({"name": "wavlm-large", "why": "CNN feature extractor of the speaker encoder", "ok": True,
                        "path": str(teacher_dir("wavlm-large"))})
        except TeacherUnavailable as e:
            out.append({"name": "wavlm-large", "why": "CNN feature extractor of the speaker encoder", "ok": False,
                        "hint": str(e)})
        try:
            import sparc  # noqa: F401

            out.append({"name": "sparc", "why": "EMA pseudo-labels (training only)", "ok": True})
        except ImportError:
            out.append({"name": "sparc", "why": "EMA pseudo-labels (training only)", "ok": False,
                        "hint": "pip install speech-articulatory-coding"})
        return out


REPRODUCTION = RTVCReproduction()
