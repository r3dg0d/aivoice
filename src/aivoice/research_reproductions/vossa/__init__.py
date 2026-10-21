"""VOSSA: integration of the official release (no reproduction needed)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..common.base import Reproduction
from .info import INFO

OFFICIAL_TRAIN = ("python src/main.py fit -c configs/model/vossa_synthesizer.yaml -c configs/data/resynthesis.yaml "
                  "(in your VOSSA checkout; content encoder and vocoder from checkpoints/, logging to W&B)")


class VossaReproduction(Reproduction):
    info = INFO
    package_dir = Path(__file__).parent
    stages = {"official": "official training CLI (not run by aivoice)"}
    presets = ("official",)

    def config(self, preset: str = "official", stage: str | None = None, overrides: dict[str, Any] | None = None):
        return {"preset": "official"}

    def recipe(self, stage: str, config, manifest, **kw: Any):
        raise NotImplementedError(f"VOSSA has official training code; run it directly: {OFFICIAL_TRAIN}")

    def load_streaming(self, checkpoint, device: str = "cpu", allow_untrained: bool = False, **kw: Any):
        raise NotImplementedError("VOSSA runs in the official worker: `aivoice live --engine vossa ...`")

    def status(self, run_dir=None) -> dict[str, Any]:
        from aivoice.vc.engines import VossaEngine

        st = VossaEngine().status()
        return {"engine": INFO.key, "source_status": INFO.source_status.value, "arch_confidence": "HIGH",
                "state": "Ready" if st.state.value in ("READY", "EXPERIMENTAL") else "Blocked" if st.state.value ==
                "BLOCKED_LICENSE" else "Not prepared", "why": st.detail, "official_weights": True}

    def requirements(self) -> list[dict[str, Any]]:
        from aivoice.vc.engines import VossaEngine

        st = VossaEngine().status()
        return [{"name": "VOSSA checkout + checkpoints + Python 3.10/torch 2.8 env", "why": "official engine",
                 "ok": st.state.value in ("READY", "EXPERIMENTAL"), "hint": st.detail}]


REPRODUCTION = VossaReproduction()
