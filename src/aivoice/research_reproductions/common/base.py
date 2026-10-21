"""Common interface of a reproduction package (what the registry, CLI and research panel talk to)."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from .checkpoint import is_usable_for_inference, read_meta
from .provenance import ReproductionInfo, SourceStatus

# Panel / CLI states
NOT_PREPARED = "Not prepared"
ARCH_READY = "Architecture ready"
TRAINING = "Training"
CHECKPOINT = "Checkpoint available"
READY = "Ready"
FAILED = "Failed"
BLOCKED = "Blocked"


class Reproduction:
    """Subclassed once per paper. Subclasses set ``info`` and ``stages`` and implement the factories."""

    info: ReproductionInfo
    stages: dict[str, str] = {}
    package_dir: Path

    def config(self, preset: str = "paper", stage: str | None = None, overrides: dict[str, Any] | None = None) -> Any:
        raise NotImplementedError

    def recipe(self, stage: str, config: Any, manifest: str | None, **kw: Any):
        raise NotImplementedError

    def load_streaming(self, checkpoint: str | Path, device: str = "cpu", allow_untrained: bool = False, **kw: Any):
        raise NotImplementedError

    def requirements(self) -> list[dict[str, Any]]:
        """External things a full run needs (teachers, data, official code), each with ``ok``."""
        return []

    # -- generic --------------------------------------------------------------------------------
    def spec_text(self) -> str:
        p = self.package_dir / "SPEC.md"
        return p.read_text() if p.exists() else "(no SPEC.md)"

    def status(self, run_dir: str | Path | None = None) -> dict[str, Any]:
        info = self.info
        st: dict[str, Any] = {"engine": info.key, "source_status": info.source_status.value,
                              "arch_confidence": info.arch_confidence.value}
        if info.source_status == SourceStatus.BLOCKED:
            return {**st, "state": BLOCKED, "why": info.blocked_by}
        if run_dir is None:
            return {**st, "state": ARCH_READY, "why": "architecture implemented; no training run given"}
        run = Path(run_dir)
        status_file = run / "status.json"
        live = json.loads(status_file.read_text()) if status_file.exists() else {}
        ckpts = sorted((run / "checkpoints").glob("*.pt")) if (run / "checkpoints").exists() else []
        st["run"] = live
        if live.get("state") in ("training", "paused", "starting"):
            st["state"] = TRAINING
        elif live.get("state") == "failed":
            st["state"] = FAILED
        elif ckpts:
            meta = read_meta(ckpts[-1])
            ok, why = is_usable_for_inference(meta)
            st.update(state=READY if ok and meta.get("extra_ready", True) else CHECKPOINT, checkpoint=str(ckpts[-1]),
                      weights=why, step=meta.get("step"))
        elif not (run / "prepared.json").exists():
            st["state"] = NOT_PREPARED
        else:
            st["state"] = ARCH_READY
        return st


def to_dict(cfg: Any) -> dict[str, Any]:
    return asdict(cfg) if is_dataclass(cfg) else dict(cfg)


__all__ = ["ARCH_READY", "BLOCKED", "CHECKPOINT", "FAILED", "NOT_PREPARED", "READY", "Reproduction", "TRAINING",
           "to_dict"]
