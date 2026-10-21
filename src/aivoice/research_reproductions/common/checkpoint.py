"""Self-describing checkpoints with strict, verified loading and exact resume.

A checkpoint is ``<name>.pt`` plus ``<name>.pt.sha256`` and ``<name>.json`` (the metadata, readable
without torch). The metadata records what the weights are (engine, model name, provenance labels,
licenses of everything that went into them), how they were made (config + fingerprint, git commit,
environment, datasets, step) and the parameter shapes, so a loader can refuse a mismatched file
before touching any tensor.

Loading uses ``torch.load(weights_only=True)``: aivoice checkpoints contain only tensors and plain
Python values, so no pickle code ever runs.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .provenance import WeightsStatus, weights_label

FORMAT = "aivoice-repro-checkpoint/1"


class CheckpointError(RuntimeError):
    pass


def config_fingerprint(config: dict[str, Any]) -> str:
    """Stable hash of the architecture-relevant config (sorted JSON)."""
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def git_state(path: str | Path | None = None) -> dict[str, Any]:
    root = Path(path or Path(__file__).resolve().parents[4])
    try:
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False,
                                timeout=5).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                                    capture_output=True, text=True, timeout=5).stdout.strip())
        return {"commit": commit or None, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def environment() -> dict[str, Any]:
    from aivoice import __version__

    env: dict[str, Any] = {
        "aivoice": __version__, "python": sys.version.split()[0], "platform": platform.platform(),
        "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "numpy": np.__version__, "git": git_state(),
    }
    if torch.cuda.is_available():
        env["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return env


def rng_state() -> dict[str, Any]:
    st: dict[str, Any] = {"python": list(_listify(random.getstate())), "torch": torch.get_rng_state()}
    np_state = np.random.get_state()
    st["numpy"] = [np_state[0], torch.from_numpy(np_state[1].astype(np.int64)), *np_state[2:]]
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(st: dict[str, Any]) -> None:
    random.setstate(_tuplify(st["python"]))
    torch.set_rng_state(st["torch"])
    n = st["numpy"]
    np.random.set_state((n[0], n[1].numpy().astype(np.uint32), *n[2:]))
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


def _listify(x: Any) -> Any:
    return [_listify(v) for v in x] if isinstance(x, (tuple, list)) else x


def _tuplify(x: Any) -> Any:
    return tuple(_tuplify(v) for v in x) if isinstance(x, list) else x


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@dataclass
class CheckpointMeta:
    engine: str
    model_name: str                       # e.g. "aivoice/zerovc-paper-repro-v0.1"
    weights_status: str                   # WeightsStatus value
    source_status: str
    arch_confidence: str
    config: dict[str, Any]
    step: int = 0
    epoch: int = 0
    datasets: list[str] = field(default_factory=list)
    component_licenses: dict[str, str] = field(default_factory=dict)  # e.g. {"wavlm-large teacher": "MIT"}
    redistributable: bool = False
    notes: str = ""
    # filled in by save_checkpoint
    format: str = FORMAT
    config_fingerprint: str = ""
    created: float = 0.0
    environment: dict[str, Any] = field(default_factory=dict)
    param_shapes: dict[str, dict[str, list[int]]] = field(default_factory=dict)
    trainable_params: dict[str, int] = field(default_factory=dict)
    weights_label: str = ""


def save_checkpoint(path: str | Path, meta: CheckpointMeta, models: dict[str, nn.Module], *,
                    optimizers: dict[str, torch.optim.Optimizer] | None = None,
                    schedulers: dict[str, Any] | None = None, scaler: Any = None,
                    sampler_state: dict[str, Any] | None = None, extra: dict[str, Any] | None = None) -> Path:
    """Write atomically (tmp + rename) with checksum and JSON sidecar. Returns the .pt path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta.config_fingerprint = config_fingerprint(meta.config)
    meta.created = time.time()
    meta.environment = environment()
    meta.weights_label = weights_label(meta.weights_status)
    meta.param_shapes = {n: {k: list(v.shape) for k, v in _unwrap(m).state_dict().items()} for n, m in models.items()}
    meta.trainable_params = {n: sum(p.numel() for p in m.parameters() if p.requires_grad) for n, m in models.items()}
    blob: dict[str, Any] = {
        "format": FORMAT,
        "meta": json.loads(json.dumps(meta.__dict__, default=str)),
        "models": {n: _unwrap(m).state_dict() for n, m in models.items()},
        "optimizers": {n: o.state_dict() for n, o in (optimizers or {}).items()},
        "schedulers": {n: s.state_dict() for n, s in (schedulers or {}).items()},
        "scaler": scaler.state_dict() if scaler is not None else None,
        "rng": rng_state(),
        "sampler": sampler_state or {},
        "extra": extra or {},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(blob, tmp)
    digest = sha256_file(tmp)
    os.replace(tmp, path)
    Path(str(path) + ".sha256").write_text(f"{digest}  {path.name}\n")
    meta_json = dict(blob["meta"], sha256=digest, file=path.name)
    path.with_suffix(".json").write_text(json.dumps(meta_json, indent=2) + "\n")
    return path


def _unwrap(m: nn.Module) -> nn.Module:
    m = getattr(m, "_orig_mod", m)  # torch.compile
    return getattr(m, "module", m)  # DDP


def read_meta(path: str | Path) -> dict[str, Any]:
    """Metadata without loading tensors (uses the JSON sidecar)."""
    side = Path(path).with_suffix(".json")
    if not side.exists():
        raise CheckpointError(f"{path}: no metadata sidecar {side.name}; not an aivoice reproduction checkpoint")
    return json.loads(side.read_text())


def verify_checksum(path: str | Path) -> str:
    path = Path(path)
    side = Path(str(path) + ".sha256")
    if not side.exists():
        raise CheckpointError(f"{path}: missing checksum file {side.name}")
    want = side.read_text().split()[0]
    got = sha256_file(path)
    if got != want:
        raise CheckpointError(f"{path}: checksum mismatch (file {got[:12]}..., expected {want[:12]}...)")
    return got


def load_checkpoint(path: str | Path, *, engine: str | None = None, config: dict[str, Any] | None = None,
                    allow_config_change: bool = False, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Verify checksum, format, engine and config fingerprint, then load (weights_only)."""
    path = Path(path)
    verify_checksum(path)
    blob = torch.load(path, map_location=map_location, weights_only=True)
    if blob.get("format") != FORMAT:
        raise CheckpointError(f"{path}: unknown format {blob.get('format')!r}")
    meta = blob["meta"]
    if engine is not None and meta["engine"] != engine:
        raise CheckpointError(f"{path}: checkpoint is for engine {meta['engine']!r}, not {engine!r}")
    if config is not None and not allow_config_change:
        fp = config_fingerprint(config)
        if fp != meta["config_fingerprint"]:
            diff = sorted(k for k in set(config) | set(meta["config"]) if config.get(k) != meta["config"].get(k))
            raise CheckpointError(f"{path}: config fingerprint {meta['config_fingerprint']} != {fp}; differing keys: "
                                  f"{diff[:12]}")
    return blob


def load_model_state(model: nn.Module, state: dict[str, torch.Tensor], *, name: str = "model") -> None:
    """Strict load with a readable shape report instead of torch's first-error message."""
    own = _unwrap(model).state_dict()
    missing = sorted(set(own) - set(state))
    unexpected = sorted(set(state) - set(own))
    shapes = [f"{k}: checkpoint {tuple(state[k].shape)} vs model {tuple(own[k].shape)}"
              for k in sorted(set(own) & set(state)) if own[k].shape != state[k].shape]
    if missing or unexpected or shapes:
        lines = [f"{name}: checkpoint does not match the architecture"]
        if missing:
            lines.append(f"  missing ({len(missing)}): {missing[:8]}")
        if unexpected:
            lines.append(f"  unexpected ({len(unexpected)}): {unexpected[:8]}")
        lines += ["  " + s for s in shapes[:8]]
        raise CheckpointError("\n".join(lines))
    _unwrap(model).load_state_dict(state, strict=True)


def is_usable_for_inference(meta: dict[str, Any]) -> tuple[bool, str]:
    cfg = meta.get("config") or {}
    if cfg.get("test_standin") or cfg.get("test_tiny"):
        return False, "TEST STAND-IN: trained against stand-in teachers / tiny test model; a pipeline test, not a model"
    ws = meta.get("weights_status")
    if ws in (WeightsStatus.RANDOM_INIT.value, WeightsStatus.NONE.value):
        return False, weights_label(ws)
    if int(meta.get("step", 0)) <= 0:
        return False, "checkpoint was saved before any training step"
    return True, weights_label(ws)


__all__ = ["CheckpointError", "CheckpointMeta", "config_fingerprint", "environment", "git_state",
           "is_usable_for_inference", "load_checkpoint", "load_model_state", "read_meta", "rng_state",
           "save_checkpoint", "set_rng_state", "sha256_file", "verify_checksum"]
