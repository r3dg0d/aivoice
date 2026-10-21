"""Zero-VC streaming inference: true per-layer state, one 20 ms frame in -> 320 samples out, no lookahead.

Not offline inference on a sliding window: every layer advances its own state (streaming_nn), and the
harness verifies the streamed output equals the offline forward exactly (up to float rounding).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ...common.checkpoint import (
    CheckpointError,
    is_usable_for_inference,
    load_checkpoint,
    load_model_state,
)
from ...common.teachers import load_teacher
from ..config import HOP, SAMPLE_RATE, ZeroVCConfig, from_dict
from ..info import INFO
from ..model.zerovc import ZeroVC
from .buffers import to_numpy, to_tensor
from .state import state_nbytes


class Session:
    def __init__(self, engine: ZeroVCStreaming, g: torch.Tensor) -> None:
        self.e, self.g = engine, g
        self.reset()

    def reset(self) -> None:
        self.state = None

    @torch.inference_mode()
    def push(self, block: np.ndarray) -> np.ndarray:
        if len(block) == 0:
            return np.zeros(0, np.float32)
        y, self.state = self.e.net.step(to_tensor(block, self.e.device), self.g, self.state)
        return to_numpy(y)

    def state_bytes(self) -> int:
        return state_nbytes(self.state)


class ZeroVCStreaming:
    sample_rate = SAMPLE_RATE
    hop = HOP
    # each output sample of frame t depends on the whole input frame t (20 ms framing), never on frame t+1
    lookahead_samples = HOP - 1
    future_frames = 0

    def __init__(self, net: ZeroVC, device: str = "cpu", meta: dict[str, Any] | None = None,
                 teacher_dirs: dict[str, str] | None = None) -> None:
        self.net = net.to(device).eval()
        self.device = torch.device(device)
        self.meta = meta or {}
        self.teacher_dirs = teacher_dirs or {}
        self._wavlm = None

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "cpu", allow_untrained: bool = False,
                        teacher_dirs: dict[str, str] | None = None) -> ZeroVCStreaming:
        blob = load_checkpoint(path, engine=INFO.key, map_location=device)
        meta = blob["meta"]
        cfg = from_dict(meta["config"])
        if cfg.stage != "vc":
            raise CheckpointError(f"{path} is a `{cfg.stage}` stage checkpoint; conversion needs a `vc` checkpoint")
        ok, why = is_usable_for_inference(meta)
        if (not ok or cfg.test_standin) and not allow_untrained:
            raise CheckpointError(f"{path}: {why if not ok else 'trained against TEST STAND-IN teachers'}; refusing "
                                  "to use it for conversion")
        net = ZeroVC(cfg)
        for k in ("content", "timbre", "decoder"):
            load_model_state(getattr(net, k), blob["models"][k], name=k)
        return cls(net, device, meta, teacher_dirs)

    @classmethod
    def untrained(cls, cfg: ZeroVCConfig, device: str = "cpu", seed: int = 0) -> ZeroVCStreaming:
        """RANDOM-INIT network for tests and compute measurements only (never for conversion)."""
        torch.manual_seed(seed)
        return cls(ZeroVC(cfg), device, {"weights_status": "RANDOM_INIT"})

    # -- timbre -----------------------------------------------------------------------------------
    @torch.inference_mode()
    def timbre_from_features(self, ssl_feats: np.ndarray | torch.Tensor) -> torch.Tensor:
        f = torch.as_tensor(np.asarray(ssl_feats, np.float32))[None].to(self.device)
        return self.net.timbre(f)

    @torch.inference_mode()
    def enroll(self, reference_16k: np.ndarray, cache_dir: str | Path | None = None) -> torch.Tensor:
        """Timbre vector of a target voice: WavLM-large L7 -> attention pooling, cached per (voice, checkpoint)."""
        key = hashlib.sha256(np.asarray(reference_16k, np.float32).tobytes()).hexdigest()[:16] + "-" + \
            str(self.meta.get("config_fingerprint", "untrained")) + f"-s{self.meta.get('step', 0)}"
        if cache_dir is not None:
            p = Path(cache_dir) / f"zerovc-timbre-{key}.npy"
            if p.exists():
                return torch.from_numpy(np.load(p)).to(self.device)
        if self._wavlm is None:
            c = self.net.cfg.timbre
            self._wavlm = load_teacher("wavlm-large", c.ssl_layer, c.ssl_dim, self.net.cfg.test_standin,
                                       self.teacher_dirs.get("wavlm-large")).to(self.device)
        feats = self._wavlm(torch.from_numpy(np.asarray(reference_16k, np.float32))[None].to(self.device))
        g = self.net.timbre(feats.float())
        if cache_dir is not None:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            np.save(p, g.cpu().numpy())
        return g

    # -- harness protocol -------------------------------------------------------------------------
    def new_session(self, target: torch.Tensor) -> Session:
        return Session(self, target)

    @torch.inference_mode()
    def offline(self, x: np.ndarray, target: torch.Tensor) -> np.ndarray:
        return to_numpy(self.net(to_tensor(x, self.device), target))
