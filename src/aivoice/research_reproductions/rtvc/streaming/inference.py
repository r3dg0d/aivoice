"""RT-VC streaming inference with explicit per-layer state (5 ms frames, zero future frames)."""

from __future__ import annotations

from dataclasses import dataclass
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
from ..config import FRAME, SAMPLE_RATE, RTVCConfig, from_dict
from ..info import INFO
from ..model.rtvc import RTVC
from .buffers import to_numpy, to_tensor
from .state import state_nbytes


@dataclass
class Target:
    g: torch.Tensor          # speaker embedding [1, E]
    f0_ratio: float = 1.0    # target_median / source_median


class Session:
    def __init__(self, e: RTVCStreaming, target: Target) -> None:
        self.e, self.t = e, target
        self.reset()

    def reset(self) -> None:
        self.state = None

    @torch.inference_mode()
    def push(self, block: np.ndarray) -> np.ndarray:
        if len(block) == 0:
            return np.zeros(0, np.float32)
        y, self.state = self.e.net.step(to_tensor(block, self.e.device), self.t.g, self.t.f0_ratio, self.state)
        return to_numpy(y)

    def state_bytes(self) -> int:
        return state_nbytes({k: v for k, v in (self.state or {}).items()})


class RTVCStreaming:
    sample_rate = SAMPLE_RATE
    lookahead_samples = FRAME - 1  # 5 ms framing; no future frames
    future_frames = 0

    def __init__(self, net: RTVC, device: str = "cpu", meta: dict[str, Any] | None = None, speaker=None) -> None:
        self.net = net.to(device).eval()
        self.device = torch.device(device)
        self.meta = meta or {}
        self.speaker = speaker.to(device).eval() if speaker is not None else None

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "cpu", allow_untrained: bool = False,
                        teacher_dirs: dict[str, str] | None = None) -> RTVCStreaming:
        blob = load_checkpoint(path, engine=INFO.key, map_location=device)
        meta = blob["meta"]
        cfg = from_dict(meta["config"])
        missing = [k for k in ("pitch", "ema", "vocoder", "speaker") if k not in blob["models"]]
        if missing:
            raise CheckpointError(f"{path}: not a complete RT-VC model (missing {missing}); use `reproduce export` "
                                  "to assemble pitch + ema + vocoder checkpoints")
        ok, why = is_usable_for_inference(meta)
        if (not ok or cfg.test_standin) and not allow_untrained:
            raise CheckpointError(f"{path}: {why if not ok else 'trained against TEST STAND-IN teachers'}")
        from ..model.timbre import SpeakerEncoder

        net = RTVC(cfg)
        for k in ("pitch", "ema", "vocoder"):
            load_model_state(getattr(net, k), blob["models"][k], name=k)
        spk = SpeakerEncoder(cfg.speaker, (teacher_dirs or {}).get("wavlm-large"), cfg.test_standin)
        load_model_state(spk, blob["models"]["speaker"], name="speaker")
        return cls(net, device, meta, spk)

    @classmethod
    def untrained(cls, cfg: RTVCConfig, device: str = "cpu", seed: int = 0) -> RTVCStreaming:
        torch.manual_seed(seed)
        return cls(RTVC(cfg), device, {"weights_status": "RANDOM_INIT"})

    @torch.inference_mode()
    def median_f0(self, wav16k: np.ndarray) -> float:
        a = self.net.analyse(to_tensor(wav16k, self.device))
        v = a["f0"][a["periodicity"] > 0.5]
        return float(v.median()) if v.numel() else float("nan")

    @torch.inference_mode()
    def enroll(self, target_16k: np.ndarray, source_16k: np.ndarray | None = None,
               source_median: float | None = None) -> Target:
        if self.speaker is None:
            raise RuntimeError("no speaker encoder loaded")
        x = to_tensor(target_16k, self.device)
        a = self.net.analyse(x)
        g = self.speaker(x, a["periodicity"])
        tgt = self.median_f0(target_16k)
        src = source_median if source_median is not None else (self.median_f0(source_16k) if source_16k is not None else tgt)
        return Target(g, float(tgt / src) if np.isfinite(tgt) and np.isfinite(src) and src > 0 else 1.0)

    def new_session(self, target: Target) -> Session:
        return Session(self, target)

    @torch.inference_mode()
    def offline(self, x: np.ndarray, target: Target) -> np.ndarray:
        return to_numpy(self.net(to_tensor(x, self.device), target.g, target.f0_ratio))
