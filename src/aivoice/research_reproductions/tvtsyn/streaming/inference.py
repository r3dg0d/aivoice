"""TVTSyn streaming inference (exact per-layer state; output delayed by the encoder's total lookahead)."""

from __future__ import annotations

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
from ...common.streaming_nn import state_nbytes
from ...zerovc.streaming.buffers import to_numpy, to_tensor
from ..config import HOP, SAMPLE_RATE, TVTSynConfig, from_dict
from ..model.tvtsyn import TVTSyn


class Session:
    def __init__(self, e: TVTSynStreaming, g: torch.Tensor) -> None:
        self.e, self.g = e, g
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


class TVTSynStreaming:
    sample_rate = SAMPLE_RATE

    def __init__(self, net: TVTSyn, device: str = "cpu", meta: dict[str, Any] | None = None) -> None:
        self.net = net.to(device).eval()
        self.device = torch.device(device)
        self.meta = meta or {}
        self.future_frames = net.lookahead_frames
        self.lookahead_samples = net.lookahead_frames * HOP + HOP - 1

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "cpu", allow_untrained: bool = False) -> TVTSynStreaming:
        blob = load_checkpoint(path, map_location=device)
        meta = blob["meta"]
        if meta["engine"] not in ("tvtsyn-repro", "tvtsyn-aivoice"):
            raise CheckpointError(f"{path} is a {meta['engine']} checkpoint")
        cfg = from_dict(meta["config"])
        if cfg.stage != "synth":
            raise CheckpointError(f"{path} is a `{cfg.stage}` checkpoint; conversion needs a `synth` checkpoint")
        ok, why = is_usable_for_inference(meta)
        if (not ok or cfg.test_standin) and not allow_untrained:
            raise CheckpointError(f"{path}: {why if not ok else 'trained against TEST STAND-IN teachers'}")
        net = TVTSyn(cfg)
        for k in ("content", "timbre", "decoder"):
            load_model_state(getattr(net, k), blob["models"][k], name=k)
        return cls(net, device, meta)

    @classmethod
    def untrained(cls, cfg: TVTSynConfig, device: str = "cpu", seed: int = 0,
                  codebook_init_audio: np.ndarray | None = None) -> TVTSynStreaming:
        """RANDOM-INIT network for tests. A random VQ maps every frame to one code, which would make streaming
        tests vacuous; ``codebook_init_audio`` seeds the codebook with encoder outputs of that audio (data init)."""
        torch.manual_seed(seed)
        net = TVTSyn(cfg).eval()
        if codebook_init_audio is not None:
            with torch.no_grad():
                h = net.content.tf(net.content.cnn(torch.from_numpy(np.asarray(codebook_init_audio, np.float32))[None, None]))
                ze = net.content.vq.inp(h)[0].T                      # [frames, cdim]
                n = net.content.vq.codebook.weight.shape[0]
                net.content.vq.codebook.weight.copy_(ze[torch.randint(0, ze.shape[0], (n,))] + 1e-3 * torch.randn(n, ze.shape[1]))
        return cls(net, device, {"weights_status": "RANDOM_INIT"})

    def target_from_embeddings(self, xvector: np.ndarray, ecapa: np.ndarray) -> torch.Tensor:
        g = np.concatenate([np.asarray(xvector, np.float32).reshape(-1), np.asarray(ecapa, np.float32).reshape(-1)])
        if g.shape[0] != self.net.cfg.timbre.g_dim:
            raise ValueError(f"expected {self.net.cfg.timbre.g_dim}-d [x-vector; ECAPA], got {g.shape[0]}")
        return torch.from_numpy(g)[None].to(self.device)

    def new_session(self, g: torch.Tensor) -> Session:
        return Session(self, g)

    @torch.inference_mode()
    def offline(self, x: np.ndarray, g: torch.Tensor) -> np.ndarray:
        y, _ = self.net(to_tensor(x, self.device), g)
        return to_numpy(y)
