"""Frozen pretrained teachers / feature extractors (training-time only unless a recipe says otherwise).

Real teachers are loaded from a *local* Hugging Face snapshot directory (aivoice never downloads
them implicitly); set the directory with ``--teacher NAME=DIR`` or ``AIVOICE_TEACHER_<NAME>``.

``StandInSSL`` is a frozen, randomly initialised CNN with the same frame rate and width as a real
teacher. It exists only so the pipelines can be exercised end-to-end (tiny-overfit tests, shape and
compute checks) where the real teachers are unavailable. Anything trained against it is tagged
``test_standin`` in its checkpoint and is refused by every inference path.
"""

from __future__ import annotations

import zlib
from pathlib import Path

import torch
from torch import nn

from .blocks import CausalSEANetEncoder
from .teacher_paths import (
    KNOWN,
    TeacherUnavailable,
    teacher_dir,
)


class HFSSLTeacher(nn.Module):
    """Hidden states at ``layer`` (hidden_states[layer]: 0 = input to the first transformer layer)."""

    def __init__(self, name: str, layer: int, directory: str | Path | None = None) -> None:
        super().__init__()
        import transformers

        hf_id, cls, self.license = KNOWN[name]
        d = teacher_dir(name, directory)
        self.name, self.layer, self.source = name, layer, hf_id
        self.model = getattr(transformers, cls).from_pretrained(str(d)).eval()
        self.model.requires_grad_(False)
        self.is_w2vbert = cls == "Wav2Vec2BertModel"
        if self.is_w2vbert:
            self.fe = transformers.SeamlessM4TFeatureExtractor.from_pretrained(str(d))
        self.dim = self.model.config.hidden_size
        from .features import file_sha256

        w = next((q for q in (d / "model.safetensors", d / "pytorch_model.bin") if q.exists()), None)
        self.revision = file_sha256(w) if w else None

    @torch.no_grad()
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """wav [B, T] at 16 kHz -> [B, frames(50 Hz), dim]."""
        if self.is_w2vbert:
            feats = self.fe([w.cpu().numpy() for w in wav], sampling_rate=16000, return_tensors="pt", padding=True)
            out = self.model(input_features=feats["input_features"].to(wav.device), output_hidden_states=True)
        else:
            out = self.model(wav, output_hidden_states=True)
        return out.hidden_states[self.layer]


class StandInSSL(nn.Module):
    """TEST STAND-IN for an SSL teacher: frozen random causal CNN, 50 Hz, ``dim`` wide. Not a model of speech."""

    standin = True

    def __init__(self, dim: int, seed: int = 0) -> None:
        super().__init__()
        g = torch.random.fork_rng()
        with g:
            torch.manual_seed(seed)
            self.net = CausalSEANetEncoder(dim, 8, (2, 4, 5, 8))
        self.requires_grad_(False)
        self.eval()
        self.dim, self.name, self.layer, self.source, self.revision, self.license = dim, "standin", 0, None, None, "n/a"

    @torch.no_grad()
    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        return self.net(wav[:, None]).transpose(1, 2)

    def train(self, mode: bool = True) -> StandInSSL:  # always frozen
        return super().train(False)


def load_teacher(name: str, layer: int, dim: int, standin: bool, directory: str | Path | None = None) -> nn.Module:
    if standin:
        return StandInSSL(dim, seed=zlib.crc32(name.encode()) & 0xFFFF)
    return HFSSLTeacher(name, layer, directory)


__all__ = ["KNOWN", "HFSSLTeacher", "StandInSSL", "TeacherUnavailable", "load_teacher", "teacher_dir"]
