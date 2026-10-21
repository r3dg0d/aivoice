"""StreamVoiceAnon+ fine-tuning extension config.

The base is the official StreamVoiceAnon (Plachtaa/StreamVoiceAnon @ 2017051, Apache-2.0) dual-AR codec LM.
StreamVoiceAnon+ (arXiv 2603.06079) weights were unreleased ("coming soon") at that commit; this package
re-creates the *training extension* described for it and fine-tunes from the public base checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

CODEC_SR = 44100
CODEC_HOP = 2048          # VERIFIED: firefly codec frame = 2048 samples @ 44.1 kHz (~21.5 Hz)


@dataclass
class PlusConfig:
    base_config: str = "configs/config_firefly_arvcasr_8192_delay0_8.yaml"   # VERIFIED (official repo)
    base_checkpoint: str = "pretrained_checkpoints/dual_ar_delay_0_8.pth"    # VERIFIED (official HF release)
    lambda_emo: float = 1.0          # ASSUMPTION P-L1: lambda not available to aivoice
    slow_ar_weight: float = 1.0      # VERIFIED official semantic_loss_weight
    fast_ar_weight: float = 1.0      # VERIFIED official codebook_loss_weight
    emo_dim: int = 768               # emotion2vec base/plus-base hidden size
    emo_frame_rate: float = 50.0     # emotion2vec frame rate
    emo_head_hidden: int = 512       # ASSUMPTION P-H1: 2-layer MLP distillation head
    emo_layer: str = "last"          # Slow-AR hidden states used for distillation (final pre-norm output)
    prompt_frames: int = 64          # ASSUMPTION: neutral prompt crop (~3 s)
    source_frames: int = 128         # ASSUMPTION: emotional source crop (~6 s)
    lr: float = 1e-5                 # ASSUMPTION: fine-tuning lr (official from-scratch lr 1e-4)
    betas: tuple[float, float] = (0.9, 0.95)
    freeze_embeddings: bool = False
    # tiny official-architecture test settings (never a real model)
    test_tiny: bool = False
    test_dims: dict[str, Any] = field(default_factory=lambda: {"dim": 64, "n_layer": 2, "n_head": 4, "head_dim": 16,
                                                                "n_fast_layer": 1, "num_codebooks": 2, "vocab": 32,
                                                                "codebook": 16, "emo_dim": 16})


def paper() -> PlusConfig:
    return PlusConfig()


def tiny() -> PlusConfig:
    c = PlusConfig(test_tiny=True, prompt_frames=6, source_frames=10, emo_dim=16, emo_head_hidden=32, lr=1e-3)
    return c


PRESETS = {"paper": paper, "tiny": tiny}


def from_dict(d: dict[str, Any]) -> PlusConfig:
    return PlusConfig(**{k: tuple(v) if k == "betas" else v for k, v in d.items()})
