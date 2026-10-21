"""Zero-VC reproduction configuration.

``paper`` follows what the paper states (16 kHz, 20 ms frames, WavLM-large layer 7, HiFi-GAN
training framework, LibriTTS >= 4 s); every value the paper does not state is marked ``# ASSUMPTION``
and listed in SPEC.md. ``tiny`` is the same architecture shrunk for the mandatory tiny-overfit test.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

SAMPLE_RATE = 16000  # paper: data resampled to 16 kHz
HOP = 320            # paper: 20 ms frame shift / 20 ms streaming chunks


@dataclass
class ContentConfig:
    variant: str = "A"                 # A = distilled streaming w2v-BERT 2.0 student; B = HuBERT k-means CE student
    dim: int = 512                     # ASSUMPTION (student width)
    n_filters: int = 32                # ASSUMPTION
    strides: tuple[int, ...] = (2, 4, 5, 8)  # ASSUMPTION (any product of 320 gives 20 ms)
    depth: int = 8                     # ASSUMPTION
    heads: int = 8                     # ASSUMPTION
    context: int = 100                 # ASSUMPTION: 2 s attention window
    right_context: int = 0             # paper: zero lookahead
    # variant A: regression target
    # variant A teacher: facebook/w2v-bert-2.0 (paper: "e.g., a distilled streaming w2v-bert-2.0")
    teacher_layer: int = 16            # ASSUMPTION (layer not stated)
    teacher_dim: int = 1024
    # variant B: classification target
    # variant B teacher: facebook/hubert-base-ls960
    km_layer: int = 9
    km_codes: int = 200
    out_dim: int = 512                 # feature dim handed to the decoder (projection of the student output)


@dataclass
class TimbreConfig:
    ssl_model: str = "microsoft/wavlm-large"  # paper
    ssl_layer: int = 7                 # paper (7th layer hidden states)
    ssl_dim: int = 1024
    embed_dim: int = 256               # ASSUMPTION
    pool_hidden: int = 256             # ASSUMPTION
    reference_s: float = 3.0           # ASSUMPTION: timbre reference crop during training
    pooling: str = "attention"         # paper (Eq. 1); "mean" = ABLATION: static mean pooling


@dataclass
class DecoderConfig:
    ch: int = 512                      # HiFi-GAN V1 width (paper: "mainly based on HiFi-GAN")
    rates: tuple[int, ...] = (8, 5, 4, 2)  # ASSUMPTION: HiFi-GAN V1 is (8,8,2,2) for hop 256; 320 needs another split
    mrf_kernels: tuple[int, ...] = (3, 7, 11)
    mrf_dilations: tuple[int, ...] = (1, 3, 5)


@dataclass
class AnonymizerConfig:
    kind: str = "mcadams"              # ASSUMPTION: paper uses an unnamed off-the-shelf SA module
    #                                    "none" = ABLATION: train without anonymisation (paper's motivation test)
    alpha_range: tuple[float, float] = (0.7, 0.9)  # McAdams coefficient range (VoicePrivacy B2 uses 0.8)
    lpc_order: int = 20


@dataclass
class LossConfig:                       # HiFi-GAN weights (paper: "same adversarial framework as HiFi-GAN")
    mel: float = 45.0
    fm: float = 2.0
    adv: float = 1.0


@dataclass
class OptimConfig:                      # HiFi-GAN optimiser (ASSUMPTION: paper defers to HiFi-GAN)
    lr: float = 2e-4
    betas: tuple[float, float] = (0.8, 0.99)
    weight_decay: float = 0.01
    lr_decay: float = 0.999            # per 1000 steps here (HiFi-GAN: per epoch)


@dataclass
class ZeroVCConfig:
    stage: str = "vc"                  # content | vc
    segment_s: float = 2.0             # ASSUMPTION (training crop)
    min_utt_s: float = 4.0             # paper: utterances shorter than 4 s discarded
    content: ContentConfig = field(default_factory=ContentConfig)
    timbre: TimbreConfig = field(default_factory=TimbreConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    anonymizer: AnonymizerConfig = field(default_factory=AnonymizerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    disc_ch_mult: float = 1.0
    test_standin: bool = False         # True only in tests/tiny runs: random frozen stand-ins for SSL teachers


def paper(variant: str = "A", stage: str = "vc") -> ZeroVCConfig:
    c = ZeroVCConfig(stage=stage)
    c.content.variant = variant
    return c


def tiny(variant: str = "A", stage: str = "vc") -> ZeroVCConfig:
    c = paper(variant, stage)
    c.segment_s = 0.64
    c.content = replace(c.content, dim=64, n_filters=8, depth=2, heads=4, context=25, teacher_dim=64, out_dim=64,
                        km_codes=16)
    c.timbre = replace(c.timbre, ssl_dim=64, embed_dim=32, pool_hidden=32, reference_s=0.64)
    c.decoder = replace(c.decoder, ch=128)
    c.disc_ch_mult = 1 / 16
    c.test_standin = True
    return c


PRESETS = {"paper": paper, "tiny": tiny}


def from_dict(d: dict[str, Any]) -> ZeroVCConfig:
    c = ZeroVCConfig(**{k: v for k, v in d.items() if k not in ("content", "timbre", "decoder", "anonymizer", "loss",
                                                               "optim")})
    for k, cls in (("content", ContentConfig), ("timbre", TimbreConfig), ("decoder", DecoderConfig),
                   ("anonymizer", AnonymizerConfig), ("loss", LossConfig), ("optim", OptimConfig)):
        if k in d:
            vals = {kk: tuple(vv) if isinstance(vv, list) else vv for kk, vv in d[k].items()}
            setattr(c, k, cls(**vals))
    return c
