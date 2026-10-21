"""TVTSyn configuration. VERIFIED = read from the official VOSSA repository (Morris88826/VOSSA @ 5dd1a5c,
whose README calls its content encoder "the TVTSyn backbone": configs/model/pasrwithquant_la4.yaml and
components/tv_timbre.py). PAPER = stated in the task brief from the TVTSyn paper. ASSUMPTION = neither."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

SAMPLE_RATE = 16000  # VERIFIED
HOP = 320            # VERIFIED (frame_rate 50)


@dataclass
class ContentConfig:
    n_filters: int = 64                         # VERIFIED
    strides: tuple[int, ...] = (2, 4, 5, 8)     # VERIFIED ratios [8,5,4,2] (SEANet encoders apply them reversed)
    dim: int = 512                              # VERIFIED
    depth: int = 8                              # VERIFIED / PAPER (8 MHSA blocks)
    heads: int = 8                              # VERIFIED
    context: int = 100                          # VERIFIED
    right_context: tuple[int, ...] = (4,) * 8   # VERIFIED: right_context 4 in every layer (see SPEC T-S1)
    codebook_size: int = 4096                   # VERIFIED / PAPER
    codebook_dim: int = 8                       # VERIFIED / PAPER
    commitment: float = 0.15                    # VERIFIED
    codebook_loss_weight: float = 1.0           # VERIFIED
    l2_normalize: bool = True                   # VERIFIED
    num_codes: int = 200                        # VERIFIED / PAPER (HuBERT-base L9 k-means 200)
    km_layer: int = 9                           # PAPER


@dataclass
class TimbreConfig:
    g_dim: int = 704          # VERIFIED (x-vector 512 + ECAPA 192)
    d_kv: int = 128           # VERIFIED
    d_out: int = 192          # VERIFIED
    num_tokens: int = 48      # VERIFIED
    heads: int = 4            # VERIFIED
    gate_hidden: int = 256    # VERIFIED
    gtm_hidden: int = 256     # VERIFIED
    token_dropout: float = 0.1  # VERIFIED


@dataclass
class DecoderConfig:
    d_model: int = 512        # VERIFIED (frame synthesizer)
    heads: int = 8
    context: int = 100
    b1: int = 3               # VERIFIED
    b2: int = 3
    b3: int = 2
    var_kernel: int = 7       # VERIFIED
    var_dropout: float = 0.1  # VERIFIED
    seanet_filters: int = 64  # ASSUMPTION (waveform decoder width)
    ratios: tuple[int, ...] = (8, 5, 4, 2)  # PAPER brief: upsampling [2,4,5,8] (order of application here: 8,5,4,2)


@dataclass
class LossConfig:  # PAPER brief
    mel: float = 20.0
    adv: float = 1.0
    fm: float = 2.0
    f0_energy: float = 20.0


@dataclass
class TVTSynConfig:
    stage: str = "synth"      # content | synth
    segment_s: float = 2.0    # ASSUMPTION
    lr: float = 1e-4          # VERIFIED-ish (VOSSA synthesizer lr 1e-4, betas 0.8/0.99); PASR stage 5e-4
    content_lr: float = 5e-4  # VERIFIED (pasrwithquant_la4.yaml)
    betas: tuple[float, float] = (0.8, 0.99)
    lr_gamma: float = 0.999996  # VERIFIED (per step)
    content: ContentConfig = field(default_factory=ContentConfig)
    timbre: TimbreConfig = field(default_factory=TimbreConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    disc_ch_mult: float = 1.0
    variant: str = "repro"    # repro | aivoice (first-layer-only lookahead: exact 80 ms)
    test_standin: bool = False


def paper(stage: str = "synth") -> TVTSynConfig:
    return TVTSynConfig(stage=stage)


def aivoice_variant(stage: str = "synth") -> TVTSynConfig:
    """tvtsyn-aivoice: identical except the 4-frame lookahead sits in the first transformer layer only, so
    exact streaming has 80 ms lookahead instead of 640 ms. NOT the paper configuration."""
    c = paper(stage)
    c.variant = "aivoice"
    c.content = replace(c.content, right_context=(4,) + (0,) * (c.content.depth - 1))
    return c


def tiny(stage: str = "synth") -> TVTSynConfig:
    c = paper(stage)
    c.segment_s = 0.64
    c.content = replace(c.content, n_filters=8, dim=64, depth=2, heads=4, context=25, right_context=(1, 0),
                        codebook_size=64, num_codes=16)
    c.timbre = replace(c.timbre, g_dim=32, d_kv=32, d_out=24, num_tokens=8, gate_hidden=32, gtm_hidden=32)
    c.decoder = replace(c.decoder, d_model=64, heads=4, context=25, b1=1, b2=1, b3=1, seanet_filters=8)
    c.disc_ch_mult = 1 / 16
    c.lr = c.content_lr = 1e-3
    c.test_standin = True
    return c


PRESETS = {"paper": paper, "aivoice": aivoice_variant, "tiny": tiny}
_SUB = {"content": ContentConfig, "timbre": TimbreConfig, "decoder": DecoderConfig, "loss": LossConfig}


def from_dict(d: dict[str, Any]) -> TVTSynConfig:
    c = TVTSynConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items() if k not in _SUB})
    for k, cls in _SUB.items():
        if k in d:
            setattr(c, k, cls(**{kk: tuple(vv) if isinstance(vv, list) else vv for kk, vv in d[k].items()}))
    return c
