"""Zero-VC streaming decoder: "mainly based on HiFi-GAN, with critical modifications for zero-lookahead
streaming". aivoice realises those modifications as causal convolutions, causally trimmed transposed
convolutions and frame-synchronous state (one 20 ms frame in, 320 samples out). The exact modifications
in the paper are not public; see SPEC.md (assumption Z-D1..Z-D3)."""

from __future__ import annotations

from ...common.blocks import CausalHiFiGAN
from ..config import DecoderConfig


def build_decoder(cfg: DecoderConfig, cin: int, cond_dim: int) -> CausalHiFiGAN:
    return CausalHiFiGAN(cin, cond_dim, cfg.ch, cfg.rates, cfg.mrf_kernels, cfg.mrf_dilations)
