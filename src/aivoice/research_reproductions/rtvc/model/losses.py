"""RT-VC losses: DDSP multi-scale spectral loss (vocoder); BCE pitch bins / periodicity and L1 loudness
(pitch tracker); L1 to articulatory pseudo-labels (EMA inverter)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ...common.losses import MultiScaleSpectralLoss

__all__ = ["MultiScaleSpectralLoss", "ema_loss", "pitch_losses"]


def pitch_losses(out: dict[str, torch.Tensor], target_bins: torch.Tensor, periodicity: torch.Tensor,
                 loudness: torch.Tensor) -> dict[str, torch.Tensor]:
    n = min(out["logits"].shape[1], target_bins.shape[1], periodicity.shape[-1], loudness.shape[-1])
    p = out["periodicity"][..., :n]
    # BCE on the sigmoid probabilities (as specified), in fp32 with autocast off: torch refuses
    # binary_cross_entropy under autocast, which stopped bf16 training of the pitch stage.
    with torch.autocast(device_type=p.device.type, enabled=False):
        per_bce = F.binary_cross_entropy(p.float().clamp(1e-6, 1 - 1e-6), periodicity[..., :n].float().clamp(0, 1))
    return {"pitch_bce": F.binary_cross_entropy_with_logits(out["logits"][:, :n], target_bins[:, :n]),
            "periodicity_bce": per_bce,
            "loudness_l1": F.l1_loss(out["loudness"][..., :n], loudness[..., :n])}


def ema_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    n = min(pred.shape[-1], target.shape[-1])
    return F.l1_loss(pred[..., :n], target[..., :n])
