"""TVTSyn losses (weights from the paper brief): 20 * mel L1 + 1 * adversarial (LSGAN) + 2 * feature matching
+ 20 * (pitch MSE + energy MSE) of the variance predictors; content stage: CE + VQ commitment + codebook."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ...common.losses import MelL1, feature_matching_loss, lsgan_d_loss, lsgan_g_loss
from ..config import LossConfig

__all__ = ["MelL1", "discriminator_loss", "generator_loss"]


def generator_loss(cfg: LossConfig, mel: MelL1, y_hat, y, d_fake, f_real, f_fake, var_pred, pitch, energy):
    n = min(var_pred["pitch"].shape[-1], pitch.shape[-1])
    l_var = F.mse_loss(var_pred["pitch"][..., :n], pitch[..., :n]) + F.mse_loss(var_pred["energy"][..., :n],
                                                                                energy[..., :n])
    l_mel = mel(y_hat[:, 0], y[:, 0])
    l_adv = lsgan_g_loss(d_fake)
    l_fm = feature_matching_loss(f_real, f_fake)
    total = cfg.mel * l_mel + cfg.adv * l_adv + cfg.fm * l_fm + cfg.f0_energy * l_var
    return {"g_total": total, "mel": l_mel, "adv": l_adv, "fm": l_fm, "f0_energy": l_var}


def discriminator_loss(real, fake) -> torch.Tensor:
    return lsgan_d_loss(real, fake)
