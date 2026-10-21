"""Zero-VC losses: HiFi-GAN generator loss = adv + 2 * feature matching + 45 * mel L1; LSGAN discriminator."""

from __future__ import annotations

import torch

from ...common.losses import MelL1, feature_matching_loss, lsgan_d_loss, lsgan_g_loss
from ..config import LossConfig


def generator_loss(cfg: LossConfig, mel: MelL1, y_hat: torch.Tensor, y: torch.Tensor, d_fake, d_real_feats,
                   d_fake_feats) -> dict[str, torch.Tensor]:
    l_mel = mel(y_hat[:, 0], y[:, 0])
    l_adv = lsgan_g_loss(d_fake)
    l_fm = feature_matching_loss(d_real_feats, d_fake_feats)
    total = cfg.adv * l_adv + cfg.fm * l_fm + cfg.mel * l_mel
    return {"g_total": total, "mel": l_mel, "adv": l_adv, "fm": l_fm}


def discriminator_loss(d_real, d_fake) -> torch.Tensor:
    return lsgan_d_loss(d_real, d_fake)
