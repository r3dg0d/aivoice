"""Zero-VC discriminators (training only): HiFi-GAN multi-period + multi-scale (paper: "the same adversarial
framework as HiFi-GAN"; MPD and MSD are discarded after training)."""

from ...common.losses import HiFiGANDiscriminator

__all__ = ["HiFiGANDiscriminator"]
