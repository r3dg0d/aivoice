"""TVTSyn discriminators (training only). ASSUMPTION T-L2: HiFi-GAN MPD + MSD on the waveform (the paper's
adversarial term is used with weight 1; VOSSA's own synthesizer uses a multi-scale *mel* discriminator
because it predicts mels, not waveforms)."""

from ...common.losses import HiFiGANDiscriminator

__all__ = ["HiFiGANDiscriminator"]
