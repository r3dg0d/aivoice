"""RT-VC has no discriminator in the official code or the DDSP recipe it builds on; training uses the
multi-scale spectral loss only. ``use_adversarial`` is not offered (it would be an aivoice variant)."""

DISCRIMINATOR = None
