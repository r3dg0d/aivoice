"""Zero-VC checkpoints use the shared self-describing format (common/checkpoint.py) with engine
``zerovc-repro``; a ``vc``-stage checkpoint contains content + timbre + decoder (+ discriminator for resume)."""

from ...common.checkpoint import load_checkpoint, read_meta, save_checkpoint

__all__ = ["load_checkpoint", "read_meta", "save_checkpoint"]
