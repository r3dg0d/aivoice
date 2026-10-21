"""RT-VC checkpoints: one per stage (pitch / ema / vocoder+speaker), assembled for inference by ``export``."""

from __future__ import annotations

from pathlib import Path

from ...common.checkpoint import CheckpointMeta, load_checkpoint, save_checkpoint
from ...common.provenance import WeightsStatus
from ..config import from_dict
from ..info import INFO
from ..model.decoder import DDSPVocoder
from ..model.encoder import EMAInverter, PitchTracker
from ..model.timbre import SpeakerEncoder


def export(pitch_ckpt: str, ema_ckpt: str, vocoder_ckpt: str, out: str | Path, teacher_dirs=None) -> Path:
    """Combine the three stage checkpoints into one self-describing inference checkpoint."""
    blobs = {k: load_checkpoint(p, engine=INFO.key) for k, p in (("pitch", pitch_ckpt), ("ema", ema_ckpt),
                                                                  ("vocoder", vocoder_ckpt))}
    cfg = from_dict(blobs["vocoder"]["meta"]["config"])
    for k in ("pitch", "ema"):
        c = from_dict(blobs[k]["meta"]["config"])
        setattr(cfg, k, getattr(c, k))  # each stage checkpoint is authoritative for its own module
        if c.frontend != cfg.frontend:
            raise ValueError(f"{k} checkpoint used a different front-end config")
    cfg.stage = "inference"
    models = {"pitch": PitchTracker(cfg.pitch, cfg.frontend.n_mels), "ema": EMAInverter(cfg.ema, cfg.frontend.n_mfcc),
              "vocoder": DDSPVocoder(cfg.vocoder, cfg.speaker.spkr_emb_dim, cfg.ema.channels),
              "speaker": SpeakerEncoder(cfg.speaker, (teacher_dirs or {}).get("wavlm-large"), cfg.test_standin)}
    models["pitch"].load_state_dict(blobs["pitch"]["models"]["pitch"])
    models["ema"].load_state_dict(blobs["ema"]["models"]["ema"])
    models["vocoder"].load_state_dict(blobs["vocoder"]["models"]["vocoder"])
    models["speaker"].load_state_dict(blobs["vocoder"]["models"]["speaker"])
    from dataclasses import asdict

    steps = {k: int(b["meta"]["step"]) for k, b in blobs.items()}
    meta = CheckpointMeta(engine=INFO.key, model_name=INFO.model_name,
                          weights_status=WeightsStatus.RETRAINED_PAPER_REPRODUCTION.value if min(steps.values()) > 0
                          else WeightsStatus.RANDOM_INIT.value,
                          source_status=INFO.source_status.value, arch_confidence=INFO.arch_confidence.value,
                          config=asdict(cfg), step=min(steps.values()),
                          datasets=sorted({d for b in blobs.values() for d in b["meta"]["datasets"]}),
                          component_licenses={k: str(b["meta"]["component_licenses"]) for k, b in blobs.items()},
                          notes=f"assembled from stage checkpoints {steps}; NOT official weights")
    return save_checkpoint(out, meta, models)
