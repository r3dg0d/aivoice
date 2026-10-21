"""StreamVoiceAnon+ fine-tuning recipe: official base -> + SEP tokens + emotion distillation head."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch

from ...common.checkpoint import CheckpointMeta
from ...common.provenance import WeightsStatus
from ...common.trainer import StepContext, TrainConfig, make_adamw
from ..config import PlusConfig
from ..info import INFO
from ..model.extension import StreamVoiceAnonPlus
from ..model.official import load_base, tiny_official
from .dataset import PairDataset


class PlusRecipe:
    engine = INFO.key
    overfit_loss_key = "loss"

    def __init__(self, cfg: PlusConfig, manifest, val_manifest=None, caches=None, repo_dir: str | None = None,
                 trust_pickle: bool = False, datasets: list[str] | None = None, fixed_crops: bool = False) -> None:
        self.cfg, self.manifest, self.val_manifest, self.caches = cfg, manifest, val_manifest, caches
        self.repo_dir, self.trust_pickle, self.datasets, self.fixed_crops = repo_dir, trust_pickle, datasets or [], fixed_crops

    def model_config(self) -> dict[str, Any]:
        return asdict(self.cfg)

    def build(self, tc: TrainConfig, device: torch.device) -> None:
        cfg = self.cfg
        if cfg.test_tiny:
            arvc = tiny_official(cfg.test_dims, self.repo_dir)
            self.base = {"checkpoint": None, "note": "TEST: tiny random official-architecture model"}
            d = cfg.test_dims
            syn = {"vocab": d["vocab"], "codebook": d["codebook"], "K": d["num_codebooks"], "ctx": 16, "style": 24,
                   "emo": cfg.emo_dim, "n": 4}
        else:
            arvc = load_base(cfg.base_config, cfg.base_checkpoint, self.repo_dir, self.trust_pickle)
            self.base = {"checkpoint": cfg.base_checkpoint, "license": "Apache-2.0 (Plachtaa/StreamVoiceAnon)"}
            syn = None
        self.net = StreamVoiceAnonPlus(arvc, cfg).to(device)
        if cfg.freeze_embeddings:
            self.net.arvc.embedding.requires_grad_(False)
        self.models = {"plus": self.net}
        self.optimizers = {"main": make_adamw([p for p in self.net.parameters() if p.requires_grad], cfg.lr, cfg.betas,
                                              0.01, device=device)}
        self.schedulers = {}
        self.train_set = PairDataset(self.manifest, cfg, self.caches, None if isinstance(self.manifest, list) else "train",
                                     tc.seed, syn, self.fixed_crops)
        self.val_set = PairDataset(self.val_manifest, cfg, self.caches, None, tc.seed + 1, syn) if self.val_manifest else None

    def train_step(self, b: dict[str, Any], ctx: StepContext) -> dict[str, float]:
        with ctx.autocast():
            out = self.net(b)
        ctx.optimize("main", out["loss"])
        return {k: float(v.detach()) for k, v in out.items()}

    def evaluate(self, batches, ctx: StepContext) -> dict[str, float]:
        tot: dict[str, float] = {}
        for b in batches:
            for k, v in self.net(b).items():
                tot[k] = tot.get(k, 0.0) + float(v) / len(batches)
        return tot

    def checkpoint_meta(self, step: int) -> CheckpointMeta:
        cfg = self.cfg
        return CheckpointMeta(
            engine=INFO.key, model_name=INFO.model_name,
            weights_status=(WeightsStatus.FINETUNED_FROM_PUBLIC_BASE if step > 0 and not cfg.test_tiny else
                            WeightsStatus.RANDOM_INIT).value,
            source_status=INFO.source_status.value, arch_confidence=INFO.arch_confidence.value, config=asdict(cfg),
            datasets=self.datasets, component_licenses={"base model": str(self.base), "aivoice code": "Apache-2.0",
                                                         "emotion teacher": "emotion2vec (training only; see its card)"},
            redistributable=False,
            notes=("TEST: tiny random model. " if cfg.test_tiny else "") +
                  "Fine-tuned by an aivoice user from the public StreamVoiceAnon base; NOT the authors' "
                  "StreamVoiceAnon+ weights (unreleased).")
