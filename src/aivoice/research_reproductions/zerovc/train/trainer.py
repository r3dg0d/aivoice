"""Zero-VC training recipe for the generic trainer.

Stage ``content``: train the causal content student (variant A: regress w2v-BERT 2.0 hidden states;
variant B: classify HuBERT k-means codes). Stage ``vc``: freeze it; train attention pooling + causal
HiFi-GAN decoder with MPD/MSD on (anonymised source -> original source) reconstruction, timbre from
WavLM-large layer 7 of another utterance of the same speaker.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

import torch

from ...common.blocks import set_grad_checkpoint
from ...common.checkpoint import CheckpointMeta, load_checkpoint
from ...common.losses import MelL1
from ...common.provenance import WeightsStatus
from ...common.teachers import load_teacher
from ...common.trainer import StepContext, TrainConfig, make_adamw
from ..config import SAMPLE_RATE, ZeroVCConfig
from ..info import INFO, model_name
from ..model.discriminator import HiFiGANDiscriminator
from ..model.losses import discriminator_loss, generator_loss
from ..model.zerovc import ZeroVC
from .dataset import ZeroVCDataset


class ZeroVCRecipe:
    engine = INFO.key

    def __init__(self, cfg: ZeroVCConfig, manifest: str | list, val_manifest: str | list | None = None,
                 content_checkpoint: str | None = None, teacher_dirs: dict[str, str] | None = None,
                 cache_dir: str | None = None, teacher_cache=None, timbre_cache=None, datasets: list[str] | None = None,
                 kmeans_centroids: str | None = None, fixed_crops: bool = False) -> None:
        self.cfg, self.manifest, self.val_manifest = cfg, manifest, val_manifest
        self.content_checkpoint, self.teacher_dirs = content_checkpoint, teacher_dirs or {}
        self.cache_dir, self.teacher_cache, self.timbre_cache = cache_dir, teacher_cache, timbre_cache
        self.datasets = datasets or []
        self.kmeans_centroids = kmeans_centroids
        self.fixed_crops = fixed_crops

    @property
    def overfit_loss_key(self) -> str:
        return "loss" if self.cfg.stage == "content" else "mel"

    def model_config(self) -> dict[str, Any]:
        return asdict(self.cfg)

    # -- build ------------------------------------------------------------------------------------
    def build(self, tc: TrainConfig, device: torch.device) -> None:
        cfg = self.cfg
        self.device = device
        net = ZeroVC(cfg).to(device)
        set_grad_checkpoint(net, tc.grad_checkpoint)
        self.net = net
        self.schedulers = {}
        o = cfg.optim
        if cfg.stage == "content":
            self.models = {"content": net.content}
            self.optimizers = {"content": make_adamw(net.content.parameters(), o.lr, o.betas, o.weight_decay,
                                                     device=device)}
            c = cfg.content
            if self.teacher_cache is None:
                if c.variant == "A":
                    self.teacher = load_teacher("w2v-bert-2.0", c.teacher_layer, c.teacher_dim, cfg.test_standin,
                                                self.teacher_dirs.get("w2v-bert-2.0")).to(device)
                else:
                    self.teacher = load_teacher("hubert-base", c.km_layer, 768 if not cfg.test_standin else c.dim,
                                                cfg.test_standin, self.teacher_dirs.get("hubert-base")).to(device)
                    if self.kmeans_centroids:
                        import numpy as np

                        self.centroids = torch.from_numpy(np.load(self.kmeans_centroids)).to(device)
                    elif cfg.test_standin:
                        g = torch.Generator().manual_seed(0)
                        self.centroids = torch.randn(c.km_codes, self.teacher.dim, generator=g).to(device)
                    else:
                        raise ValueError("variant B needs --kmeans CENTROIDS.npy (from `aivoice reproduce prepare`)")
        else:
            if self.content_checkpoint:
                blob = load_checkpoint(self.content_checkpoint, engine=INFO.key, allow_config_change=True,
                                       map_location=device)
                if blob["meta"]["config"]["content"] != json.loads(json.dumps(asdict(cfg.content))):
                    raise ValueError("content checkpoint was trained with a different content config")
                net.content.load_state_dict(blob["models"]["content"])
                self.content_source = {"checkpoint": self.content_checkpoint, "step": blob["meta"]["step"]}
            elif not cfg.test_standin:
                raise ValueError("stage vc needs --content-checkpoint (train stage `content` first)")
            else:
                self.content_source = {"checkpoint": None, "note": "TEST: untrained content encoder"}
            net.content.requires_grad_(False).eval()
            self.disc = HiFiGANDiscriminator(ch_mult=cfg.disc_ch_mult).to(device)
            # the frozen content encoder is saved too, so a vc checkpoint is a complete inference model
            self.models = {"content": net.content, "timbre": net.timbre, "decoder": net.decoder, "disc": self.disc}
            gparams = list(net.timbre.parameters()) + list(net.decoder.parameters())
            self.optimizers = {"g": make_adamw(gparams, o.lr, o.betas, o.weight_decay, device=device),
                               "d": make_adamw(self.disc.parameters(), o.lr, o.betas, o.weight_decay, device=device)}
            gamma = o.lr_decay ** (1 / 1000)
            self.schedulers = {k: torch.optim.lr_scheduler.ExponentialLR(v, gamma) for k, v in self.optimizers.items()}
            self.mel = MelL1(SAMPLE_RATE, 1024, 256, 80).to(device)
            if self.timbre_cache is None:
                self.wavlm = load_teacher("wavlm-large", cfg.timbre.ssl_layer, cfg.timbre.ssl_dim, cfg.test_standin,
                                          self.teacher_dirs.get("wavlm-large")).to(device)
        self.train_set = ZeroVCDataset(self.manifest, cfg, split=None if isinstance(self.manifest, list) else "train",
                                       cache_dir=self.cache_dir, teacher_cache=self.teacher_cache,
                                       timbre_cache=self.timbre_cache, seed=tc.seed,
                                       fixed_crops=self.fixed_crops)
        self.val_set = ZeroVCDataset(self.val_manifest, cfg, split=None, cache_dir=self.cache_dir, seed=tc.seed + 1,
                                     teacher_cache=self.teacher_cache, timbre_cache=self.timbre_cache) \
            if self.val_manifest else None

    # -- steps ------------------------------------------------------------------------------------
    def _teacher_target(self, batch: dict[str, Any]) -> torch.Tensor:
        if "teacher" in batch:
            return batch["teacher"]
        with torch.no_grad():
            h = self.teacher(batch["audio"])
            if self.cfg.content.variant == "B":
                return torch.cdist(h.float(), self.centroids[None]).argmin(-1)
            return h.float()

    def _timbre(self, batch: dict[str, Any]) -> torch.Tensor:
        feats = batch["ref_feats"] if "ref_feats" in batch else self.wavlm(batch["reference"])
        return self.net.timbre(feats.float())

    def train_step(self, batch: dict[str, Any], ctx: StepContext) -> dict[str, float]:
        if self.cfg.stage == "content":
            target = self._teacher_target(batch)
            with ctx.autocast():
                out = self.net.content.distill_loss(batch["audio"], target)
            ctx.optimize("content", out["loss"], self.net.content)
            return {k: float(v.detach()) for k, v in out.items()}
        y = batch["audio"][:, None]
        with torch.no_grad():
            h = self.net.content.features(batch["content_in"])
        with ctx.autocast():
            g = self._timbre(batch)
            y_hat = self.net.decoder(h, g)
            d_real, _ = self.disc(y)
            d_fake, _ = self.disc(y_hat.detach())
            l_d = discriminator_loss(d_real, d_fake)
        ctx.optimize("d", l_d, self.disc)
        with ctx.frozen(self.disc), ctx.autocast():
            d_fake, f_fake = self.disc(y_hat)
            _, f_real = self.disc(y)
            out = generator_loss(self.cfg.loss, self.mel, y_hat.float(), y, d_fake, f_real, f_fake)
        ctx.optimize("g", out["g_total"])
        return {"d": float(l_d.detach()), **{k: float(v.detach()) for k, v in out.items()}}

    def evaluate(self, batches: list[dict[str, Any]], ctx: StepContext) -> dict[str, float]:
        tot: dict[str, float] = {}
        for b in batches:
            if self.cfg.stage == "content":
                out = {k: float(v) for k, v in self.net.content.distill_loss(b["audio"], self._teacher_target(b)).items()}
            else:
                y_hat = self.net.decoder(self.net.content.features(b["content_in"]), self._timbre(b))
                out = {"mel": float(self.mel(y_hat[:, 0].float(), b["audio"]))}
            for k, v in out.items():
                tot[k] = tot.get(k, 0.0) + v / len(batches)
        return tot

    def checkpoint_meta(self, step: int) -> CheckpointMeta:
        cfg = self.cfg
        lic = {"aivoice code": "Apache-2.0"}
        if cfg.stage == "content":
            lic["teacher"] = ("TEST STAND-IN" if cfg.test_standin else
                              "w2v-bert-2.0: MIT" if cfg.content.variant == "A" else "hubert-base: Apache-2.0/MIT")
        else:
            lic["timbre teacher"] = "TEST STAND-IN" if cfg.test_standin else "wavlm-large: MIT"
            lic["content encoder"] = str(self.content_source)
        return CheckpointMeta(
            engine=INFO.key, model_name=model_name(cfg.content.variant, cfg.stage),
            weights_status=(WeightsStatus.RETRAINED_PAPER_REPRODUCTION if step > 0 else WeightsStatus.RANDOM_INIT).value,
            source_status=INFO.source_status.value, arch_confidence=INFO.arch_confidence.value,
            config=asdict(cfg), datasets=self.datasets, component_licenses=lic, redistributable=False,
            notes=("TEST STAND-IN TEACHERS: not a usable model. " if cfg.test_standin else "") +
                  "Weights trained with aivoice's Zero-VC reproduction; not official. Redistribution depends on the "
                  "training data licenses (LibriTTS: CC BY 4.0) and is never automatic.")
