"""TVTSyn recipe. Stage ``content``: PASR content encoder with FVQ on HuBERT-L9 k-means-200 labels.
Stage ``synth``: content encoder frozen; time-varying timbre + decoder with MPD/MSD; 20 mel / 1 adv / 2 fm /
20 F0-energy."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

import numpy as np
import torch

from ...common.blocks import set_grad_checkpoint
from ...common.checkpoint import CheckpointMeta, load_checkpoint
from ...common.losses import MelL1
from ...common.provenance import WeightsStatus
from ...common.trainer import StepContext, TrainConfig, make_adamw
from ..config import SAMPLE_RATE, TVTSynConfig
from ..info import INFO, model_name
from ..model.discriminator import HiFiGANDiscriminator
from ..model.losses import discriminator_loss, generator_loss
from ..model.tvtsyn import TVTSyn
from .dataset import TVTSynDataset


class _StandIns:
    """TEST STAND-INS for HuBERT k-means codes and x-vector/ECAPA (frozen random functions of the audio)."""

    def __init__(self, cfg: TVTSynConfig) -> None:
        rng = np.random.default_rng(3)
        self.wc = rng.standard_normal((cfg.content.num_codes, 64)).astype(np.float32)
        self.ws = rng.standard_normal((cfg.timbre.g_dim, 64)).astype(np.float32)

    def _feat(self, x: np.ndarray, hop: int = 320) -> np.ndarray:
        n = len(x) // hop
        fr = x[: n * hop].reshape(n, hop)
        spec = np.log(np.abs(np.fft.rfft(fr, 126, axis=1)) + 1e-4)[:, :64]
        return spec.astype(np.float32)

    def codes(self, x: np.ndarray) -> np.ndarray:
        return (self._feat(x) @ self.wc.T).argmax(1).astype(np.int64)

    def spk(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(self.ws @ self._feat(x).mean(0) / 8).astype(np.float32)


class TVTSynRecipe:
    def __init__(self, cfg: TVTSynConfig, manifest, val_manifest=None, content_checkpoint: str | None = None,
                 cache_dir: str | None = None, code_cache=None, spk_cache=None, datasets: list[str] | None = None,
                 fixed_crops: bool = False) -> None:
        self.cfg, self.manifest, self.val_manifest = cfg, manifest, val_manifest
        self.content_checkpoint, self.cache_dir = content_checkpoint, cache_dir
        self.code_cache, self.spk_cache, self.datasets, self.fixed_crops = code_cache, spk_cache, datasets or [], fixed_crops

    @property
    def engine(self) -> str:  # per-variant registry name
        return INFO.key if self.cfg.variant == "repro" else "tvtsyn-aivoice"

    @property
    def overfit_loss_key(self) -> str:
        return "ce" if self.cfg.stage == "content" else "mel"

    def model_config(self) -> dict[str, Any]:
        return asdict(self.cfg)

    def build(self, tc: TrainConfig, device: torch.device) -> None:
        cfg = self.cfg
        self.net = TVTSyn(cfg).to(device)
        set_grad_checkpoint(self.net, tc.grad_checkpoint)
        self.schedulers = {}
        if cfg.stage == "content":
            self.models = {"content": self.net.content}
            self.optimizers = {"content": make_adamw(self.net.content.parameters(), cfg.content_lr, (0.9, 0.999),
                                                     1e-6, device=device)}
        else:
            if self.content_checkpoint:
                blob = load_checkpoint(self.content_checkpoint, engine=self.engine, allow_config_change=True,
                                       map_location=device)
                if blob["meta"]["config"]["content"] != json.loads(json.dumps(asdict(cfg.content))):
                    raise ValueError("content checkpoint was trained with a different content config")
                self.net.content.load_state_dict(blob["models"]["content"])
                self.content_source = {"checkpoint": self.content_checkpoint, "step": blob["meta"]["step"]}
            elif not cfg.test_standin:
                raise ValueError("stage synth needs --content-checkpoint (train stage `content` first)")
            else:
                self.content_source = {"checkpoint": None, "note": "TEST: untrained content encoder"}
            self.net.content.requires_grad_(False).eval()
            self.disc = HiFiGANDiscriminator(ch_mult=cfg.disc_ch_mult).to(device)
            self.models = {"content": self.net.content, "timbre": self.net.timbre, "decoder": self.net.decoder,
                           "disc": self.disc}
            gp = list(self.net.timbre.parameters()) + list(self.net.decoder.parameters())
            self.optimizers = {"g": make_adamw(gp, cfg.lr, cfg.betas, device=device),
                               "d": make_adamw(self.disc.parameters(), cfg.lr, cfg.betas, device=device)}
            self.schedulers = {k: torch.optim.lr_scheduler.ExponentialLR(o, cfg.lr_gamma) for k, o in self.optimizers.items()}
            self.mel = MelL1(SAMPLE_RATE, 1024, 320, 80).to(device)
        si = _StandIns(cfg) if cfg.test_standin else None
        kw = dict(cache_dir=self.cache_dir, code_cache=self.code_cache, spk_cache=self.spk_cache,
                  standin_codes=si.codes if si else None, standin_spk=si.spk if si else None)
        self.train_set = TVTSynDataset(self.manifest, cfg, None if isinstance(self.manifest, list) else "train",
                                       seed=tc.seed, fixed_crops=self.fixed_crops, **kw)
        self.val_set = TVTSynDataset(self.val_manifest, cfg, None, seed=tc.seed + 1, **kw) if self.val_manifest else None

    def _synth(self, b: dict[str, Any]):
        with torch.no_grad():
            z = self.net.content(b["audio"])["z_q"]
        tv, _ = self.net.timbre(z, b["g"])
        return self.net.decoder(z, tv, b["pitch"], b["energy"])

    def train_step(self, b: dict[str, Any], ctx: StepContext) -> dict[str, float]:
        if self.cfg.stage == "content":
            with ctx.autocast():
                out = self.net.content.pasr_loss(b["audio"], b["codes"])
            ctx.optimize("content", out["loss"], self.net.content)
            return {k: float(v.detach()) for k, v in out.items()}
        y = b["audio"][:, None]
        with ctx.autocast():
            y_hat, var = self._synth(b)
            y_hat = y_hat[..., : y.shape[-1]]
            l_d = discriminator_loss(self.disc(y)[0], self.disc(y_hat.detach())[0])
        ctx.optimize("d", l_d, self.disc)
        with ctx.frozen(self.disc), ctx.autocast():
            d_fake, f_fake = self.disc(y_hat)
            _, f_real = self.disc(y)
            out = generator_loss(self.cfg.loss, self.mel, y_hat.float(), y, d_fake, f_real, f_fake, var,
                                 b["pitch"], b["energy"])
        ctx.optimize("g", out["g_total"])
        return {"d": float(l_d.detach()), **{k: float(v.detach()) for k, v in out.items()}}

    def evaluate(self, batches, ctx: StepContext) -> dict[str, float]:
        tot: dict[str, float] = {}
        for b in batches:
            if self.cfg.stage == "content":
                out = {k: float(v) for k, v in self.net.content.pasr_loss(b["audio"], b["codes"]).items()}
            else:
                y_hat, _ = self._synth(b)
                out = {"mel": float(self.mel(y_hat[:, 0, : b["audio"].shape[-1]].float(), b["audio"]))}
            for k, v in out.items():
                tot[k] = tot.get(k, 0.0) + v / len(batches)
        return tot

    def checkpoint_meta(self, step: int) -> CheckpointMeta:
        cfg = self.cfg
        lic = {"aivoice code": "Apache-2.0",
               "architecture reference": "VOSSA official code (CC BY-NC 4.0): read for verification, not copied",
               "teachers": "TEST STAND-IN" if cfg.test_standin else
               "HuBERT-base (Apache-2.0), speechbrain x-vector / ECAPA (Apache-2.0)"}
        if cfg.stage == "synth":
            lic["content encoder"] = str(self.content_source)
        return CheckpointMeta(
            engine=self.engine, model_name=model_name(cfg.variant, cfg.stage),
            weights_status=(WeightsStatus.RETRAINED_PAPER_REPRODUCTION if step > 0 else WeightsStatus.RANDOM_INIT).value,
            source_status=INFO.source_status.value, arch_confidence=INFO.arch_confidence.value, config=asdict(cfg),
            datasets=self.datasets, component_licenses=lic, redistributable=False,
            notes=("TEST STAND-IN TEACHERS: not a usable model. " if cfg.test_standin else "") +
                  "aivoice-trained TVTSyn reproduction; NOT official weights.")
