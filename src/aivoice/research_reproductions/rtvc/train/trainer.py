"""RT-VC training recipe (aivoice; the official repository has no training code).

stage ``pitch``  : causal pitch tracker on log-mel -> BCE on blurred pitch bins, BCE periodicity, L1 loudness
stage ``ema``    : causal EMA inverter on MFCC -> L1 to articulatory pseudo-labels (SPARC)
stage ``vocoder``: speaker encoder + DDSP vocoder on teacher features -> multi-scale spectral loss
``reproduce export rtvc`` then assembles pitch + ema + vocoder + speaker into one inference checkpoint.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch

from ...common.checkpoint import CheckpointMeta
from ...common.provenance import WeightsStatus
from ...common.trainer import StepContext, TrainConfig, make_adamw
from ..config import FRAME, RTVCConfig
from ..info import INFO
from ..model.encoder import EMAInverter, PitchTracker
from ..model.frontend import CausalMelMFCC
from ..model.losses import MultiScaleSpectralLoss, ema_loss, pitch_losses
from .dataset import RTVCDataset


class StandInEMA(torch.nn.Module):
    """TEST STAND-IN for SPARC pseudo-labels: fixed random projection of the MFCCs. Not articulation."""

    def __init__(self, n_mfcc: int, ch: int) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(7)
        self.register_buffer("w", torch.randn(ch, n_mfcc, generator=g) / n_mfcc ** 0.5)

    def forward(self, mfcc: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.w @ (mfcc / 50.0))


class RTVCRecipe:
    engine = INFO.key

    def __init__(self, cfg: RTVCConfig, manifest, val_manifest=None, cache_dir: str | None = None, ema_cache=None,
                 pitch_cache=None, teacher_dirs: dict[str, str] | None = None, datasets: list[str] | None = None,
                 fixed_crops: bool = False) -> None:
        self.cfg, self.manifest, self.val_manifest, self.cache_dir = cfg, manifest, val_manifest, cache_dir
        self.ema_cache, self.pitch_cache, self.teacher_dirs = ema_cache, pitch_cache, teacher_dirs or {}
        self.datasets, self.fixed_crops = datasets or [], fixed_crops

    @property
    def overfit_loss_key(self) -> str:
        return {"pitch": "loss", "ema": "ema_l1", "vocoder": "mss"}[self.cfg.stage]

    def model_config(self) -> dict[str, Any]:
        return asdict(self.cfg)

    def build(self, tc: TrainConfig, device: torch.device) -> None:
        cfg = self.cfg
        fc = cfg.frontend
        self.fe = CausalMelMFCC(16000, fc.n_fft, FRAME, fc.n_mels, fc.n_mfcc).to(device)
        self.schedulers = {}
        if cfg.stage == "pitch":
            self.net = PitchTracker(cfg.pitch, fc.n_mels).to(device)
            self.models = {"pitch": self.net}
        elif cfg.stage == "ema":
            self.net = EMAInverter(cfg.ema, fc.n_mfcc).to(device)
            self.models = {"ema": self.net}
        elif cfg.stage == "vocoder":
            from ..model.decoder import DDSPVocoder
            from ..model.timbre import SpeakerEncoder

            self.voc = DDSPVocoder(cfg.vocoder, cfg.speaker.spkr_emb_dim, cfg.ema.channels).to(device)
            self.spk = SpeakerEncoder(cfg.speaker, self.teacher_dirs.get("wavlm-large"), cfg.test_standin).to(device)
            self.models = {"vocoder": self.voc, "speaker": self.spk}
            self.mss = MultiScaleSpectralLoss(cfg.loss.fft_sizes)
        else:
            raise ValueError(f"unknown RT-VC stage {cfg.stage!r} (pitch | ema | vocoder)")
        params = [p for m in self.models.values() for p in m.parameters() if p.requires_grad]
        self.optimizers = {"main": make_adamw(params, cfg.lr, cfg.betas, device=device)}
        self.standin_ema = StandInEMA(fc.n_mfcc, cfg.ema.channels).to(device) if cfg.test_standin else None
        kw = dict(cache_dir=self.cache_dir, ema_cache=self.ema_cache, pitch_cache=self.pitch_cache)
        self.train_set = RTVCDataset(self.manifest, cfg, None if isinstance(self.manifest, list) else "train",
                                     seed=tc.seed, fixed_crops=self.fixed_crops, **kw)
        self.val_set = RTVCDataset(self.val_manifest, cfg, None, seed=tc.seed + 1, **kw) if self.val_manifest else None

    def _ema_target(self, batch: dict[str, Any]) -> torch.Tensor:
        if "ema" in batch:
            return batch["ema"].transpose(1, 2).float()
        with torch.no_grad():
            return self.standin_ema(self.fe(batch["audio"])[1])

    def _losses(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        cfg, lc = self.cfg, self.cfg.loss
        if cfg.stage == "pitch":
            logmel, _ = self.fe(batch["input"])
            out = self.net(logmel)
            with torch.no_grad():
                loud = self.fe.loudness(batch["audio"])
            parts = pitch_losses(out, self.net.targets(batch["f0"]), batch["periodicity"][:, None], loud)
            loss = lc.pitch_bce * parts["pitch_bce"] + lc.periodicity_bce * parts["periodicity_bce"] + \
                lc.loudness_l1 * parts["loudness_l1"]
            return {"loss": loss, **parts}
        if cfg.stage == "ema":
            _, mfcc = self.fe(batch["input"])
            l = ema_loss(self.net(mfcc), self._ema_target(batch))
            return {"loss": lc.ema_l1 * l, "ema_l1": l}
        # vocoder: teacher-forced controls from the target segment itself
        y = batch["audio"]
        with torch.no_grad():
            loud = self.fe.loudness(y)
            ref_logmel, _ = self.fe(batch["reference"])
        n = loud.shape[-1]
        f0, per = batch["f0"][:, None, :n], batch["periodicity"][:, None, :n]
        ema = self._ema_target(batch)[..., :n]
        # reference periodicity for the speaker pooling: YIN-free proxy = normalised frame energy (training only)
        ref_per = torch.sigmoid(ref_logmel.mean(1, keepdim=True) - ref_logmel.mean())
        g = self.spk(batch["reference"], ref_per)
        # fresh noise every step; the overfit gate fixes it so the target is exactly reachable
        seed = 0 if self.fixed_crops else int(torch.randint(0, 2**31, (1,)))
        y_hat = self.voc(f0, loud, per, ema, g, seed=seed)
        m = min(y_hat.shape[-1], y.shape[-1])
        l = self.mss(y_hat[:, :m], y[:, :m])
        return {"loss": lc.mss * l, "mss": l}

    def train_step(self, batch: dict[str, Any], ctx: StepContext) -> dict[str, float]:
        with ctx.autocast():
            out = self._losses(batch)
        ctx.optimize("main", out["loss"])
        return {k: float(v.detach()) for k, v in out.items()}

    def evaluate(self, batches, ctx: StepContext) -> dict[str, float]:
        tot: dict[str, float] = {}
        for b in batches:
            for k, v in self._losses(b).items():
                tot[k] = tot.get(k, 0.0) + float(v) / len(batches)
        return tot

    def checkpoint_meta(self, step: int) -> CheckpointMeta:
        cfg = self.cfg
        lic = {"aivoice code": "Apache-2.0", "official RT-VC code": "not used (no license); layout re-implemented"}
        if cfg.stage == "vocoder":
            lic["wavlm-large CNN"] = "TEST STAND-IN" if cfg.test_standin else "MIT"
        if cfg.stage in ("ema", "vocoder"):
            lic["EMA pseudo-labels"] = "TEST STAND-IN" if cfg.test_standin else "SPARC (check its license)"
        return CheckpointMeta(
            engine=INFO.key, model_name=f"{INFO.model_name}-{cfg.stage}",
            weights_status=(WeightsStatus.RETRAINED_PAPER_REPRODUCTION if step > 0 else WeightsStatus.RANDOM_INIT).value,
            source_status=INFO.source_status.value, arch_confidence=INFO.arch_confidence.value, config=asdict(cfg),
            datasets=self.datasets, component_licenses=lic, redistributable=False,
            notes=("TEST STAND-IN TEACHERS: not a usable model. " if cfg.test_standin else "") +
                  "aivoice-trained RT-VC reconstruction; NOT official weights (none were released).")
