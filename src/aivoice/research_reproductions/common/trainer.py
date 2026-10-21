"""Generic trainer for the paper reproductions.

A *recipe* (one per reproduction) owns the models, losses and optimizers and implements one
training step; this module owns everything that must be identical across reproductions: seeding,
device/precision (BF16/FP16 autocast, TF32), gradient accumulation and clipping, torch.compile,
DDP (``torchrun``), exact resume, local metrics (JSONL, CSV, TensorBoard; W&B only when explicitly
enabled), checkpoints, a status file and pause/resume/stop control for the background service, OOM
reports, profiling and throughput/cost measurement.

Nothing here changes a hyper-parameter on its own. An out-of-memory error stops the run with a
report and suggestions; it never retries with a smaller batch.
"""

from __future__ import annotations

import contextlib
import csv
import json
import math
import os
import random
import sys
import time
import traceback
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .checkpoint import (
    CheckpointMeta,
    load_checkpoint,
    load_model_state,
    save_checkpoint,
    set_rng_state,
)
from .data import ResumableSampler, collate


@dataclass
class TrainConfig:
    output: str
    device: str = "auto"            # auto | cpu | cuda | cuda:N
    precision: str = "bf16"         # bf16 | fp16 | fp32
    batch_size: int = 16
    grad_accum: int = 1
    workers: int = 4
    seed: int = 1234
    max_steps: int = 100_000
    checkpoint_every: int = 5_000
    eval_every: int = 2_500
    log_every: int = 50
    compile: bool = False
    profile: bool = False           # profile steps 2-5 then continue
    tf32: bool = True
    grad_clip: float | None = None  # recipe default when None
    grad_checkpoint: bool = False   # activation checkpointing in transformer stacks
    keep_checkpoints: int = 3
    wandb: bool = False             # never on by default
    wandb_project: str = "aivoice-reproductions"
    resume: str | None = None       # checkpoint path, or "latest"
    allow_config_change: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


class StepContext:
    """Passed to ``Recipe.train_step``: autocast, accumulation-aware optimisation, frozen modules."""

    def __init__(self, trainer: Trainer) -> None:
        self.t = trainer

    @property
    def device(self) -> torch.device:
        return self.t.device

    @property
    def step(self) -> int:
        return self.t.step

    def autocast(self):
        return self.t.autocast()

    def optimize(self, name: str, loss: torch.Tensor, model: nn.Module | None = None,
                 clip: float | None = None) -> None:
        """Backward ``loss / grad_accum``; on the accumulation boundary clip, step and zero ``name``."""
        t = self.t
        t.scaler.scale(loss / t.cfg.grad_accum).backward()
        if not t.boundary:
            return
        opt = t.optimizers[name]
        clip = clip if clip is not None else t.cfg.grad_clip
        if clip:
            t.scaler.unscale_(opt)
            params = model.parameters() if model is not None else [p for g in opt.param_groups for p in g["params"]]
            gn = torch.nn.utils.clip_grad_norm_(params, clip)
            t.last_grad_norm[name] = float(gn)
        t.scaler.step(opt)
        opt.zero_grad(set_to_none=True)
        t.stepped.add(name)

    @contextlib.contextmanager
    def frozen(self, *models: nn.Module) -> Iterator[None]:
        """Block gradients into ``models`` (e.g. the discriminator during the generator step)."""
        flags = [[p.requires_grad for p in m.parameters()] for m in models]
        for m in models:
            m.requires_grad_(False)
        try:
            yield
        finally:
            for m, fl in zip(models, flags):
                for p, f in zip(m.parameters(), fl):
                    p.requires_grad_(f)


class Recipe(Protocol):
    engine: str

    def build(self, cfg: TrainConfig, device: torch.device) -> None:
        """Create ``self.models``, ``self.optimizers``, ``self.schedulers`` and the datasets."""

    models: dict[str, nn.Module]
    optimizers: dict[str, torch.optim.Optimizer]
    schedulers: dict[str, Any]
    train_set: Any
    val_set: Any

    def train_step(self, batch: dict[str, Any], ctx: StepContext) -> dict[str, float]: ...

    def evaluate(self, batches: list[dict[str, Any]], ctx: StepContext) -> dict[str, float]: ...

    def checkpoint_meta(self, step: int) -> CheckpointMeta: ...

    def model_config(self) -> dict[str, Any]: ...


def make_adamw(params, lr: float, betas=(0.9, 0.999), weight_decay: float = 0.0, eps: float = 1e-8,
               device: torch.device | None = None) -> torch.optim.Optimizer:
    """AdamW, fused on CUDA (one kernel per step)."""
    fused = device is not None and device.type == "cuda"
    return torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay, eps=eps, fused=fused)


class MetricsLogger:
    """metrics.jsonl + metrics.csv (long format: step,name,value) + TensorBoard (if installed) + W&B (opt-in)."""

    def __init__(self, out: Path, cfg: TrainConfig, run_config: dict[str, Any], enabled: bool = True) -> None:
        self.enabled = enabled
        self.out = out
        if not enabled:
            return
        self.jsonl = (out / "metrics.jsonl").open("a")
        new = not (out / "metrics.csv").exists()
        self.csvf = (out / "metrics.csv").open("a", newline="")
        self.csv = csv.writer(self.csvf)
        if new:
            self.csv.writerow(["step", "time", "name", "value"])
        self.tb = None
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.tb = SummaryWriter(str(out / "tensorboard"))
        except Exception:  # noqa: BLE001 - tensorboard is optional
            self.tb = None
        self.wb = None
        if cfg.wandb:  # explicit opt-in only; sends metrics (never audio) to wandb.ai
            import wandb

            self.wb = wandb.init(project=cfg.wandb_project, config=run_config, dir=str(out))

    def log(self, step: int, metrics: dict[str, float]) -> None:
        if not self.enabled:
            return
        now = time.time()
        self.jsonl.write(json.dumps({"step": step, "time": now, **metrics}) + "\n")
        self.jsonl.flush()
        for k, v in metrics.items():
            self.csv.writerow([step, f"{now:.3f}", k, v])
            if self.tb is not None:
                self.tb.add_scalar(k, v, step)
        self.csvf.flush()
        if self.wb is not None:
            self.wb.log(metrics, step=step)

    def close(self) -> None:
        if not self.enabled:
            return
        self.jsonl.close()
        self.csvf.close()
        if self.tb is not None:
            self.tb.close()
        if self.wb is not None:
            self.wb.finish()


def oom_report(cfg: TrainConfig, device: torch.device, err: BaseException) -> dict[str, Any]:
    peak = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None
    total = torch.cuda.get_device_properties(device).total_memory / 2**30 if device.type == "cuda" else None
    sugg = []
    if cfg.batch_size > 1:
        sugg.append(f"--batch-size {max(1, cfg.batch_size // 2)} --grad-accum {cfg.grad_accum * 2} "
                    "(same effective batch)")
    if cfg.precision == "fp32":
        sugg.append("--precision bf16")
    if not cfg.grad_checkpoint:
        sugg.append("--grad-checkpoint (recompute activations in transformer stacks)")
    sugg.append("a shorter training segment in the recipe config (changes the recipe; record it)")
    return {"error": "out_of_memory", "message": str(err).splitlines()[0][:300], "peak_allocated_gib": peak,
            "device_total_gib": total, "batch_size": cfg.batch_size, "grad_accum": cfg.grad_accum,
            "precision": cfg.precision, "suggestions": sugg,
            "note": "nothing was changed automatically; rerun with one of the suggestions"}


class Trainer:
    def __init__(self, recipe: Recipe, cfg: TrainConfig) -> None:
        self.recipe, self.cfg = recipe, cfg
        self.out = Path(cfg.output)
        self.out.mkdir(parents=True, exist_ok=True)
        self.rank = int(os.environ.get("RANK", "0"))
        self.world = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.device = self._device()
        self.step = 0
        self.boundary = True
        self.stepped: set[str] = set()
        self.last_grad_norm: dict[str, float] = {}
        self.status: dict[str, Any] = {"state": "starting", "pid": os.getpid(), "engine": recipe.engine}
        self._t0 = time.time()

    # -- setup ------------------------------------------------------------------------------------
    def _device(self) -> torch.device:
        d = self.cfg.device
        if d == "auto":
            d = "cuda" if torch.cuda.is_available() else "cpu"
        if d.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("--device cuda requested but CUDA is not available")
            if self.world > 1:
                d = f"cuda:{self.local_rank}"
            torch.cuda.set_device(torch.device(d))
        if self.cfg.precision == "fp16" and not d.startswith("cuda"):
            raise RuntimeError("--precision fp16 needs CUDA; use bf16 or fp32 on CPU")
        return torch.device(d)

    def autocast(self):
        p = self.cfg.precision
        if p == "fp32":
            return contextlib.nullcontext()
        dtype = torch.bfloat16 if p == "bf16" else torch.float16
        return torch.autocast(self.device.type, dtype=dtype)

    def _seed(self) -> None:
        s = self.cfg.seed + self.rank
        random.seed(s)
        np.random.seed(s % 2**32)
        torch.manual_seed(s)

    def _setup(self) -> None:
        cfg = self.cfg
        self._seed()
        if self.device.type == "cuda" and cfg.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        if self.world > 1 and not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl" if self.device.type == "cuda" else "gloo")
        self.recipe.build(cfg, self.device)
        self.models = self.recipe.models
        self.optimizers = self.recipe.optimizers
        self.schedulers = getattr(self.recipe, "schedulers", {}) or {}
        if self.world > 1:
            from torch.nn.parallel import DistributedDataParallel as DDP

            for k, m in list(self.models.items()):
                if any(p.requires_grad for p in m.parameters()):
                    self.models[k] = DDP(m, device_ids=[self.device.index] if self.device.type == "cuda" else None,
                                         find_unused_parameters=True)
            self.recipe.models = self.models
        if cfg.compile:
            for k, m in list(self.models.items()):
                self.models[k] = torch.compile(m)
            self.recipe.models = self.models
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=cfg.precision == "fp16")
        self.sampler = ResumableSampler(len(self.recipe.train_set), seed=cfg.seed, rank=self.rank,
                                        world_size=self.world)
        self.data_pos = {"epoch": 0, "pos": 0}
        if cfg.resume:
            self._resume(cfg.resume)
        run_cfg = {"train": asdict(cfg), "model": self.recipe.model_config(), "data": self._data_record(),
                   "optimizers": {k: [{h: v for h, v in g.items() if h != "params"} for g in o.param_groups]
                                  for k, o in self.optimizers.items()}}
        self.run_record = run_cfg
        if self.rank == 0:
            (self.out / "run_config.json").write_text(json.dumps(run_cfg, indent=2, default=str) + "\n")
        self.logger = MetricsLogger(self.out, cfg, run_cfg, enabled=self.rank == 0)

    def _data_record(self) -> dict[str, Any]:
        """Manifest path + sha256 (reproducibility: which data a run saw)."""
        from .checkpoint import sha256_file

        m = getattr(self.recipe, "manifest", None)
        if isinstance(m, (str, Path)) and Path(m).exists():
            return {"manifest": str(Path(m).resolve()), "sha256": sha256_file(m), "items": len(self.recipe.train_set)}
        return {"manifest": "in-memory list" if m is not None else None, "items": len(self.recipe.train_set)}

    def _loader(self) -> Iterator[dict[str, Any]]:
        cfg = self.cfg
        while True:
            ds = self.recipe.train_set
            if hasattr(ds, "epoch"):
                ds.epoch = self.sampler.epoch
            # own generator: a DataLoader otherwise draws its worker base seed from the global RNG, which would
            # make a resumed run diverge from an uninterrupted one
            gen = torch.Generator().manual_seed(cfg.seed * 7919 + self.sampler.epoch * 104729 + self.rank)
            loader = DataLoader(ds, batch_size=cfg.batch_size, sampler=self.sampler, num_workers=cfg.workers,
                                collate_fn=collate, pin_memory=self.device.type == "cuda", drop_last=True,
                                persistent_workers=False, prefetch_factor=4 if cfg.workers > 0 else None,
                                generator=gen)
            # DataLoader workers prefetch, so the sampler runs ahead of training; the resume point is
            # what training actually consumed, tracked here.
            epoch, pos = self.sampler.epoch, self.sampler.pos
            n = 0
            for batch in loader:
                n += 1
                self.data_pos = {"epoch": epoch, "pos": pos + n * cfg.batch_size}
                yield batch
            self.sampler.epoch, self.sampler.pos = epoch + 1, 0
            self.data_pos = {"epoch": epoch + 1, "pos": 0}
            if n == 0 and len(ds) < cfg.batch_size * self.world:  # (an empty remainder after a resume is fine)
                raise RuntimeError(f"training set has fewer items ({len(ds)}) than one batch ({cfg.batch_size})")

    # -- checkpoints ------------------------------------------------------------------------------
    def _ckpt_dir(self) -> Path:
        return self.out / "checkpoints"

    def save(self, tag: str | None = None) -> Path | None:
        if self.rank != 0:
            return None
        meta = self.recipe.checkpoint_meta(self.step)
        meta.step = self.step
        meta.epoch = self.data_pos["epoch"]
        path = self._ckpt_dir() / f"{tag or f'step{self.step:08d}'}.pt"
        save_checkpoint(path, meta, self.models, optimizers=self.optimizers, schedulers=self.schedulers,
                        scaler=self.scaler, sampler_state={**self.sampler.state_dict(), **self.data_pos},
                        extra={"train_config": asdict(self.cfg), "data": self.run_record["data"],
                               "optimizers": self.run_record["optimizers"]})
        (self._ckpt_dir() / "latest").write_text(path.name + "\n")
        self.status["last_checkpoint"] = str(path)
        self._prune()
        return path

    def _prune(self) -> None:
        ck = sorted(self._ckpt_dir().glob("step*.pt"))
        for p in ck[: -self.cfg.keep_checkpoints] if self.cfg.keep_checkpoints > 0 else []:
            for q in (p, Path(str(p) + ".sha256"), p.with_suffix(".json")):
                q.unlink(missing_ok=True)

    def _resume(self, which: str) -> None:
        path = Path(which)
        if which == "latest":
            ptr = self._ckpt_dir() / "latest"
            if not ptr.exists():
                raise FileNotFoundError(f"--resume latest: no checkpoints in {self._ckpt_dir()}")
            path = self._ckpt_dir() / ptr.read_text().strip()
        blob = load_checkpoint(path, engine=self.recipe.engine, config=self.recipe.model_config(),
                               allow_config_change=self.cfg.allow_config_change, map_location=self.device)
        for k, m in self.models.items():
            load_model_state(m, blob["models"][k], name=k)
        for k, o in self.optimizers.items():
            o.load_state_dict(blob["optimizers"][k])
        for k, s in self.schedulers.items():
            s.load_state_dict(blob["schedulers"][k])
        if blob.get("scaler"):
            self.scaler.load_state_dict(blob["scaler"])
        self.sampler.load_state_dict(blob["sampler"])
        self.data_pos = {"epoch": self.sampler.epoch, "pos": self.sampler.pos}
        set_rng_state(blob["rng"])
        self.step = int(blob["meta"]["step"])
        self.status["resumed_from"] = str(path)

    # -- control / status -------------------------------------------------------------------------
    def _write_status(self, **kw: Any) -> None:
        if self.rank != 0:
            return
        self.status.update(kw, updated=time.time(), step=self.step, max_steps=self.cfg.max_steps)
        if self.device.type == "cuda":
            self.status["gpu"] = torch.cuda.get_device_name(self.device)
            self.status["vram_gib"] = round(torch.cuda.max_memory_allocated(self.device) / 2**30, 2)
        tmp = self.out / "status.json.tmp"
        tmp.write_text(json.dumps(self.status, indent=2, default=str) + "\n")
        tmp.replace(self.out / "status.json")

    def _control(self) -> str | None:
        p = self.out / "control.json"
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text()).get("command")
        except (json.JSONDecodeError, OSError):
            return None

    def _handle_control(self) -> bool:
        """Returns True to stop."""
        cmd = self._control()
        if cmd == "stop":
            return True
        if cmd == "pause":
            self.save()
            self._write_status(state="paused")
            while (cmd := self._control()) == "pause":
                time.sleep(1.0)
            if cmd == "stop":
                return True
            self._write_status(state="training")
        return False

    # -- loop -------------------------------------------------------------------------------------
    def fit(self) -> dict[str, Any]:
        if self.rank == 0:  # a stale stop/pause request from a previous run must not affect this one
            (self.out / "control.json").unlink(missing_ok=True)
        try:
            self._setup()
        except torch.cuda.OutOfMemoryError as e:
            return self._fail_oom(e)
        except Exception as e:
            self._write_status(state="failed", error=f"{type(e).__name__}: {e}")
            raise
        ctx = StepContext(self)
        cfg = self.cfg
        data = self._loader()
        val_batches = self._val_batches()
        prof = self._profiler() if cfg.profile else None
        step_times: list[float] = []
        samples = 0
        window_t0 = time.time()
        self._write_status(state="training")
        result: dict[str, Any] = {"state": "finished"}
        try:
            while self.step < cfg.max_steps:
                t0 = time.perf_counter()
                for m in self.models.values():
                    m.train()
                logs: dict[str, float] = {}
                self.stepped = set()
                for micro in range(cfg.grad_accum):
                    self.boundary = micro == cfg.grad_accum - 1
                    batch = _to_device(next(data), self.device)
                    out = self.recipe.train_step(batch, ctx)
                    for k, v in out.items():
                        logs[k] = logs.get(k, 0.0) + float(v) / cfg.grad_accum
                    samples += cfg.batch_size * self.world
                self.scaler.update()
                for name, s in self.schedulers.items():
                    if name in self.stepped or name not in self.optimizers:
                        s.step()
                self.step += 1
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                dt = time.perf_counter() - t0
                step_times.append(dt)
                if prof is not None:
                    prof.step()
                    if self.step >= 6:
                        prof.stop()
                        self._write_profile(prof)
                        prof = None
                if any(not math.isfinite(v) for v in logs.values()):
                    raise FloatingPointError(f"non-finite loss at step {self.step}: {logs}")
                if self.step % cfg.log_every == 0 or self.step == 1:
                    el = time.time() - window_t0
                    sps = samples / el if el > 0 else 0.0
                    med = float(np.median(step_times[-cfg.log_every:]))
                    eta = med * (cfg.max_steps - self.step)
                    logs.update({"perf/samples_per_s": sps, "perf/step_s": med,
                                 **{f"grad_norm/{k}": v for k, v in self.last_grad_norm.items()},
                                 **{f"lr/{k}": o.param_groups[0]["lr"] for k, o in self.optimizers.items()}})
                    if self.device.type == "cuda":
                        logs["perf/vram_gib"] = torch.cuda.max_memory_allocated(self.device) / 2**30
                    self.logger.log(self.step, logs)
                    self._write_status(state="training", loss={k: round(v, 5) for k, v in logs.items()
                                                               if not k.startswith(("perf/", "lr/", "grad_norm/"))},
                                       samples_per_s=round(sps, 2), step_s=round(med, 4), eta_s=round(eta))
                    samples, window_t0 = 0, time.time()
                if val_batches and cfg.eval_every and self.step % cfg.eval_every == 0:
                    self.logger.log(self.step, {f"val/{k}": v for k, v in self._evaluate(val_batches, ctx).items()})
                if cfg.checkpoint_every and self.step % cfg.checkpoint_every == 0:
                    self.save()
                if self._handle_control():
                    result["state"] = "stopped"
                    break
            self.save()
        except torch.cuda.OutOfMemoryError as e:
            return self._fail_oom(e)
        except Exception as e:
            self._write_status(state="failed", error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-2000:])
            raise
        finally:
            self.logger.close() if hasattr(self, "logger") else None
        result.update(step=self.step, step_times=_summ(step_times), checkpoint=self.status.get("last_checkpoint"))
        self._write_status(state=result["state"])
        if self.world > 1:
            torch.distributed.destroy_process_group()
        return result

    def _val_batches(self) -> list[dict[str, Any]]:
        vs = getattr(self.recipe, "val_set", None)
        if vs is None or len(vs) == 0:
            return []
        n = min(len(vs), int(self.cfg.extra.get("val_items", 32)))
        items = [vs[i] for i in range(n)]
        bs = self.cfg.batch_size
        return [collate(items[i:i + bs]) for i in range(0, n, bs)]

    @torch.no_grad()
    def _evaluate(self, batches: list[dict[str, Any]], ctx: StepContext) -> dict[str, float]:
        for m in self.models.values():
            m.eval()
        return self.recipe.evaluate([_to_device(b, self.device) for b in batches], ctx)

    def _profiler(self):
        from torch.profiler import ProfilerActivity, profile, schedule

        acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if self.device.type == "cuda" else [])
        p = profile(activities=acts, schedule=schedule(wait=1, warmup=1, active=3), record_shapes=True,
                    profile_memory=True)
        p.start()
        return p

    def _write_profile(self, prof) -> None:
        d = self.out / "profile"
        d.mkdir(exist_ok=True)
        prof.export_chrome_trace(str(d / "trace.json"))
        sort = "self_cuda_time_total" if self.device.type == "cuda" else "self_cpu_time_total"
        (d / "top_ops.txt").write_text(prof.key_averages().table(sort_by=sort, row_limit=40))

    def _fail_oom(self, e: BaseException) -> dict[str, Any]:
        rep = oom_report(self.cfg, self.device, e)
        (self.out / "oom_report.json").write_text(json.dumps(rep, indent=2) + "\n")
        self._write_status(state="failed", error="out of memory", oom=rep)
        print(json.dumps(rep, indent=2), file=sys.stderr)
        return {"state": "failed", **rep}


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    nb = device.type == "cuda"
    return {k: v.to(device, non_blocking=nb) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def _summ(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {}
    a = np.asarray(xs[1:] or xs)  # first step includes warm-up / compilation
    return {"n": len(xs), "mean_s": float(a.mean()), "p50_s": float(np.median(a)), "p95_s": float(np.percentile(a, 95)),
            "first_s": float(xs[0])}


def estimate_cost(step_s: float, target_steps: int, measured_steps: int, gpu: str,
                  price_per_hour: float | None = None) -> dict[str, Any]:
    """Extrapolate wall time (and cost) of a full run from a measured short run."""
    hours = step_s * target_steps / 3600
    return {"label": "MEASURED BY AIVOICE (extrapolated)", "measured_steps": measured_steps, "step_s": step_s,
            "target_steps": target_steps, "hours": round(hours, 2), "gpu": gpu,
            "cost": round(hours * price_per_hour, 2) if price_per_hour else None,
            "caveat": "extrapolation from a short run; data loading, evaluation and checkpointing add time"}


__all__ = ["MetricsLogger", "Recipe", "StepContext", "TrainConfig", "Trainer", "estimate_cost", "make_adamw",
           "oom_report"]
