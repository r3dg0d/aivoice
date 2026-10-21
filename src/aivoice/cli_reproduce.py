"""`aivoice reproduce`: paper reproductions (list, info, status, prepare, train, resume, evaluate, export,
verify, paper-map, control, cost).

Nothing here downloads data or models, uploads audio or sends telemetry. W&B logging happens only with
``--wandb``. Training never runs inside a GUI process: ``--background`` starts a detached worker that
reports through ``<output>/status.json`` and obeys ``<output>/control.json`` (pause / resume / stop).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import click


def _echo_json(d: Any) -> None:
    click.echo(json.dumps(d, indent=2, default=str))


def _fail(e: BaseException) -> click.ClickException:
    return click.ClickException(f"{type(e).__name__}: {e}")


def _repro(name: str):
    from .research_reproductions import registry

    try:
        return registry.resolve(name), registry.get(name)
    except KeyError as e:
        raise click.ClickException(str(e)) from e


def _teachers(items: tuple[str, ...]) -> dict[str, str]:
    out = {}
    for it in items:
        k, sep, v = it.partition("=")
        if not sep:
            raise click.BadParameter(f"--teacher expects NAME=DIR, got {it!r}")
        out[k] = v
    return out


def _overrides(items: tuple[str, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for it in items:
        k, sep, v = it.partition("=")
        if not sep:
            raise click.BadParameter(f"--set expects KEY=VALUE, got {it!r}")
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def _config(key: str, r, config: str | None, stage: str | None, sets: tuple[str, ...]):
    """--config is a preset name (paper | tiny | aivoice ...) or a JSON file {"preset": ..., "overrides": {...}}."""
    from .research_reproductions import registry

    preset, ov = registry.default_preset(key), {}
    if config:
        p = Path(config)
        if p.suffix == ".json" and p.exists():
            d = json.loads(p.read_text())
            preset, ov = d.get("preset", preset), dict(d.get("overrides", {}))
        else:
            preset = config
    if preset not in getattr(r, "presets", (preset,)):
        raise click.ClickException(f"{key}: unknown preset {preset!r}; available: {', '.join(r.presets)}")
    ov.update(_overrides(sets))
    try:
        return r.config(preset, stage, ov), preset
    except (KeyError, TypeError, ValueError) as e:
        raise _fail(e) from e


def _cfg_dict(cfg: Any) -> dict[str, Any]:
    return asdict(cfg) if is_dataclass(cfg) else dict(cfg)


@click.group("reproduce")
def reproduce_group():
    """Paper reproductions: reconstructed architectures you can train (never official, never default)."""


@reproduce_group.command("list")
@click.option("--json", "as_json", is_flag=True)
def repro_list(as_json):
    from .research_reproductions import registry

    rows = [registry.summary(k) for k in registry.ENTRIES]
    if as_json:
        return _echo_json(rows)
    click.echo(f"{'name':<22} {'provenance':<19} {'source status':<37} {'arch':<7} model name")
    for r in rows:
        click.echo(f"{r['name']:<22} {r['provenance']:<19} {r['source_status']:<37} {r['arch_confidence']:<7} {r['model_name']}")
    click.echo("\nAll are EXPERIMENTAL and never chosen by --engine auto. Reproduced weights are never official.")


@reproduce_group.command("info")
@click.argument("name")
@click.option("--json", "as_json", is_flag=True)
def repro_info(name, as_json):
    from .research_reproductions import registry

    key, r = _repro(name)
    d = {**registry.summary(key), "info": r.info.as_dict(), "requirements": r.requirements(), "stages": r.stages}
    if as_json:
        return _echo_json(d)
    i = d["info"]
    click.echo(f"{key}: {i['title']}\n  provenance      {d['provenance']}\n  source status   {i['source_status']}"
               f"\n  architecture    {i['arch_confidence']} confidence\n  weights name    {d['model_name']}"
               f"\n  official code   {i['official_code'] or 'none'} ({i['official_license'] or 'n/a'})"
               f"\n  rate / hop      {i['sample_rate']} Hz / {i['hop_samples']}"
               f"\n  lookahead       {1000 * i['lookahead_samples'] / i['sample_rate']:.0f} ms by construction")
    for s, desc in r.stages.items():
        click.echo(f"  stage {s:<10} {desc}")
    for v, desc in i["variants"].items():
        click.echo(f"  variant {v:<8} {desc}")
    click.echo("  assumptions:")
    for a in i["assumptions"]:
        click.echo(f"    - [{a['confidence']}] {a['what']}: {a['choice']} ({a['why']})")
    click.echo("  requirements:")
    for q in d["requirements"]:
        mark = "ok" if q.get("ok") else ("?" if q.get("ok") is None else "MISSING")
        click.echo(f"    [{mark}] {q['name']}: {q['why']}" + (f" -> {q['hint']}" if q.get("hint") and not q.get("ok") else ""))
    if i["notes"]:
        click.echo(f"  notes: {i['notes']}")


@reproduce_group.command("status")
@click.argument("name", required=False)
@click.option("--run", "run_dir", type=click.Path(path_type=Path), default=None, help="training output directory")
@click.option("--json", "as_json", is_flag=True)
def repro_status(name, run_dir, as_json):
    """Not prepared / Architecture ready / Training / Checkpoint available / Ready / Failed / Blocked."""
    from .research_reproductions import registry

    keys = [registry.resolve(name)] if name else list(registry.ENTRIES)
    rows = []
    for k in keys:
        try:
            rows.append(registry.get(k).status(run_dir if name else None))
        except Exception as e:  # noqa: BLE001
            rows.append({"engine": k, "state": "Failed", "why": f"{type(e).__name__}: {e}"})
    if as_json:
        return _echo_json(rows if not name else rows[0])
    for s in rows:
        run = s.get("run") or {}
        extra = f" step {run.get('step')}/{run.get('max_steps')}" if run.get("step") is not None else ""
        click.echo(f"{s['engine']:<22} {s['state']:<22}{extra} {s.get('weights') or s.get('why') or ''}")


@reproduce_group.command("paper-map")
@click.argument("name")
def repro_paper_map(name):
    """Print SPEC.md: paper section / equation / figure -> module, decision, confidence, assumption."""
    _, r = _repro(name)
    click.echo(r.spec_text())


@reproduce_group.command("prepare")
@click.argument("name")
@click.option("--dataset", required=True, help="KIND:PATH (libritts|vctk|librispeech|custom) or a manifest .jsonl")
@click.option("--output", required=True, type=click.Path(path_type=Path))
@click.option("--stage", default=None)
@click.option("--config", default=None, help="preset name or JSON file")
@click.option("--set", "sets", multiple=True, help="config override KEY=VALUE (JSON value)")
@click.option("--teacher", "teachers", multiple=True, help="NAME=DIR of a local teacher snapshot")
@click.option("--device", default="cpu")
@click.option("--limit", type=int, default=None, help="only the first N utterances (smoke runs)")
@click.option("--no-features", is_flag=True, help="manifest only, no teacher caches")
def repro_prepare(name, dataset, output, stage, config, sets, teachers, device, limit, no_features):
    key, r = _repro(name)
    cfg, _ = _config(key, r, config, stage, sets)
    try:
        info = r.prepare(cfg, dataset, str(output), teacher_dirs=_teachers(teachers), device=device, limit=limit,
                         features=not no_features)
    except NotImplementedError as e:
        raise _fail(e) from e
    except Exception as e:
        raise _fail(e) from e
    _echo_json(info)


def _recipe_kwargs(key: str, dataset: str, teachers: dict[str, str], content_ckpt: str | None) -> tuple[Any, dict]:
    """Manifest + cache arguments from a prepared run directory (prepared.json), a manifest, or nothing."""
    from .research_reproductions.common.features import FeatureCache

    kw: dict[str, Any] = {}
    p = Path(dataset)
    if (p / "prepared.json").exists():
        prep = json.loads((p / "prepared.json").read_text())
        manifest = prep["manifest"]
        caches = {k: FeatureCache.open(v["cache"]) for k, v in prep.get("caches", {}).items()
                  if isinstance(v, dict) and v.get("cache")}
        if key == "zerovc-repro":
            kw.update(teacher_cache=caches.get("teacher"), timbre_cache=caches.get("timbre"),
                      kmeans_centroids=prep.get("kmeans"))
        elif key == "rtvc":
            kw.update(pitch_cache=caches.get("pitch"), ema_cache=caches.get("ema"))
        elif key.startswith("tvtsyn"):
            kw.update(code_cache=caches.get("codes"), spk_cache=caches.get("speaker"))
        elif key == "streamvoiceanon-plus":
            kw.update(caches=caches)
        kw["cache_dir"] = str(p / "audio-cache")
        kw["datasets"] = [prep.get("manifest_stats", {}).get("kind", "custom")]
    elif p.suffix == ".jsonl" and p.exists():
        manifest = str(p)
    else:
        raise click.ClickException("--dataset must be a prepared directory (from `reproduce prepare`) or a manifest .jsonl")
    if teachers:
        kw["teacher_dirs"] = teachers
    if content_ckpt:
        kw["content_checkpoint"] = content_ckpt
    return manifest, kw


def _train_args(f):
    opts = [
        click.option("--stage", default=None), click.option("--config", default=None),
        click.option("--set", "sets", multiple=True), click.option("--dataset", required=True),
        click.option("--output", required=True, type=click.Path(path_type=Path)),
        click.option("--resume", default=None, help="checkpoint path or `latest`"),
        click.option("--device", default="auto"), click.option("--precision", type=click.Choice(["bf16", "fp16", "fp32"]), default="bf16"),
        click.option("--batch-size", type=int, default=16), click.option("--grad-accum", type=int, default=1),
        click.option("--workers", type=int, default=4), click.option("--seed", type=int, default=1234),
        click.option("--max-steps", type=int, default=100_000), click.option("--checkpoint-every", type=int, default=5_000),
        click.option("--eval-every", type=int, default=2_500), click.option("--compile", "compile_", is_flag=True),
        click.option("--profile", is_flag=True), click.option("--grad-checkpoint", is_flag=True),
        click.option("--no-tf32", is_flag=True), click.option("--wandb", is_flag=True, help="opt-in W&B logging (metrics only)"),
        click.option("--teacher", "teachers", multiple=True), click.option("--content-checkpoint", default=None),
        click.option("--val-manifest", default=None),
        click.option("--overfit-steps", type=int, default=300, help="length of the mandatory tiny-overfit gate"),
        click.option("--background", is_flag=True, help="run as a detached background service"),
    ]
    for o in reversed(opts):
        f = o(f)
    return f


def _record_run(argv: list[str], output: Path) -> None:
    """Remember the latest run per engine (the research panel shows it)."""
    from .research_reproductions import registry

    try:
        i = argv.index("train") if "train" in argv else argv.index("resume")
        key = registry.resolve(argv[i + 1])
    except (ValueError, IndexError, KeyError):
        return
    f = registry.repro_home() / "runs.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    runs = json.loads(f.read_text()) if f.exists() else {}
    stage = argv[argv.index("--stage") + 1] if "--stage" in argv else None
    runs[key] = {"output": str(Path(output).resolve()), "stage": stage}
    f.write_text(json.dumps(runs, indent=2) + "\n")


def _spawn_background(output: Path, argv: list[str]) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    _record_run(argv, output)
    log = output / "train.log"
    args = [a for a in argv if a != "--background"]
    cmd = [sys.executable, "-m", "aivoice", *args]
    with open(log, "ab") as fh:
        p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True,
                             env={**os.environ, "PYTHONUNBUFFERED": "1"})
    info = {"pid": p.pid, "cmd": cmd, "log": str(log), "status": str(output / "status.json")}
    (output / "service.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def _run_training(name, stage, config, sets, dataset, output, resume, device, precision, batch_size, grad_accum, workers,
                  seed, max_steps, checkpoint_every, eval_every, compile_, profile, grad_checkpoint, no_tf32, wandb,
                  teachers, content_checkpoint, val_manifest, overfit_steps):
    from .research_reproductions.common.checkpoint import config_fingerprint
    from .research_reproductions.common.data import read_manifest
    from .research_reproductions.common.overfit import overfit_passed, run_overfit
    from .research_reproductions.common.trainer import TrainConfig, Trainer

    key, r = _repro(name)
    cfg, preset = _config(key, r, config, stage, sets)
    stage = stage or next(iter(r.stages))
    tdirs = _teachers(teachers)
    manifest, kw = _recipe_kwargs(key, dataset, tdirs, content_checkpoint)
    if val_manifest:
        kw["val_manifest"] = val_manifest
    try:
        recipe = r.recipe(stage, cfg, manifest, **kw)
    except NotImplementedError as e:
        raise _fail(e) from e
    model_cfg = recipe.model_config()
    gate_dir = output / "overfit"
    if not resume and not overfit_passed(gate_dir, model_cfg):
        # phase B: the same recipe on the first 2 utterances, fixed crops; must pass before any large run
        rows = read_manifest(manifest)[:2]
        click.echo(f"[{key}] running the mandatory tiny-overfit gate ({overfit_steps} steps on 2 utterances)...", err=True)
        rep = run_overfit(lambda: r.recipe(stage, _config(key, r, config, stage, sets)[0], rows,
                                            **{**kw, "fixed_crops": True, "val_manifest": None}),
                          gate_dir, steps=overfit_steps, device=device, precision="fp32")
        click.echo(json.dumps({k: rep[k] for k in ("passed", "loss_key", "initial", "final", "ratio")}), err=True)
        if not rep["passed"]:
            raise click.ClickException(f"tiny-overfit gate FAILED ({rep['loss_key']} {rep['initial']:.4g} -> "
                                       f"{rep['final']:.4g}); not starting full training. See {gate_dir}/overfit.json")
    output.mkdir(parents=True, exist_ok=True)
    argv = [a for a in sys.argv[1:] if a != "--background"]
    if "--resume" not in argv:  # remembered for `reproduce resume`
        (output / "train_args.json").write_text(json.dumps(argv) + "\n")
    _record_run(argv, output)
    tc = TrainConfig(output=str(output), device=device, precision=precision, batch_size=batch_size, grad_accum=grad_accum,
                     workers=workers, seed=seed, max_steps=max_steps, checkpoint_every=checkpoint_every,
                     eval_every=eval_every, compile=compile_, profile=profile, tf32=not no_tf32,
                     grad_checkpoint=grad_checkpoint, wandb=wandb, resume=resume)
    res = Trainer(recipe, tc).fit()
    res["config_fingerprint"] = config_fingerprint(model_cfg)
    _echo_json({k: v for k, v in res.items() if k != "curve"})
    if res.get("state") == "failed":
        raise SystemExit(3)


@reproduce_group.command("train", context_settings={"ignore_unknown_options": False})
@click.argument("name")
@_train_args
@click.pass_context
def repro_train(ctx, name, background, **kw):
    """Train one stage (mandatory tiny-overfit gate first). Options are all honoured; none is silently changed."""
    if background:
        info = _spawn_background(kw["output"], sys.argv[1:])
        return _echo_json({"background": True, **info})
    _run_training(name, **kw)


@reproduce_group.command("resume")
@click.argument("name")
@click.option("--output", required=True, type=click.Path(path_type=Path))
@click.option("--checkpoint", default="latest")
@click.option("--background", is_flag=True)
def repro_resume(name, output, checkpoint, background):
    """Resume exactly (weights, optimisers, schedulers, scaler, step, RNG, sampler) with the original options."""
    rc = output / "run_config.json"
    svc = output / "service.json"
    if not rc.exists():
        raise click.ClickException(f"{output}: no run_config.json (not a training run)")
    if not svc.exists() and not (output / "train_args.json").exists():
        raise click.ClickException(f"{output}: original command unknown (no service.json / train_args.json)")
    args = json.loads((output / "train_args.json").read_text()) if (output / "train_args.json").exists() else \
        json.loads(svc.read_text())["cmd"][3:]  # [python, -m, aivoice, ...]
    args = [a for a in args if a != "--background"]
    if "--resume" in args:
        i = args.index("--resume")
        del args[i:i + 2]
    args += ["--resume", checkpoint]
    if background:
        return _echo_json({"background": True, **_spawn_background(output, args)})
    os.execv(sys.executable, [sys.executable, "-m", "aivoice", *args])


@reproduce_group.command("control")
@click.argument("action", type=click.Choice(["pause", "resume", "stop"]))
@click.option("--output", required=True, type=click.Path(path_type=Path))
def repro_control(action, output):
    """Pause / resume / stop a running training service (checkpoint is saved on pause and stop)."""
    p = output / "control.json"
    p.write_text(json.dumps({"command": "train" if action == "resume" else action}) + "\n")
    click.echo(f"{action} requested ({p})")


@reproduce_group.command("cost")
@click.option("--output", required=True, type=click.Path(path_type=Path), help="run that measured some steps")
@click.option("--target-steps", type=int, required=True)
@click.option("--price-per-hour", type=float, default=None)
def repro_cost(output, target_steps, price_per_hour):
    """Extrapolate full-training time (and cost) from the measured steps of a short run."""
    import numpy as np

    from .research_reproductions.common.trainer import estimate_cost

    rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines() if line.strip()]
    st = [r["perf/step_s"] for r in rows if "perf/step_s" in r][1:]
    if not st:
        raise click.ClickException("no step timings in metrics.jsonl")
    status = json.loads((output / "status.json").read_text()) if (output / "status.json").exists() else {}
    _echo_json(estimate_cost(float(np.median(st)), target_steps, int(rows[-1]["step"]), status.get("gpu", "cpu"),
                             price_per_hour))


@reproduce_group.command("evaluate")
@click.argument("name")
@click.option("--checkpoint", default=None)
@click.option("--untrained", "untrained_preset", default=None, help="evaluate a RANDOM-INIT network of this preset "
              "(streaming/compute tests only; never conversion quality)")
@click.option("--source", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--target", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--long-minutes", type=float, default=0.0, help="also run a long-session test (30 recommended)")
@click.option("--device", default="cpu")
@click.option("--output", type=click.Path(path_type=Path), default=None)
@click.option("--teacher", "teachers", multiple=True)
def repro_evaluate(name, checkpoint, untrained_preset, source, target, long_minutes, device, output, teachers):
    """Streaming harness + compute (+ conversion metrics with --source/--target). All MEASURED BY AIVOICE."""
    import numpy as np

    from .research_reproductions.common.evaluate import (
        compute_report,
        conversion_metrics,
        streaming_report,
    )

    key, r = _repro(name)
    if bool(checkpoint) == bool(untrained_preset):
        raise click.ClickException("give exactly one of --checkpoint or --untrained PRESET")
    if untrained_preset:
        model, tgt = _untrained(key, r, untrained_preset, device)
        rep: dict[str, Any] = {"engine": key, "weights": "RANDOM_INIT (architecture test only)"}
    else:
        try:
            model = r.load_streaming(checkpoint, device=device, teacher_dirs=_teachers(teachers))
        except Exception as e:
            raise _fail(e) from e
        rep = {"engine": key, "checkpoint": checkpoint, "weights": model.meta.get("weights_label"),
               "model_name": model.meta.get("model_name")}
        if target is None:
            raise click.ClickException("--target reference audio is needed to evaluate a trained checkpoint")
        import librosa

        ref, _ = librosa.load(str(target), sr=model.sample_rate)
        tgt = model.enroll(ref)
    rep["streaming"] = streaming_report(model, tgt, long_minutes)
    rep["compute"] = compute_report(model, tgt)
    if source is not None and target is not None and not untrained_preset:
        import librosa

        sr = model.sample_rate
        src, _ = librosa.load(str(source), sr=sr)
        ref, _ = librosa.load(str(target), sr=sr)
        y = np.asarray(model.offline(src, tgt))
        rep["conversion"] = conversion_metrics(y, src, ref, sr)
    if output:
        Path(output).write_text(json.dumps(rep, indent=2, default=str) + "\n")
    _echo_json(rep)


def _untrained(key: str, r, preset: str, device: str):
    import torch

    cfg = r.config(preset)
    if key == "zerovc-repro":
        from .research_reproductions.zerovc.streaming.inference import ZeroVCStreaming

        return ZeroVCStreaming.untrained(cfg, device), torch.randn(1, cfg.timbre.embed_dim)
    if key == "rtvc":
        from .research_reproductions.rtvc.streaming.inference import (
            RTVCStreaming,
            Target,
        )

        return RTVCStreaming.untrained(cfg, device), Target(torch.randn(1, cfg.speaker.spkr_emb_dim), 1.0)
    if key.startswith("tvtsyn"):
        from .research_reproductions.common.harness import test_signal
        from .research_reproductions.tvtsyn.streaming.inference import TVTSynStreaming

        return (TVTSynStreaming.untrained(cfg, device, codebook_init_audio=test_signal(16000, 4.0, 9)),
                torch.randn(1, cfg.timbre.g_dim))
    raise click.ClickException(f"{key}: no aivoice streaming graph (runs through the official engine)")


@reproduce_group.command("export")
@click.argument("name")
@click.option("--checkpoint", default=None, help="vc/synth/finetune checkpoint")
@click.option("--pitch", default=None, help="RT-VC pitch-stage checkpoint")
@click.option("--ema", default=None, help="RT-VC ema-stage checkpoint")
@click.option("--vocoder", default=None, help="RT-VC vocoder-stage checkpoint")
@click.option("--output", required=True, type=click.Path(path_type=Path))
@click.option("--install", is_flag=True, help="also copy it to where the live engine looks for it")
def repro_export(name, checkpoint, pitch, ema, vocoder, output, install):
    """Write an inference checkpoint (training-only parts removed). Never uploads anything."""
    import shutil

    from .research_reproductions import registry
    from .research_reproductions.common.checkpoint import (
        CheckpointMeta,
        load_checkpoint,
        read_meta,
        save_checkpoint,
    )

    key, r = _repro(name)
    try:
        if key == "rtvc":
            if not (pitch and ema and vocoder):
                raise click.ClickException("rtvc export needs --pitch, --ema and --vocoder stage checkpoints")
            from .research_reproductions.rtvc.train.checkpoint import export

            path = export(pitch, ema, vocoder, output)
        else:
            if not checkpoint:
                raise click.ClickException("--checkpoint is required")
            blob = load_checkpoint(checkpoint)
            meta = blob["meta"]
            keep = {"zerovc-repro": ("content", "timbre", "decoder"), "tvtsyn-repro": ("content", "timbre", "decoder"),
                    "tvtsyn-aivoice": ("content", "timbre", "decoder"), "streamvoiceanon-plus": ("plus",)}[key]
            import torch

            class _Holder(torch.nn.Module):
                def __init__(self, sd):
                    super().__init__()
                    self._sd = sd

                def state_dict(self, *a, **k):
                    return self._sd

                def parameters(self, recurse=True):
                    return iter(())

            models = {}
            for k in keep:
                sd = blob["models"][k]
                if key == "streamvoiceanon-plus":
                    sd = {n: v for n, v in sd.items() if not n.startswith("emotion_head.")}  # training-only head
                models[k] = _Holder(sd)
            fields = {f: meta[f] for f in CheckpointMeta.__dataclass_fields__ if f in meta and f not in (
                "format", "config_fingerprint", "created", "environment", "param_shapes", "trainable_params", "weights_label")}
            m = CheckpointMeta(**fields)
            m.notes = (m.notes + " | exported for inference (training-only modules removed)").strip()
            path = save_checkpoint(output, m, models)
    except click.ClickException:
        raise
    except Exception as e:
        raise _fail(e) from e
    out = {"exported": str(path), "meta": {k: v for k, v in read_meta(path).items() if k in (
        "engine", "model_name", "weights_status", "weights_label", "step", "redistributable")}}
    if install:
        dest = registry.repro_home() / key / "model.pt"
        dest.parent.mkdir(parents=True, exist_ok=True)
        for suf in ("", ".sha256"):
            shutil.copy2(str(path) + suf, str(dest) + suf)
        shutil.copy2(path.with_suffix(".json"), dest.with_suffix(".json"))
        out["installed"] = str(dest)
    out["note"] = "weights are never published automatically; check the data and component licenses before sharing"
    _echo_json(out)


@reproduce_group.command("verify")
@click.argument("name")
@click.option("--official-checkout", type=click.Path(exists=True, path_type=Path), default=None,
              help="your checkout of the official code: compare architectures parameter by parameter")
@click.option("--overfit", "overfit_steps", type=int, default=0, help="also run the tiny-overfit gate (steps)")
@click.option("--output", type=click.Path(path_type=Path), default=None)
def repro_verify(name, official_checkout, overfit_steps, output):
    """Phase A (architecture: shapes, params, streaming exactness on a tiny random net) [+ parity, + phase B]."""
    import tempfile

    from .research_reproductions.common import harness
    from .research_reproductions.common.blocks import param_count
    from .research_reproductions.common.overfit import run_overfit, synthetic_corpus

    key, r = _repro(name)
    rep: dict[str, Any] = {"engine": key, "phase_A": {}}
    if key in ("zerovc-repro", "rtvc", "tvtsyn-repro", "tvtsyn-aivoice"):
        model, tgt = _untrained(key, r, "tiny", "cpu")
        rep["phase_A"]["streaming_tiny_random"] = {t["test"]: t["passed"] for t in harness.run_all(model, tgt)["tests"]}
        full, _ = _untrained(key, r, r.presets[0] if key != "tvtsyn-aivoice" else "aivoice", "cpu")
        rep["phase_A"]["params_paper_config_M"] = {n: round(param_count(m) / 1e6, 3) for n, m in full.net.named_children()}
    if official_checkout:
        if key == "rtvc":
            from .research_reproductions.rtvc.eval.official_parity import check
        elif key.startswith("tvtsyn"):
            from .research_reproductions.tvtsyn.eval.official_parity import check
        else:
            raise click.ClickException(f"{key}: no official code to compare against")
        rep["official_parity"] = check(official_checkout)
    if overfit_steps:
        with tempfile.TemporaryDirectory() as d:
            man = synthetic_corpus(Path(d) / "corpus", 2, speakers=1)
            stage = next(iter(r.stages))
            o = run_overfit(lambda: r.recipe(stage, r.config("tiny", stage), str(man), fixed_crops=True), Path(d) / "ov",
                            steps=overfit_steps)
            rep["phase_B_tiny_overfit"] = {k: o[k] for k in ("passed", "loss_key", "initial", "final", "ratio", "steps")}
    if output:
        Path(output).write_text(json.dumps(rep, indent=2) + "\n")
    _echo_json(rep)


@reproduce_group.command("panel", hidden=True)
def repro_panel():
    """One JSON document for the QuickShell research panel: every reproduction + its latest run."""
    import time

    from .research_reproductions import registry

    f = registry.repro_home() / "runs.json"
    runs = json.loads(f.read_text()) if f.exists() else {}
    rows = []
    for k in registry.ENTRIES:
        run = runs.get(k)
        try:
            st = registry.get(k).status(run["output"] if run else None)
        except Exception as e:  # noqa: BLE001
            st = {"engine": k, "state": "Failed", "why": f"{type(e).__name__}: {e}"}
        live = st.get("run") or {}
        if live.get("state") in ("training", "paused") and live.get("pid"):
            try:
                os.kill(int(live["pid"]), 0)
            except OSError:
                st["state"], st["why"] = "Failed", "training process is gone (crashed or killed); see train.log"
        if live.get("updated") and time.time() - float(live["updated"]) > 600 and st.get("state") == "Training":
            st["stale"] = True
        rows.append({**registry.summary(k), **st, "output": run["output"] if run else None,
                     "stage": run.get("stage") if run else None})
    _echo_json(rows)


def register(main: click.Group) -> None:
    main.add_command(reproduce_group)
