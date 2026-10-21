"""CLI: live voice conversion engines (vc), identity-preserving modulation (modulate), VC benchmark.

Registered onto the main group by :func:`register`. Direct VC, modulation and speech-to-speech
(`aivoice experimental`) stay separate commands because they are separate systems; all three
play into the same ``aivoice-virtual-mic``.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

import click

VC_ENGINE_CHOICES = ["auto", "xvc", "rvc", "meanvc2", "dsp", "passthrough", "stylestream", "streamvoiceanon"]
QUALITIES = ["lowest-latency", "balanced", "best-quality"]


def _fail(e: BaseException) -> click.ClickException:
    return click.ClickException(str(e))


def _json(d: Any) -> None:
    click.echo(json.dumps(d, indent=2, default=str))


def register(main: click.Group) -> None:
    main.add_command(vc_group)
    main.add_command(modulate_group)


# =============================================================================
# shared helpers
# =============================================================================


def resolve_target(voice: str | None, reference: Path | None = None):
    """A VoiceTarget from the voice library (reference + RVC .pth/.index + rt.json) or a bare reference WAV."""
    from .vc.base import VoiceTarget

    if reference is not None:
        return VoiceTarget(Path(reference).stem, reference=Path(reference))
    if not voice:
        return None
    from .pipeline import voice_rt_settings, voice_rvc_files
    from .voices import load_voice

    try:
        v = load_voice(voice)
    except FileNotFoundError as e:
        raise click.ClickException(str(e)) from e
    pth, idx = voice_rvc_files(v)
    ref = Path(v.reference) if v.reference and Path(v.reference).is_file() and v.status == "ready" else None
    return VoiceTarget(v.display_name, reference=ref, model_file=pth, index_file=idx,
                       extra={"rt": voice_rt_settings(v), "voice_id": v.id})


def build_engine(name: str, *, target=None, priority: str = "balanced", quality: str = "balanced", modulation=None,
                 echo: bool = True):
    """Engine instance for *name* (``auto`` routes on local measurements). Returns (engine, routing note)."""
    from .vc.base import StreamOptions
    from .vc.engines import DspEngine, get_vc_engine

    note = ""
    if name == "auto":
        from .vc.router import route_vc

        dec = route_vc(priority, target=target)
        if dec.engine is None:
            reasons = "; ".join(f"{k}: {v}" for k, v in dec.rejected.items())
            raise click.ClickException(f"no voice-conversion engine available ({reasons}). "
                                       "For your own voice, `aivoice modulate` needs no model.")
        name = dec.engine
        note = f"auto -> {name} ({' '.join(dec.reasons)})"
    if name == "dsp":
        eng = DspEngine(modulation)
    else:
        eng = get_vc_engine(name)
    eng.options = StreamOptions(quality=quality)
    st = eng.status()
    if not st.state.runnable:
        raise click.ClickException(f"{name}: {st.state.value}: {st.detail}")
    return eng, note


def _input_device(explicit: str | None) -> str | None:
    from .pipeline import _input_source

    return explicit or _input_source()


def run_live_unified(*, engine, cfg, target=None, input_file: Path | None = None, output: str = "virtual",
                     duration: float | None = None, quiet: bool = False) -> int:
    """Foreground live session on the shared pipeline (Ctrl-C / `aivoice vc stop` to end)."""
    from .audio.capture import ArrayCapture
    from .vc.engines import get_vc_engine
    from .vc.live import PIPE_SR, LivePipeline, LiveRunner, pid_path
    from .virtualmic import ensure_virtual_mic, feedback_risk

    risk = feedback_risk(cfg.input_device)
    if risk:
        raise click.ClickException(risk)
    if output == "virtual":
        info = ensure_virtual_mic("aivoice")
        cfg.output_sink = info.sink or "aivoice_sink"
        click.echo(f"Output    {info.hint}")
    elif output == "default":
        cfg.output_sink = None
    capture = ArrayCapture.from_file(input_file, target_rate=PIPE_SR, block_ms=int(cfg.block_ms)) if input_file else None

    def on_event(kind, data):
        if not quiet and kind == "fallback":
            click.echo(f"\n! {data['from']} failed -> {data['to']} ({data['reason'][:160]})", err=True)

    primary = cfg.engine
    pipe = LivePipeline(cfg, engine, target=target, on_event=on_event)
    driver = "clock" if output == "null" else "auto"
    runner = LiveRunner(pipe, capture=capture, driver_kind=driver,
                        primary_factory=(lambda: get_vc_engine(primary)) if primary not in ("dsp", "passthrough") else None)
    stop = {"flag": False}

    def handler(signum, frame):  # noqa: ARG001
        if signum == signal.SIGUSR1:
            pipe.bypass = not pipe.bypass
        elif signum == signal.SIGUSR2:
            pipe.muted = not pipe.muted
        else:
            stop["flag"] = True
            runner._stop.set()

    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGUSR1, signal.SIGUSR2):
        signal.signal(s, handler)
    pid_path().parent.mkdir(parents=True, exist_ok=True)
    pid_path().write_text(str(os.getpid()))
    try:
        click.echo("loading engine...")
        t0 = time.perf_counter()
        runner.start()
        caps = engine.capabilities()
        click.echo(f"ready in {time.perf_counter() - t0:.1f}s  {caps.display_name}  algorithmic latency "
                   f"{caps.algorithmic_latency_ms if caps.algorithmic_latency_ms is not None else '?'} ms "
                   f"(+{pipe._pipeline_extra_ms():.1f} ms resampling/modulation)")
        click.echo("live (Ctrl-C to stop; `aivoice vc control bypass|mute`)")
        end = time.monotonic() + duration if duration else None
        while not runner._stop.is_set():
            if end and time.monotonic() >= end:
                break
            if not quiet and sys.stderr.isatty():
                s = pipe.snapshot()
                w = s["processing"].get("engine_ms", {})
                click.echo(f"\r\033[K  {s['engine']}  in {s['level_in']:.2f} out {s['level_out']:.2f}  engine p95 "
                           f"{w.get('p95', '-')} ms  rtf {s['rtf']}  underruns {runner.player.underruns}"
                           f"{'  BYPASS' if pipe.bypass else ''}{'  MUTED' if pipe.muted else ''}", nl=False, err=True)
            runner._stop.wait(0.5)
    except click.ClickException:
        raise
    except Exception as e:  # noqa: BLE001
        raise _fail(e) from e
    finally:
        runner.stop()
        pid_path().unlink(missing_ok=True)
        # the virtual mic is left in place for the next session, so apps never need re-pointing
        s = pipe.snapshot()
        click.echo(f"\nsession: {s['audio_s']} s audio, engine {s['engine']}, rtf {s['rtf']}, fallbacks "
                   f"{len(s['fallbacks'])}, underruns {runner.player.underruns}, dropped {getattr(runner.capture, 'dropped', 0)}")
    return 0


def _pid_alive() -> int | None:
    from .vc.live import pid_path

    try:
        pid = int(pid_path().read_text().strip())
        os.kill(pid, 0)
        return pid
    except (OSError, ValueError):
        return None


# =============================================================================
# vc
# =============================================================================


@click.group("vc")
def vc_group():
    """Live voice-conversion engines: status, control, routing (direct VC; see `modulate` for your own voice)."""


@vc_group.command("engines")
@click.option("--json", "as_json", is_flag=True)
def vc_engines(as_json):
    """Every live VC / modulation engine with its real status on this machine."""
    from .vc.engines import all_vc_status

    st = all_vc_status()
    if as_json:
        _json([s.to_dict() for s in st])
        return
    click.echo(f"{'ENGINE':<16} {'STATE':<20} {'TASK':<17} {'LATENCY':<9} DETAIL")
    for s in st:
        lat = f"{s.caps.algorithmic_latency_ms:.0f} ms" if s.caps.algorithmic_latency_ms is not None else "-"
        click.echo(f"{s.name:<16} {s.state.value:<20} {s.caps.task:<17} {lat:<9} {s.detail[:110]}")
    click.echo("\nLatency = by construction (chunk + lookahead + smoothing); `aivoice benchmark vc` measures it.")


@vc_group.command("route")
@click.option("--priority", default="balanced", show_default=True)
@click.option("--voice", default=None)
@click.option("--json", "as_json", is_flag=True)
def vc_route(priority, voice, as_json):
    """Which engine `--engine auto` would pick, and why (local measurements only)."""
    from .vc.router import route_vc

    try:
        d = route_vc(priority, target=resolve_target(voice))
    except ValueError as e:
        raise _fail(e) from e
    if as_json:
        _json(d.to_dict())
        return
    click.echo(f"engine: {d.engine or '(none)'}  measured: {d.measured}")
    for r in d.reasons:
        click.echo(f"  {r}")
    for k, v in d.rejected.items():
        click.echo(f"  - {k}: {v}")


@vc_group.command("status")
@click.option("--json", "as_json", is_flag=True)
@click.option("--watch", is_flag=True)
def vc_status(as_json, watch):
    """Live session state (engine, latency, RTF, levels, VRAM/GPU, fallbacks)."""
    from .vc.live import read_status

    while True:
        s = read_status()
        if as_json:
            _json(s or {"running": False})
            return
        if watch:
            click.clear()
        if not s or not s.get("running"):
            click.echo("no live session running")
        else:
            p = s.get("processing", {})
            click.echo(f"mode {s['mode']}  engine {s['engine']} ({s['engine_display']})  voice {s.get('voice') or '-'}  pid {s['pid']}")
            click.echo(f"input {s.get('input') or 'default'}  output {s.get('output')}  {s['sample_rate']} Hz (engine {s['engine_sample_rate']} Hz)")
            click.echo(f"latency: algorithmic {s.get('algorithmic_latency_ms')} ms + pipeline {s.get('pipeline_extra_latency_ms')} ms; "
                       f"engine p50/p95/p99 {p.get('engine_ms', {}).get('p50')}/{p.get('engine_ms', {}).get('p95')}/{p.get('engine_ms', {}).get('p99')} ms; rtf {s.get('rtf')}")
            click.echo(f"underruns {s.get('underruns')}  dropped {s.get('dropped_blocks')}  fallbacks {len(s.get('fallbacks', []))}  "
                       f"bypass {s.get('bypass')}  muted {s.get('muted')}  gpu {s.get('gpu') or '-'}  worker {s.get('worker') or '-'}")
            for e in s.get("errors", []):
                click.echo(f"  ! {e}")
        if not watch:
            return
        time.sleep(1)


@vc_group.command("stop")
def vc_stop():
    pid = _pid_alive()
    if not pid:
        click.echo("no live session running")
        return
    os.kill(pid, signal.SIGTERM)
    for _ in range(50):
        if not _pid_alive():
            click.echo(f"stopped (pid {pid})")
            return
        time.sleep(0.1)
    raise click.ClickException(f"pid {pid} did not stop within 5 s")


@vc_group.command("control")
@click.argument("action", type=click.Choice(["bypass", "unbypass", "toggle-bypass", "mute", "unmute", "toggle-mute"]))
def vc_control(action):
    """A/B bypass and mute for the running session (bind to hotkeys)."""
    from .vc.live import read_status, send_control

    s = read_status() or {}
    if not s.get("running"):
        raise click.ClickException("no live session running")
    if action.startswith("toggle"):
        key = "bypass" if action.endswith("bypass") else "mute"
        send_control(**{key: not s.get("bypass" if key == "bypass" else "muted")})
    else:
        send_control(**{action.replace("un", ""): not action.startswith("un")})
    click.echo(f"{action} sent")


@vc_group.command("set")
@click.option("--preset", default=None)
@click.option("--strength", type=float, default=None)
@click.option("--input-gain", type=float, default=None, help="dB")
@click.option("--output-gain", type=float, default=None, help="dB")
@click.option("--param", "params", multiple=True, help="override NAME=VALUE (modulation parameter)")
def vc_set(preset, strength, input_gain, output_gain, params):
    """Change modulation / gains of the running session without restarting the engine."""
    from .vc.live import send_control

    over = {}
    for p in params:
        k, _, v = p.partition("=")
        over[k] = float(v)
    send_control(preset=preset, strength=_strength(strength) if strength is not None else None, input_gain_db=input_gain,
                 output_gain_db=output_gain, overrides=over or None)
    click.echo("sent")


# =============================================================================
# modulate
# =============================================================================


def _strength(v: float | None) -> float | None:
    if v is None:
        return None
    return v / 100.0 if v > 1.5 else v


class _ModulateGroup(click.Group):
    """`aivoice modulate masculine ...` runs a preset; `aivoice modulate preset ...` manages them."""

    def resolve_command(self, ctx, args):
        if args and args[0] not in self.commands and not args[0].startswith("-"):
            return "run", self.commands["run"], args
        return super().resolve_command(ctx, args)


@click.group("modulate", cls=_ModulateGroup)
def modulate_group():
    """Identity-preserving modulation of YOUR OWN live voice (no target voice, no model download).

    \b
      aivoice modulate masculine --strength 65
      aivoice modulate deep --strength 40 --formants -6
      aivoice modulate masculine --file in.wav -o out.wav --identity 0.8
      aivoice modulate preset list | use NAME | save NAME ... | show NAME | delete NAME
    """


def _mod_flags(f):
    from .modulation.presets import ADVANCED_FLAGS

    for flag, (_, unit) in reversed(list(ADVANCED_FLAGS.items())):
        f = click.option(f"--{flag}", type=float, default=None, help=unit)(f)
    return f


@modulate_group.command("run", hidden=True)
@click.argument("preset")
@click.option("--strength", type=float, default=None, help="0-1 or 0-100 (default: the preset's)")
@_mod_flags
@click.option("--file", "in_file", type=click.Path(path_type=Path, exists=True, dir_okay=False), default=None,
              help="Process a WAV instead of the microphone")
@click.option("-o", "--output", "out_file", type=click.Path(path_type=Path), default=None)
@click.option("--identity", type=float, default=None,
              help="file mode: minimum GE2E similarity to the input; strength is reduced until it holds")
@click.option("--input", "input_device", default=None, help="Microphone (PipeWire source name)")
@click.option("--sink", type=click.Choice(["virtual", "default", "null"]), default="virtual", show_default=True)
@click.option("--test-input", type=click.Path(path_type=Path, exists=True, dir_okay=False), default=None,
              help="Live mode with a WAV paced as the microphone")
@click.option("--duration", type=float, default=None)
@click.option("--denoise", type=float, default=None, help="0-1 spectral-floor noise suppression")
@click.option("--quiet", is_flag=True)
def modulate_run(preset, strength, pitch, formants, resonance, body, brightness, roughness, breathiness, intonation, in_file,
                 out_file, identity, input_device, sink, test_input, duration, denoise, quiet):
    from .modulation.presets import get_preset, overrides_from_flags

    try:
        p = get_preset(preset)
        over = overrides_from_flags(pitch=pitch, formants=formants, resonance=resonance, body=body, brightness=brightness,
                                    roughness=roughness, breathiness=breathiness, intonation=intonation)
        if denoise is not None:
            over["denoise"] = denoise
        s = _strength(strength)
        params = p.params(s, over)
    except (KeyError, ValueError) as e:
        raise _fail(e) from e
    if in_file:
        _modulate_file(p, s, over, in_file, out_file, identity)
        return
    from .vc.engines import DspEngine
    from .vc.live import LiveConfig

    eng = DspEngine(params, window_ms=p.window_ms)
    cfg = LiveConfig(mode="modulation", engine="dsp", voice=None, input_device=_input_device(input_device),
                     post_preset=p.name, post_strength=s if s is not None else p.default_strength, post_overrides=over,
                     fallback=["passthrough"])
    click.echo("Voice modulation of your own voice (no target voice, nothing leaves this machine)")
    click.echo(f"Preset    {p.name} @ {cfg.post_strength:.2f}  ({p.description})")
    click.echo(f"Input     {cfg.input_device or '(default source)'}")
    run_live_unified(engine=eng, cfg=cfg, input_file=test_input, output=sink, duration=duration, quiet=quiet)


def _modulate_file(p, s, over, in_file: Path, out_file: Path | None, identity: float | None) -> None:
    from .audio_io import read_wav
    from .evaluation.metrics import write_labeled_wav
    from .evaluation.voice_metrics import voice_profile
    from .modulation.dsp import SpectralModulator
    from .research.speaker_identity import get_encoder

    x, sr = read_wav(in_file)
    strength = p.default_strength if s is None else s
    enc = get_encoder()
    if identity is not None and enc is None:
        raise click.ClickException("--identity needs the speaker encoder: pip install 'aivoice[research]'")

    def render(st: float):
        return SpectralModulator(sr, p.params(st, over), window_ms=p.window_ms, hop_ms=p.window_ms / 4).process_all(x)

    y = render(strength)
    sim = enc.similarity(y, sr, x, sr) if enc is not None else None
    tried = [(strength, sim)]
    if identity is not None and sim is not None and sim < identity:
        lo, hi = 0.0, strength
        for _ in range(6):  # bisection on strength
            mid = (lo + hi) / 2
            ym = render(mid)
            sm = enc.similarity(ym, sr, x, sr)
            tried.append((mid, sm))
            if sm >= identity:
                lo, y, sim, strength = mid, ym, sm, mid
            else:
                hi = mid
        if sim is not None and sim < identity:
            y, sim, strength = x.copy(), 1.0, 0.0
    out = out_file or in_file.with_name(in_file.stem + f"_{p.name}.wav")
    write_labeled_wav(out, y, sr, comment=f"aivoice modulation {p.name} strength {strength:.2f} (own-voice attribute change)")
    a, b = voice_profile(x, sr), voice_profile(y, sr)
    click.echo(f"wrote {out}  (preset {p.name}, strength {strength:.2f})")
    for k in ("f0_median_hz", "f0_range_st", "formants_hz", "apparent_vtl_cm", "spectral_centroid_hz", "spectral_tilt_db_per_oct"):
        click.echo(f"  {k:<26} {a.get(k)} -> {b.get(k)}")
    if sim is not None:
        click.echo(f"  GE2E similarity to input   {sim:.3f}  (similarity, not identity proof)"
                   + (f"; identity floor {identity} reached at strength {strength:.2f}" if identity is not None else ""))


@modulate_group.group("preset")
def preset_group():
    """Built-in and saved modulation presets."""


@preset_group.command("list")
@click.option("--json", "as_json", is_flag=True)
def preset_list(as_json):
    from .modulation.presets import all_presets

    ps = all_presets()
    if as_json:
        _json([p.to_dict() for p in ps.values()])
        return
    from . import settings

    active = settings.get("modulation.preset")
    for p in ps.values():
        mark = "*" if p.name == active else " "
        click.echo(f"{mark} {p.name:<18} {p.category:<20} {p.default_strength:<5.2f} {p.description}")


@preset_group.command("show")
@click.argument("name")
def preset_show(name):
    from .modulation.presets import get_preset

    try:
        p = get_preset(name)
    except KeyError as e:
        raise _fail(e) from e
    _json({**p.to_dict(), "params_at_default_strength": p.params().to_dict()})


@preset_group.command("use")
@click.argument("name")
@click.option("--strength", type=float, default=None)
def preset_use(name, strength):
    """Make NAME the default preset (widget, `aivoice live --modulate`); applies live to a running session."""
    from . import settings
    from .modulation.presets import get_preset
    from .vc.live import read_status, send_control

    try:
        p = get_preset(name)
    except KeyError as e:
        raise _fail(e) from e
    s = _strength(strength) if strength is not None else p.default_strength
    settings.set_value("modulation.preset", p.name)
    settings.set_value("modulation.strength", s)
    if (read_status() or {}).get("running"):
        send_control(preset=p.name, strength=s)
        click.echo("applied to the running session")
    click.echo(f"default modulation: {p.name} @ {s:.2f}")


@preset_group.command("save")
@click.argument("name")
@click.option("--from", "base", default="natural", show_default=True, help="Start from this preset")
@click.option("--strength", type=float, default=None)
@_mod_flags
@click.option("--description", default="")
def preset_save(name, base, strength, pitch, formants, resonance, body, brightness, roughness, breathiness, intonation, description):
    """Save a preset: `aivoice modulate preset save my-deep --from deep --strength 50 --formants -8`."""
    from .modulation.presets import get_preset, overrides_from_flags, save_user_preset

    try:
        b = get_preset(base)
        over = overrides_from_flags(pitch=pitch, formants=formants, resonance=resonance, body=body, brightness=brightness,
                                    roughness=roughness, breathiness=breathiness, intonation=intonation)
        p = save_user_preset(name, b.params(_strength(strength), over), description=description or f"from {base}",
                             window_ms=b.window_ms)
    except (KeyError, ValueError) as e:
        raise _fail(e) from e
    click.echo(f"saved {p.name}")


@preset_group.command("delete")
@click.argument("name")
def preset_delete(name):
    from .modulation.presets import delete_user_preset

    if not delete_user_preset(name):
        raise click.ClickException(f"no user preset {name!r}")
    click.echo(f"deleted {name}")


@modulate_group.command("calibrate")
@click.argument("sample", type=click.Path(path_type=Path, exists=True, dir_okay=False))
@click.option("--preset", default="masculine", show_default=True)
@click.option("--identity", type=float, default=0.80, show_default=True)
def modulate_calibrate(sample, preset, identity):
    """Find the strongest strength of PRESET that keeps GE2E similarity >= IDENTITY on a sample of your voice."""
    from .audio_io import read_wav
    from .modulation.dsp import SpectralModulator
    from .modulation.presets import get_preset
    from .research.speaker_identity import get_encoder

    enc = get_encoder()
    if enc is None:
        raise click.ClickException("needs the speaker encoder: pip install 'aivoice[research]'")
    x, sr = read_wav(sample)
    p = get_preset(preset)
    best = 0.0
    click.echo(f"{'strength':<9} similarity")
    for s in (0.2, 0.35, 0.5, 0.65, 0.8, 1.0):
        y = SpectralModulator(sr, p.params(s), window_ms=p.window_ms, hop_ms=p.window_ms / 4).process_all(x)
        sim = enc.similarity(y, sr, x, sr)
        click.echo(f"{s:<9.2f} {sim:.3f}")
        if sim >= identity:
            best = s
    click.echo(f"\nstrongest strength with similarity >= {identity}: {best:.2f}  (`aivoice modulate preset use {p.name} --strength {best:.2f}`)")


# =============================================================================
# benchmark vc
# =============================================================================


@click.command("vc")
@click.option("--engines", "engine_list", default="installed", show_default=True,
              help="Comma list, 'installed' (every runnable engine) or 'all'")
@click.option("--voice", "voices", multiple=True, help="Target voice(s) from the library (repeatable)")
@click.option("--reference", "references", multiple=True, type=click.Path(path_type=Path, exists=True, dir_okay=False))
@click.option("--source", "sources", multiple=True, type=click.Path(path_type=Path, exists=True, dir_okay=False),
              help="Source WAVs (default: rights-free synthetic speech)")
@click.option("--synthetic", is_flag=True, help="Synthetic sources AND targets (espeak-ng voices; rights-free)")
@click.option("--quality", type=click.Choice(QUALITIES), default="balanced", show_default=True)
@click.option("--output", type=click.Path(path_type=Path), default=Path("benchmarks/listening"), show_default=True)
@click.option("--asr/--no-asr", default=True, help="WER/CER with an installed ASR backend")
@click.option("--json", "as_json", is_flag=True)
@click.option("--consent-ack", is_flag=True)
def benchmark_vc(engine_list, voices, references, sources, synthetic, quality, output, asr, as_json, consent_ack):
    """Every live VC engine on the same sources/targets: latency p50/p95/p99, RTF, similarity, leakage, WER, clicks, VRAM."""
    import tempfile

    from .audio_io import read_wav, write_wav
    from .consent import require_consent
    from .evaluation.metrics import available_asr
    from .vc.base import VoiceTarget
    from .vc.bench import Clip, format_summary, run_vc_benchmark, synthetic_clips
    from .vc.engines import ENGINES, all_vc_status

    if not synthetic:
        require_consent(ack=consent_ack)
    if engine_list == "installed":
        names = [s.name for s in all_vc_status() if s.state.runnable]
    elif engine_list == "all":
        names = [s.name for s in all_vc_status()]
    else:
        names = [n.strip() for n in engine_list.split(",") if n.strip()]
    syn_src, syn_tgt = synthetic_clips(2, 2)
    srcs = [Clip(Path(p).stem, *read_wav(p)) for p in sources] or syn_src
    tmp = Path(tempfile.mkdtemp(prefix="aivoice-vcbench-"))
    targets = [resolve_target(v) for v in voices] + [VoiceTarget(Path(r).stem, reference=Path(r)) for r in references]
    if not targets:
        for c in syn_tgt:
            write_wav(tmp / f"{c.name}.wav", c.audio, c.sr)
            targets.append(VoiceTarget(c.name, reference=tmp / f"{c.name}.wav"))
    asr_be = available_asr() if asr else None
    res = run_vc_benchmark([n for n in names if n in ENGINES or n in ("rtvc", "zerovc", "tvtsyn")], srcs, targets, output,
                           quality=quality, asr=asr_be,
                           label="synthetic speech; imitates no real person" if synthetic or not (voices or references) else
                           "voice conversion of authorized voices")
    if as_json:
        _json(res)
    else:
        click.echo(format_summary(res))
