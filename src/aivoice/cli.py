"""Click CLI for aivoice."""

from __future__ import annotations

import json
from pathlib import Path

import click

from . import __version__
from .adapt import PERSON_NOTICE, adapt_rvc_to_meanvc2
from .consent import require_consent
from .devices import detect_audio_stack, format_devices, list_devices
from .doctor import format_doctor, run_doctor
from .models import install_model, list_models
from .paths import ensure_dirs
from .pipeline import benchmark_overhead, convert_file, run_live_subprocess
from .presets import MODES
from .profiles import create_profile, list_profiles, load_profile
from .providers import get_provider, list_providers
from .rvc.detect import inspect_path
from .selector import format_results_table, select_installed_voice
from .virtualmic import create_null_sink
from .voices import (
    create_from_reference,
    list_voices,
    load_voice,
    remove_voice,
    rename_voice,
    verify_voice,
)
from .workflow import interactive_search_install, search_voices


def _mode_choice():
    return click.Choice(sorted(MODES.keys()))


@click.group()
@click.version_option(__version__, prog_name="aivoice")
@click.pass_context
def main(ctx: click.Context) -> None:
    """Local real-time AI voice conversion (MeanVC2) with searchable voice library.

    Framing: research, VFX, avatars, filmmaking, consenting demos, disclosed
    synthetic media. No anonymity claims. Consent gates required for live/file.

    Quick start::

        aivoice virtualmic --search "Example Voice"
        aivoice virtualmic --voice "Example Voice"
    """
    ctx.ensure_object(dict)
    ensure_dirs()


@main.command("doctor")
@click.option("--json", "as_json", is_flag=True)
@click.option("--offline", is_flag=True, help="Skip network provider checks")
def doctor_cmd(as_json: bool, offline: bool):
    """Check MeanVC2, CUDA, PipeWire, voice library, and providers."""
    click.echo(format_doctor(run_doctor(check_network=not offline), as_json=as_json))


@main.command("devices")
@click.option("--json", "as_json", is_flag=True)
def devices_cmd(as_json: bool):
    """List PipeWire / Pulse / ALSA devices."""
    stack = detect_audio_stack()
    devices = list_devices()
    if as_json:
        click.echo(
            json.dumps(
                {
                    "stack": stack,
                    "devices": [
                        {"id": d.id, "name": d.name, "direction": d.direction, "backend": d.backend}
                        for d in devices
                    ],
                },
                indent=2,
            )
        )
        return
    click.echo(f"audio stack: {stack}")
    click.echo(format_devices(devices))


@main.command("search")
@click.argument("query")
@click.option("--provider", default="voice-models", show_default=True)
@click.option("--page", default=1, show_default=True)
@click.option("--details", is_flag=True, help="Fetch detail page for each hit (slow)")
@click.option("--json", "as_json", is_flag=True)
@click.option("--index", "pick", type=int, default=None, help="Select result # and print id")
def search_cmd(query, provider, page, details, as_json, pick):
    """Search a voice catalog (network). Does not upload mic audio."""
    click.echo(
        "Note: search queries go to the selected provider. "
        "Installed voices stay fully local afterward.",
        err=True,
    )
    try:
        page_res = search_voices(query, provider=provider, page=page)
    except Exception as e:  # noqa: BLE001
        raise click.ClickException(str(e)) from e
    rows = page_res.results
    if details:
        prov = get_provider(provider)
        enriched = []
        for r in rows:
            try:
                enriched.append(prov.get_model(r.id))
            except Exception:  # noqa: BLE001
                enriched.append(r)
        rows = enriched
    if as_json:
        click.echo(
            json.dumps(
                {
                    "query": query,
                    "page": page,
                    "has_more": page_res.has_more,
                    "results": [r.to_dict() for r in rows],
                },
                indent=2,
            )
        )
        return
    click.echo(f"Searching {provider}...\n")
    click.echo(f"Found {len(rows)} models" + (" (more pages available)" if page_res.has_more else ""))
    click.echo()
    click.echo(format_results_table(rows))
    if pick is not None:
        if pick < 1 or pick > len(rows):
            raise click.ClickException("index out of range")
        r = rows[pick - 1]
        click.echo(f"\nSelected: {r.name}\nid={r.id}\nurl={r.download_url}")


def _vc_options(f):
    """Options for the shared live pipeline (any engine; see `aivoice vc engines`)."""
    from .cli_vc import VC_ENGINE_CHOICES

    for opt in reversed([
        click.option("--engine", "vc_engine", type=click.Choice(VC_ENGINE_CHOICES), default=None,
                     help="Live engine on aivoice's shared pipeline (default: the previous MeanVC2/RVC runtime path)"),
        click.option("--priority", default="balanced", show_default=True,
                     help="For --engine auto: balanced|latency|gaming|similarity|naturalness|streaming-quality|low-vram|cpu"),
        click.option("--input", "input_device", default=None, help="Microphone (PipeWire source); never the aivoice mic"),
        click.option("--pitch", type=float, default=None, help="Transpose semitones (engines that support it, e.g. RVC)"),
        click.option("--index-rate", type=float, default=None, help="RVC FAISS index blend 0..1"),
        click.option("--modulate", "post_preset", default=None, help="Post-VC modulation preset (e.g. warm, broadcast)"),
        click.option("--modulate-strength", type=float, default=None),
        click.option("--denoise", type=float, default=None, help="Pre-VC noise suppression 0..1"),
        click.option("--fallback", default="dsp", show_default=True, help="Comma list tried when the engine fails"),
        click.option("--test-input", type=click.Path(path_type=Path, exists=True, dir_okay=False), default=None,
                     help="Use a WAV as the microphone (paced in real time)"),
        click.option("--duration", type=float, default=None, help="Stop after N seconds"),
    ]):
        f = opt(f)
    return f


def _run_unified(vc_engine, *, reference, profile, voice, mode, device, priority, input_device, pitch, index_rate,
                 post_preset, modulate_strength, denoise, fallback, test_input, duration, output="virtual",
                 owns_virtual_mic: bool | None = None):
    from .cli_vc import _input_device, _strength, build_engine, resolve_target, run_live_unified
    from .vc.base import StreamOptions
    from .vc.live import LiveConfig

    if voice or reference:
        target = resolve_target(voice, Path(reference) if reference else None)
    elif profile:
        target = resolve_target(None, _resolve_reference(None, profile, None))
    else:
        target = None
    if vc_engine in ("dsp", "passthrough") or target is None:
        if vc_engine not in ("dsp", "passthrough"):
            raise click.ClickException("voice conversion needs --voice/--reference; for your own voice use `aivoice modulate`")
    eng, note = build_engine(vc_engine, target=target, priority=priority, quality=mode)
    opts = StreamOptions(pitch_semitones=pitch or 0.0, index_rate=index_rate, quality=mode, device=device)
    eng.options = opts
    cfg = LiveConfig(mode="voice_conversion" if eng.capabilities().task == "voice_conversion" else "modulation",
                     engine=eng.capabilities().name, voice=target.name if target else None, quality=mode,
                     input_device=_input_device(input_device), pre={"denoise": denoise} if denoise else {},
                     post_preset=post_preset, post_strength=_strength(modulate_strength) if modulate_strength is not None else
                     (0.65 if post_preset else 0.0), fallback=[x.strip() for x in fallback.split(",") if x.strip()],
                     stream_options=opts)
    click.echo(f"Engine    {eng.capabilities().display_name}" + (f"   [{note}]" if note else ""))
    click.echo(f"Voice     {cfg.voice or '-'}   quality {mode}   fallback {' -> '.join(cfg.fallback + ['passthrough'])}")
    click.echo(f"Input     {cfg.input_device or '(default source)'}")
    return run_live_unified(engine=eng, cfg=cfg, target=target, input_file=test_input, output=output, duration=duration,
                            owns_virtual_mic=owns_virtual_mic)


@main.command("live")
@click.option("--reference", "reference", type=click.Path(path_type=Path, exists=True), default=None)
@click.option("--profile", default=None, help="Legacy XDG profile name")
@click.option("--voice", default=None, help="Installed voice library name/id")
@click.option("--mode", type=_mode_choice(), default="balanced", show_default=True)
@click.option("--device", default="cuda", show_default=True, help="cuda|cpu (CUDA preferred)")
@click.option("--chunk-ms", type=int, default=None, help="Override chunk size (ms)")
@click.option("--buffer-chunks", type=int, default=None)
@click.option("--output-to", type=click.Choice(["virtual", "default", "null"]), default="virtual", show_default=True,
              help="Shared pipeline only: aivoice-virtual-mic, your speakers, or nowhere (benchmarks)")
@_vc_options
@click.option("--consent-ack", is_flag=True)
def live_cmd(reference, profile, voice, mode, device, chunk_ms, buffer_chunks, output_to, vc_engine, priority, input_device,
             pitch, index_rate, post_preset, modulate_strength, denoise, fallback, test_input, duration, consent_ack):
    """Real-time mic conversion.

    \b
    aivoice live --engine xvc --voice NAME          shared pipeline, X-VC resident
    aivoice live --engine auto --priority gaming --voice NAME
    aivoice live --voice NAME                       previous path: MeanVC2 run_rt.py
    """
    require_consent(ack=consent_ack)
    if vc_engine:
        _run_unified(vc_engine, reference=reference, profile=profile, voice=voice, mode=mode, device=device, priority=priority,
                     input_device=input_device, pitch=pitch, index_rate=index_rate, post_preset=post_preset,
                     modulate_strength=modulate_strength, denoise=denoise, fallback=fallback, test_input=test_input,
                     duration=duration, output=output_to)
        return
    ref = _resolve_reference(reference, profile, voice)
    if chunk_ms or buffer_chunks:
        click.echo(
            f"note: chunk-ms={chunk_ms} buffer-chunks={buffer_chunks} "
            f"(passed via mode={mode}; upstream run_rt uses model preset)",
            err=True,
        )
    try:
        code = run_live_subprocess(ref, mode=mode, device=device)
    except Exception as e:  # noqa: BLE001
        raise click.ClickException(str(e)) from e
    if code != 0:
        raise click.ClickException(f"live session exited with code {code}")


@main.command("file")
@click.argument("input_wav", type=click.Path(path_type=Path, exists=True))
@click.option("--reference", type=click.Path(path_type=Path, exists=True), default=None)
@click.option("--profile", default=None)
@click.option("--voice", default=None)
@click.option("-o", "--output", type=click.Path(path_type=Path), default=None)
@click.option("--mode", type=_mode_choice(), default="balanced", show_default=True)
@click.option("--device", default="cuda", show_default=True)
@click.option("--backend", type=click.Choice(["auto", "meanvc2", "passthrough"]), default="auto")
@click.option("--engine", "vc_engine", default=None,
              help="Stream the file through a live engine (xvc, rvc, meanvc2, auto, ...): same code path as live")
@click.option("--priority", default="balanced", show_default=True)
@click.option("--consent-ack", is_flag=True)
def file_cmd(input_wav, reference, profile, voice, output, mode, device, backend, vc_engine, priority, consent_ack):
    """Convert a wav file using a reference speaker."""
    require_consent(ack=consent_ack)
    out = output or Path(input_wav).with_name(Path(input_wav).stem + "_aivoice.wav")
    if vc_engine:
        from .audio_io import read_wav
        from .cli_vc import build_engine, resolve_target
        from .evaluation.metrics import write_labeled_wav
        from .vc.base import StreamOptions
        from .vc.live import PIPE_SR, LiveConfig, LivePipeline

        target = resolve_target(voice, Path(reference) if reference else None) or (
            resolve_target(None, _resolve_reference(None, profile, None)) if profile else None)
        eng, note = build_engine(vc_engine, target=target, priority=priority, quality=mode)
        opts = StreamOptions(quality=mode, device=device)
        eng.options = opts
        pipe = LivePipeline(LiveConfig(engine=eng.capabilities().name, quality=mode, fallback=[], stream_options=opts), eng,
                            target=target)
        try:
            pipe.start()
            x, sr = read_wav(Path(input_wav))
            y = pipe.run_array(x, sr)
            eng.unload()
        except Exception as e:  # noqa: BLE001
            raise click.ClickException(str(e)) from e
        if pipe.stats.fallbacks:
            raise click.ClickException(f"engine failed: {pipe.stats.fallbacks[0]['reason']}")
        write_labeled_wav(out, y, PIPE_SR, comment=f"aivoice voice conversion ({eng.capabilities().name}); synthetic")
        click.echo(f"{out}" + (f"  [{note}]" if note else ""))
        click.echo(f"engine p95 {eng.timing_summary().get('p95_ms')} ms/chunk; output includes the engine's latency "
                   f"({eng.capabilities().algorithmic_latency_ms} ms) at the start", err=True)
        return
    ref = _resolve_reference(reference, profile, voice)
    be = None if backend == "auto" else backend
    try:
        convert_file(Path(input_wav), ref, out, mode=mode, device=device, backend=be)
    except Exception as e:  # noqa: BLE001
        raise click.ClickException(str(e)) from e
    click.echo(str(out))


@main.command("virtualmic")
@click.option("--reference", type=click.Path(path_type=Path, exists=True), default=None)
@click.option("--profile", default=None, help="Legacy profile name")
@click.option("--voice", default=None, help="Installed voice name/id")
@click.option("--search", "search_query", default=None, help="Search catalog, install, then start")
@click.option("--provider", default="voice-models", show_default=True)
@click.option("--index", type=int, default=None, help="Non-interactive search result index")
@click.option("--name", default="aivoice", show_default=True)
@click.option("--mode", type=_mode_choice(), default="balanced")
@click.option("--device", default="cuda")
@click.option("--consent-ack", is_flag=True)
@click.option("--setup-only", is_flag=True, help="Only create null sink / remap source")
@click.option("--engine", "rt_engine",
              type=click.Choice(["rvc", "meanvc2", "tvtsyn", "auto", "xvc", "dsp", "stylestream", "streamvoiceanon", "vossa"]), default=None,
              help="Realtime engine (default: the voice's saved choice; RVC v2 when it has a .pth). Every engine runs "
                   "on aivoice's shared pipeline (fallback, adaptive buffer, live status) unless --legacy-runtime")
@click.option("--unified", is_flag=True, hidden=True, help="Kept for compatibility: the shared pipeline is the default")
@click.option("--legacy-runtime", is_flag=True,
              help="rvc/meanvc2 only: the previous standalone runtimes (rt_worker.py / upstream run_rt.py) instead of the "
                   "shared pipeline")
@click.option("--denoise", type=float, default=None, help="Shared pipeline: pre-VC noise suppression 0..1")
@click.option("--priority", default="balanced", show_default=True, help="For --engine auto")
def virtualmic_cmd(
    reference,
    profile,
    voice,
    search_query,
    provider,
    index,
    name,
    mode,
    device,
    consent_ack,
    setup_only,
    rt_engine,
    unified,
    legacy_runtime,
    priority,
    denoise,
):
    """Virtual mic + live conversion. Supports --search / --voice / interactive pick."""
    require_consent(ack=consent_ack)

    if search_query:
        try:
            v = interactive_search_install(
                search_query, provider=provider, index=index, reference=reference
            )
        except Exception as e:  # noqa: BLE001
            raise click.ClickException(str(e)) from e
        voice = v.id
        reference = None
        profile = None
    elif voice is None and reference is None and profile is None:
        installed = list_voices()
        if installed:
            choice = select_installed_voice([v.display_name for v in installed])
            if choice is None:
                raise click.ClickException("cancelled")
            if choice == "__search__":
                q = click.prompt("Search query")
                v = interactive_search_install(q, provider=provider, index=index)
                voice = v.id
            elif choice == "__import_rvc__":
                path = Path(click.prompt("Path to RVC .zip/.pth"))
                result = adapt_rvc_to_meanvc2(path)
                if not result.voice:
                    raise click.ClickException(result.message)
                voice = result.voice.id
            elif choice == "__import_ref__":
                path = Path(click.prompt("Path to reference WAV"))
                disp = click.prompt("Voice name")
                v = create_from_reference(disp, path)
                voice = v.id
            else:
                voice = choice
        else:
            raise click.ClickException(
                "No installed voices. Use --search QUERY, --reference WAV, "
                "or: aivoice voices create NAME --reference file.wav"
            )

    try:
        info = create_null_sink(name)
    except Exception as e:  # noqa: BLE001
        raise click.ClickException(str(e)) from e

    ref = _resolve_reference(reference, profile, voice)
    engine = "MeanVC2"
    voice_label = voice or profile or Path(ref).name
    if voice:
        try:
            ve = load_voice(voice)
            voice_label = ve.display_name
            from .pipeline import voice_rt_settings

            engine = "RVC v2" if (rt_engine or voice_rt_settings(ve)["engine"]) == "rvc" else "MeanVC2"
        except FileNotFoundError:
            pass

    click.echo()
    click.echo("Starting virtual microphone...")
    click.echo()
    click.echo(f"Voice     {voice_label}")
    click.echo(f"Engine    {engine}")
    click.echo(f"Mode      {mode}")
    click.echo(f"Output    {info.name} ({info.backend})")
    click.echo(info.hint)
    click.echo()
    if setup_only:
        click.echo("Ready (setup-only)")
        return
    if not legacy_runtime or (rt_engine and rt_engine not in ("rvc", "meanvc2")):
        use = rt_engine
        if use is None:
            use = "meanvc2"
            if voice:
                try:
                    from .pipeline import voice_rt_settings

                    use = voice_rt_settings(load_voice(voice))["engine"]
                except FileNotFoundError:
                    pass
        _run_unified(use, reference=reference, profile=profile, voice=voice, mode=mode, device=device, priority=priority,
                     input_device=None, pitch=None, index_rate=None, post_preset=None, modulate_strength=None,
                     denoise=denoise or None, fallback="dsp", test_input=None, duration=None, output="virtual",
                     owns_virtual_mic=not info.reused)
        return
    try:
        from .pipeline import run_live_rvc, voice_rt_settings

        ve_rt = None
        if voice:
            try:
                ve_rt = load_voice(voice)
            except FileNotFoundError:
                ve_rt = None
        rt = voice_rt_settings(ve_rt) if ve_rt else {"engine": "meanvc2"}
        use = (rt_engine or rt["engine"]).lower()
        if use == "rvc" and ve_rt is not None:
            click.echo(f"Realtime  RVC v2  pitch={rt['pitch']:+g} index_rate={rt['index_rate']}")
            code = run_live_rvc(ve_rt, mode=mode, device=device, settings=rt)
        else:
            click.echo("Realtime  MeanVC2")
            code = run_live_subprocess(ref, mode=mode, device=device)
    except Exception as e:  # noqa: BLE001
        raise click.ClickException(str(e)) from e
    if code != 0:
        raise click.ClickException(f"virtualmic live exited with code {code}")


@main.group("voices")
def voices_group():
    """Installed voice library (MeanVC2 profiles / RVC imports)."""


@voices_group.command("list")
@click.option("--json", "as_json", is_flag=True)
def voices_list(as_json: bool):
    items = list_voices()
    if as_json:
        click.echo(json.dumps([v.to_dict() for v in items], indent=2))
        return
    if not items:
        click.echo("(no voices)")
        return
    click.echo(f"{'NAME':<28} {'ENGINE':<10} {'SOURCE':<18} {'STATUS'}")
    for v in items:
        src = v.source_provider or "local"
        click.echo(f"{v.display_name[:28]:<28} {v.engine:<10} {src[:18]:<18} {v.status}")


@voices_group.command("info")
@click.argument("name")
@click.option("--json", "as_json", is_flag=True)
def voices_info(name, as_json):
    try:
        v = load_voice(name)
    except FileNotFoundError as e:
        raise click.ClickException(str(e)) from e
    if as_json:
        click.echo(json.dumps(v.to_dict(), indent=2))
        return
    for k, val in v.to_dict().items():
        if k == "extra":
            continue
        click.echo(f"{k}: {val}")
    kind = (v.extra or {}).get("reference_kind")
    if kind:
        synth = (v.extra or {}).get("synthesis") or {}
        detail = f" ({synth.get('tts')} TTS -> {synth.get('rvc')})" if kind == "synthesized" and synth else ""
        click.echo(f"reference_kind: {kind}{detail}")


@voices_group.command("remove")
@click.argument("name")
@click.option("--yes", is_flag=True)
def voices_remove(name, yes):
    if not yes and not click.confirm(f"Remove voice {name!r}?"):
        raise click.Abort()
    try:
        remove_voice(name)
    except FileNotFoundError as e:
        raise click.ClickException(str(e)) from e
    click.echo(f"removed {name}")


@voices_group.command("rename")
@click.argument("name")
@click.argument("new_name")
def voices_rename(name, new_name):
    try:
        v = rename_voice(name, new_name)
    except FileNotFoundError as e:
        raise click.ClickException(str(e)) from e
    click.echo(f"renamed -> {v.display_name}")


@voices_group.command("verify")
@click.argument("name")
def voices_verify(name):
    try:
        issues = verify_voice(name)
    except FileNotFoundError as e:
        raise click.ClickException(str(e)) from e
    if not issues:
        click.echo("ok")
    else:
        for i in issues:
            click.echo(f"issue: {i}")
        raise click.ClickException("verify failed")


@voices_group.command("resynth")
@click.argument("name")
@click.option("--synth-text", default=None, help="Text to speak (default: Harvard sentences).")
def voices_resynth(name, synth_text):
    """Regenerate a voice's reference.wav (TTS -> its RVC model) with the current engines."""
    from .audio_prep import prepare_reference, score_reference
    from .rvc.detect import inspect_path
    from .rvc.synth import SynthError, synthesize_reference
    from .voices import _sha256, _write, load_voice

    try:
        v = load_voice(name)
    except FileNotFoundError as e:
        raise click.ClickException(str(e)) from e
    rvc_meta = (v.extra or {}).get("rvc") or {}
    root = rvc_meta.get("root")
    if not root or not Path(root).exists():
        raise click.ClickException("voice has no RVC package on disk to synthesize from")
    info = inspect_path(Path(root))
    try:
        synth = synthesize_reference(info, v.dir() / ".synth", text=synth_text)
    except SynthError as e:
        raise click.ClickException(str(e)) from e
    ref = v.dir() / "reference.wav"
    prep = prepare_reference(synth.path, ref)
    v.checksum_reference = _sha256(ref)
    v.extra = {
        **(v.extra or {}),
        "reference_kind": "synthesized",
        "synthesis": {"tts": synth.tts_engine, "rvc": synth.rvc_engine, "text": synth.text},
        "prep_notes": prep.notes,
        "quality_notes": score_reference(ref),
        "duration_s": prep.duration_s,
    }
    _write(v)
    click.echo(f"{v.display_name}: reference regenerated ({synth.tts_engine} -> {synth.rvc_engine})")


@voices_group.command("rt")
@click.argument("name")
@click.option("--engine", type=click.Choice(["rvc", "meanvc2"]), default=None)
@click.option("--pitch", type=float, default=None, help="Transpose in semitones (RVC)")
@click.option("--index-rate", type=float, default=None, help="0..1 FAISS index blend (RVC)")
@click.option("--threshold", type=float, default=None, help="Noise gate dBFS (RVC)")
@click.option("--json", "as_json", is_flag=True)
def voices_rt(name, engine, pitch, index_rate, threshold, as_json):
    """Show/set a voice's realtime settings (engine, pitch, index rate)."""
    import json as _json

    from .pipeline import save_voice_rt_settings, voice_rt_settings, voice_rvc_files
    from .voices import load_voice

    try:
        v = load_voice(name)
    except FileNotFoundError as e:
        raise click.ClickException(str(e)) from e
    if any(x is not None for x in (engine, pitch, index_rate, threshold)):
        cfg = save_voice_rt_settings(v, engine=engine, pitch=pitch, index_rate=index_rate, threshold=threshold)
    else:
        cfg = voice_rt_settings(v)
    pth, idx = voice_rvc_files(v)
    cfg = {**cfg, "rvc_model": str(pth) if pth else None, "rvc_index": str(idx) if idx else None}
    click.echo(_json.dumps(cfg) if as_json else "\n".join(f"{k:12} {val}" for k, val in cfg.items()))


@voices_group.command("create")
@click.argument("name")
@click.option("--reference", type=click.Path(path_type=Path, exists=True), required=True)
def voices_create(name, reference):
    """Create a MeanVC2 voice from reference audio (zero-shot)."""
    try:
        v = create_from_reference(name, Path(reference))
    except (ValueError, FileExistsError, FileNotFoundError) as e:
        raise click.ClickException(str(e)) from e
    click.echo(f"created {v.display_name} ({v.id}) engine={v.engine}")


@voices_group.command("import-rvc")
@click.argument("path", type=click.Path(path_type=Path, exists=True))
@click.option("--index", "index_path", type=click.Path(path_type=Path, exists=True), default=None)
@click.option("--reference", type=click.Path(path_type=Path, exists=True), default=None)
@click.option("--name", default=None, help="Display name")
@click.option(
    "--no-synthesize",
    is_flag=True,
    help="Do not synthesize a reference (TTS -> RVC) when the package has no reference audio.",
)
@click.option("--synth-text", default=None, help="Text to speak when synthesizing a reference.")
def voices_import_rvc(path, index_path, reference, name, no_synthesize, synth_text):
    """Import RVC v1/v2 package -> MeanVC2 profile via reference audio."""
    click.echo(PERSON_NOTICE)
    click.echo("\nRVC Voice Import\n")
    click.echo("Inspecting...")
    try:
        info = inspect_path(Path(path))
    except Exception as e:  # noqa: BLE001
        raise click.ClickException(str(e)) from e
    click.echo(f"{info.generation or 'Unknown'} detected.")
    if info.checkpoint:
        click.echo(f"Checkpoint       {info.checkpoint}")
    if info.index or index_path:
        click.echo(f"Index            {index_path or info.index}")
    if info.sample_rate:
        click.echo(f"Sample rate      {info.sample_rate}")
    if info.pitch_guidance is not None:
        click.echo(f"Pitch guidance   {'yes' if info.pitch_guidance else 'no'}")
    have_ref = bool(info.reference_audio or reference)
    will_synth = not have_ref and bool(info.checkpoint) and not no_synthesize
    click.echo(
        "Reference audio  "
        + ("available" if have_ref else "missing (will synthesize: TTS -> RVC)" if will_synth else "missing")
    )
    if will_synth:
        from .rvc.synth import PICKLE_WARNING

        click.echo(f"Note             {PICKLE_WARNING}")
    click.echo("\nConverting RVC voice -> MeanVC2 profile")
    click.echo("(reference-audio adaptation — not weight conversion)\n")
    try:
        result = adapt_rvc_to_meanvc2(
            Path(path),
            display_name=name,
            reference_override=Path(reference) if reference else None,
            synthesize=False if no_synthesize else None,
            synth_text=synth_text,
        )
    except Exception as e:  # noqa: BLE001
        raise click.ClickException(str(e)) from e
    if result.voice is None:
        raise click.ClickException(result.message)
    click.echo(result.message)
    click.echo(f"\nInstalled as:\n{result.voice.display_name}")


@voices_group.command("install")
@click.option("--search", "query", default=None)
@click.option("--provider", default="voice-models")
@click.option("--id", "model_id", default=None)
@click.option("--index", type=int, default=None)
@click.option("--name", default=None)
def voices_install(query, provider, model_id, index, name):
    """Install from catalog search or provider id."""
    if model_id:
        from .workflow import install_from_ref

        prov = get_provider(provider)
        try:
            model = prov.get_model(model_id)
        except Exception as e:  # noqa: BLE001
            raise click.ClickException(str(e)) from e
        if name:
            model.name = name
        try:
            v = install_from_ref(model)
        except Exception as e:  # noqa: BLE001
            raise click.ClickException(str(e)) from e
        click.echo(f"installed {v.display_name}")
        return
    if not query:
        raise click.ClickException("Provide --search QUERY or --id MODEL_ID")
    try:
        v = interactive_search_install(query, provider=provider, index=index)
    except Exception as e:  # noqa: BLE001
        raise click.ClickException(str(e)) from e
    click.echo(f"installed {v.display_name}")


@voices_group.command("optimize")
@click.argument("name")
def voices_optimize(name):
    """Advanced MeanVC2 speaker fine-tune (not auto-run)."""
    try:
        v = load_voice(name)
    except FileNotFoundError as e:
        raise click.ClickException(str(e)) from e
    raise click.ClickException(
        f"Fine-tuning for {v.display_name!r} is not enabled by default.\n"
        "Zero-shot reference profiles are the fast path. "
        "Speaker-specific MeanVC2 fine-tuning requires substantial clean target speech, "
        "upstream training scripts, and explicit opt-in — not implemented."
    )


@main.group("providers")
def providers_group():
    """Voice catalog providers."""


@providers_group.command("list")
@click.option("--json", "as_json", is_flag=True)
def providers_list(as_json: bool):
    rows = []
    for name, prov in list_providers().items():
        caps = prov.capabilities()
        rows.append(
            {
                "provider": name,
                "search": caps.search,
                "download": caps.download,
                "notes": caps.notes,
            }
        )
    if as_json:
        click.echo(json.dumps(rows, indent=2))
        return
    click.echo(f"{'PROVIDER':<16} {'SEARCH':<8} {'DOWNLOAD':<10} STATUS")
    for r in rows:
        click.echo(
            f"{r['provider']:<16} {'yes' if r['search'] else 'no':<8} "
            f"{'yes' if r['download'] else 'no':<10} ready"
        )


@providers_group.command("info")
@click.argument("name")
def providers_info(name):
    try:
        prov = get_provider(name)
    except KeyError as e:
        raise click.ClickException(str(e)) from e
    caps = prov.capabilities()
    click.echo(f"provider: {name}")
    click.echo(f"search: {caps.search}")
    click.echo(f"download: {caps.download}")
    click.echo(f"versions: {caps.versions}")
    click.echo(f"details: {caps.details}")
    click.echo(f"notes: {caps.notes}")


@main.group("profiles")
def profiles_group():
    """Legacy XDG voice profiles (prefer `voices`)."""


@profiles_group.command("list")
@click.option("--json", "as_json", is_flag=True)
def profiles_list(as_json: bool):
    items = list_profiles()
    if as_json:
        click.echo(json.dumps([p.to_dict() for p in items], indent=2))
        return
    if not items:
        click.echo("(no profiles)")
        return
    for p in items:
        click.echo(f"{p.name}\tmode={p.mode}\tref={p.reference}")


@profiles_group.command("create")
@click.option("--name", required=True)
@click.option("--reference", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--mode", type=_mode_choice(), default="balanced")
@click.option("--notes", default="")
def profiles_create(name, reference, mode, notes):
    try:
        p = create_profile(name, Path(reference), mode=mode, notes=notes)
    except (ValueError, FileExistsError, FileNotFoundError) as e:
        raise click.ClickException(str(e)) from e
    click.echo(f"created profile {p.name} at {p.path()}")


@main.group("profile")
def profile_group():
    """Alias for profiles."""


@profile_group.command("create")
@click.option("--name", required=True)
@click.option("--reference", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--mode", type=_mode_choice(), default="balanced")
@click.option("--notes", default="")
def profile_create(name, reference, mode, notes):
    try:
        p = create_profile(name, Path(reference), mode=mode, notes=notes)
    except (ValueError, FileExistsError, FileNotFoundError) as e:
        raise click.ClickException(str(e)) from e
    click.echo(f"created profile {p.name} at {p.path()}")


@profile_group.command("list")
def profile_list():
    for p in list_profiles():
        click.echo(f"{p.name}\tmode={p.mode}\tref={p.reference}")


@main.group("benchmark", invoke_without_command=True)
@click.option("--frames", default=50, show_default=True)
@click.option("--mode", type=_mode_choice(), default="balanced")
@click.option("--consent-ack", is_flag=True)
@click.pass_context
def benchmark_cmd(ctx, frames, mode, consent_ack):
    """Benchmark buffer overhead (full VC RTF needs models + preferably CUDA).

    `aivoice benchmark research ...` runs the engine research benchmark.
    """
    if ctx.invoked_subcommand is not None:
        return
    require_consent(ack=consent_ack)
    click.echo(benchmark_overhead(frames=frames, mode=mode))
    click.echo("CPU fallback is documented; CUDA preferred for realtime RTF < 1.")


@main.group("models")
def models_group():
    """MeanVC2 base models (list/install) and the Hugging Face model manager (search/download/...)."""


@models_group.command("list")
@click.option("--json", "as_json", is_flag=True)
def models_list(as_json: bool):
    rows = list_models()
    if as_json:
        click.echo(
            json.dumps([{"name": r.name, "installed": r.installed, **r.meta} for r in rows], indent=2)
        )
        return
    for r in rows:
        mark = "yes" if r.installed else "no"
        click.echo(f"{r.name}\tinstalled={mark}\tsize={r.meta.get('size_hint')}")
        click.echo(f"  source: {r.meta.get('source')}")
        click.echo(f"  claimed code license: {r.meta.get('code_license_claimed')}")
        click.echo(f"  compliance gap: {r.meta.get('code_license_gap')}")
        click.echo(f"  weights: {r.meta.get('weights_license')}")
        if r.meta.get("notes"):
            click.echo(f"  notes: {r.meta['notes']}")


    click.echo("\nExperimental-mode models (ASR/TTS/LLM via Hugging Face): `aivoice models known`, `aivoice models installed`")


@models_group.command("install")
@click.argument("model")
@click.option("--yes", is_flag=True, help="Acknowledge license gap/size and download")
def models_install(model: str, yes: bool):
    try:
        msg = install_model(model, yes=yes)
    except (KeyError, RuntimeError) as e:
        raise click.ClickException(str(e)) from e
    click.echo(msg)


@main.command("convert-rvc")
@click.argument("path", type=click.Path(path_type=Path, exists=True))
@click.option("--reference", type=click.Path(path_type=Path, exists=True), default=None)
@click.option("--name", default=None)
@click.pass_context
def convert_rvc_cmd(ctx, path, reference, name):
    """Alias for `voices import-rvc`."""
    ctx.invoke(voices_import_rvc, path=path, index_path=None, reference=reference, name=name)


def _resolve_reference(
    reference: Path | None, profile: str | None, voice: str | None = None
) -> Path:
    if reference is not None:
        return Path(reference)
    if voice:
        try:
            v = load_voice(voice)
        except FileNotFoundError as e:
            raise click.ClickException(str(e)) from e
        if not v.reference or not Path(v.reference).is_file():
            raise click.ClickException(
                f"voice {voice!r} has no usable reference audio "
                f"(status={v.status}, engine={v.engine})"
            )
        return Path(v.reference)
    if profile:
        try:
            p = load_profile(profile)
        except FileNotFoundError as e:
            raise click.ClickException(str(e)) from e
        return Path(p.reference)
    raise click.ClickException("Provide --reference, --voice, or --profile")


from .cli_experimental import register as _register_experimental  # noqa: E402

_register_experimental(main, models_group)

from .cli_clone import benchmark_research as _benchmark_research  # noqa: E402
from .cli_clone import register as _register_clone  # noqa: E402
from .cli_vc import benchmark_vc as _benchmark_vc  # noqa: E402
from .cli_vc import register as _register_vc  # noqa: E402

_register_clone(main)
_register_vc(main)

from .cli_reproduce import register as _register_reproduce  # noqa: E402

_register_reproduce(main)
benchmark_cmd.add_command(_benchmark_research)
benchmark_cmd.add_command(_benchmark_vc)


if __name__ == "__main__":
    main(prog_name="aivoice")
