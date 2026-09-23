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


@main.command("live")
@click.option("--reference", "reference", type=click.Path(path_type=Path, exists=True), default=None)
@click.option("--profile", default=None, help="Legacy XDG profile name")
@click.option("--voice", default=None, help="Installed voice library name/id")
@click.option("--mode", type=_mode_choice(), default="balanced", show_default=True)
@click.option("--device", default="cuda", show_default=True, help="cuda|cpu (CUDA preferred)")
@click.option("--chunk-ms", type=int, default=None, help="Override chunk size (ms)")
@click.option("--buffer-chunks", type=int, default=None)
@click.option("--consent-ack", is_flag=True)
def live_cmd(reference, profile, voice, mode, device, chunk_ms, buffer_chunks, consent_ack):
    """Real-time mic conversion (spawns MeanVC2 runtime when installed)."""
    require_consent(ack=consent_ack)
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
@click.option("--consent-ack", is_flag=True)
def file_cmd(input_wav, reference, profile, voice, output, mode, device, backend, consent_ack):
    """Convert a wav file using a reference speaker."""
    require_consent(ack=consent_ack)
    ref = _resolve_reference(reference, profile, voice)
    out = output or Path(input_wav).with_name(Path(input_wav).stem + "_aivoice.wav")
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
            engine = "MeanVC2" if ve.engine == "meanvc2" else ve.engine.upper()
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
    try:
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
def voices_import_rvc(path, index_path, reference, name):
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
    click.echo(
        f"Reference audio  {'available' if info.reference_audio or reference else 'missing'}"
    )
    click.echo("\nConverting RVC voice -> MeanVC2 profile")
    click.echo("(reference-audio adaptation — not weight conversion)\n")
    try:
        result = adapt_rvc_to_meanvc2(
            Path(path),
            display_name=name,
            reference_override=Path(reference) if reference else None,
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
        "upstream training scripts, and explicit opt-in — not implemented in 0.2.0."
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


@main.command("benchmark")
@click.option("--frames", default=50, show_default=True)
@click.option("--mode", type=_mode_choice(), default="balanced")
@click.option("--consent-ack", is_flag=True)
def benchmark_cmd(frames, mode, consent_ack):
    """Benchmark buffer overhead (full VC RTF needs models + preferably CUDA)."""
    require_consent(ack=consent_ack)
    click.echo(benchmark_overhead(frames=frames, mode=mode))
    click.echo("CPU fallback is documented; CUDA preferred for realtime RTF < 1.")


@main.group("models")
def models_group():
    """List / install MeanVC2 base models (not voice profiles)."""


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


if __name__ == "__main__":
    main(prog_name="aivoice")
