"""Click CLI for aivoice."""

from __future__ import annotations

import json
from pathlib import Path

import click

from . import __version__
from .consent import require_consent
from .devices import detect_audio_stack, format_devices, list_devices
from .models import install_model, list_models
from .paths import ensure_dirs
from .pipeline import benchmark_overhead, convert_file, run_live_subprocess
from .presets import MODES
from .profiles import create_profile, list_profiles, load_profile
from .virtualmic import create_null_sink


def _mode_choice():
    return click.Choice(sorted(MODES.keys()))


@click.group()
@click.version_option(__version__, prog_name="aivoice")
@click.pass_context
def main(ctx: click.Context) -> None:
    """Local real-time AI voice changer (MeanVC2 research wrapper).

    Framing: research, VFX, avatars, filmmaking, consenting demos, disclosed
    synthetic media. No anonymity claims. Consent gates required for live/file.
    """
    ctx.ensure_object(dict)
    ensure_dirs()


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


@main.command("live")
@click.option("--reference", "reference", type=click.Path(path_type=Path, exists=True), default=None)
@click.option("--profile", default=None, help="Use saved XDG voice profile")
@click.option("--mode", type=_mode_choice(), default="balanced", show_default=True)
@click.option("--device", default="cuda", show_default=True, help="cuda|cpu (CUDA preferred)")
@click.option("--chunk-ms", type=int, default=None, help="Override chunk size (ms)")
@click.option("--buffer-chunks", type=int, default=None)
@click.option("--consent-ack", is_flag=True)
def live_cmd(reference, profile, mode, device, chunk_ms, buffer_chunks, consent_ack):
    """Real-time mic conversion (spawns MeanVC2 runtime when installed)."""
    require_consent(ack=consent_ack)
    ref = _resolve_reference(reference, profile)
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
@click.option("-o", "--output", type=click.Path(path_type=Path), default=None)
@click.option("--mode", type=_mode_choice(), default="balanced", show_default=True)
@click.option("--device", default="cuda", show_default=True)
@click.option("--backend", type=click.Choice(["auto", "meanvc2", "passthrough"]), default="auto")
@click.option("--consent-ack", is_flag=True)
def file_cmd(input_wav, reference, profile, output, mode, device, backend, consent_ack):
    """Convert a wav file using a reference speaker."""
    require_consent(ack=consent_ack)
    ref = _resolve_reference(reference, profile)
    out = output or Path(input_wav).with_name(Path(input_wav).stem + "_aivoice.wav")
    be = None if backend == "auto" else backend
    try:
        convert_file(Path(input_wav), ref, out, mode=mode, device=device, backend=be)
    except Exception as e:  # noqa: BLE001
        raise click.ClickException(str(e)) from e
    click.echo(str(out))


@main.command("virtualmic")
@click.option("--reference", type=click.Path(path_type=Path, exists=True), default=None)
@click.option("--profile", default=None)
@click.option("--name", default="aivoice", show_default=True)
@click.option("--mode", type=_mode_choice(), default="balanced")
@click.option("--device", default="cuda")
@click.option("--consent-ack", is_flag=True)
@click.option("--setup-only", is_flag=True, help="Only create null sink / remap source")
def virtualmic_cmd(reference, profile, name, mode, device, consent_ack, setup_only):
    """Create a virtual mic (PipeWire/Pulse) and optionally start live conversion into it."""
    require_consent(ack=consent_ack)
    try:
        info = create_null_sink(name)
    except Exception as e:  # noqa: BLE001
        raise click.ClickException(str(e)) from e
    click.echo(f"virtual mic: {info.name} ({info.backend})")
    click.echo(info.hint)
    if setup_only:
        return
    ref = _resolve_reference(reference, profile)
    try:
        code = run_live_subprocess(ref, mode=mode, device=device)
    except Exception as e:  # noqa: BLE001
        raise click.ClickException(str(e)) from e
    if code != 0:
        raise click.ClickException(f"virtualmic live exited with code {code}")


@main.group("profiles")
def profiles_group():
    """Manage XDG voice profiles."""


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


# alias: profile create (user asked for `profile create` as well as profiles)
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
    """List / install MeanVC2 (never silent giant downloads)."""


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


def _resolve_reference(reference: Path | None, profile: str | None) -> Path:
    if reference is not None:
        return Path(reference)
    if profile:
        try:
            p = load_profile(profile)
        except FileNotFoundError as e:
            raise click.ClickException(str(e)) from e
        return Path(p.reference)
    raise click.ClickException("Provide --reference voice.wav or --profile <name>")


if __name__ == "__main__":
    main(prog_name="aivoice")
