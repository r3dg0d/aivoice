"""CLI for experimental real-time mode, the Hugging Face model manager, LLM, and TTS voice profiles.

Registered onto the main ``aivoice`` group by :func:`register` (see cli.py), so
there is one command tree and one framework.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

import click

from . import settings as st

CONTROL_SIGNALS = {
    "toggle-mute": signal.SIGUSR1,
    "interrupt": signal.SIGUSR2,
    "ptt-down": signal.SIGRTMIN,
    "ptt-up": signal.SIGRTMIN + 1,
    "mute": signal.SIGRTMIN + 2,
    "unmute": signal.SIGRTMIN + 3,
    "emergency-stop": signal.SIGRTMIN + 4,
}


def _fail(e: BaseException) -> click.ClickException:
    return click.ClickException(str(e))


def _echo_json(data: Any) -> None:
    click.echo(json.dumps(data, indent=2, default=str))


def _manager():
    from .hub.manager import HuggingFaceModelManager

    return HuggingFaceModelManager()


def _engine_pid() -> int | None:
    from .realtime.build import pid_path

    try:
        pid = int(pid_path().read_text().strip())
    except (OSError, ValueError):
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    except PermissionError:
        return pid
    return pid


def _read_status() -> dict[str, Any] | None:
    from .realtime.build import status_path

    try:
        return json.loads(status_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def register(main: click.Group, models_group: click.Group) -> None:
    main.add_command(experimental)
    main.add_command(llm_group)
    main.add_command(voice_group)
    main.add_command(credentials_group)
    main.add_command(hardware_cmd)
    for cmd in (
        models_search, models_info, models_download, models_installed, models_loaded, models_load,
        models_unload, models_remove, models_update, models_verify, models_benchmark, models_cache,
        models_cleanup, models_import, models_default, models_known,
    ):
        models_group.add_command(cmd)


# =============================================================================
# experimental
# =============================================================================


@click.group("experimental")
def experimental():
    """Experimental real-time mode: mic -> ASR -> (LLM) -> Fish Drama 3 / TTS -> virtual mic."""


@experimental.command("enable")
def exp_enable():
    """Turn experimental mode on (persists)."""
    st.set_value("experimental.enabled", True)
    click.echo("experimental mode enabled. Next: `aivoice experimental status` then `aivoice experimental start`.")


@experimental.command("disable")
def exp_disable():
    """Turn experimental mode off (persists)."""
    st.set_value("experimental.enabled", False)
    click.echo("experimental mode disabled (MeanVC2 live/virtualmic commands are unaffected).")


def _install_signal_handlers(eng, stop_event: threading.Event) -> None:
    def handler(signum, frame):  # noqa: ARG001
        if signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            stop_event.set()
        elif signum == CONTROL_SIGNALS["toggle-mute"]:
            eng.toggle_mute()
        elif signum == CONTROL_SIGNALS["interrupt"]:
            eng.interrupt("user")
        elif signum == CONTROL_SIGNALS["ptt-down"]:
            eng.set_ptt(True)
        elif signum == CONTROL_SIGNALS["ptt-up"]:
            eng.set_ptt(False)
        elif signum == CONTROL_SIGNALS["mute"]:
            eng.mute(True)
        elif signum == CONTROL_SIGNALS["unmute"]:
            eng.mute(False)
        elif signum == CONTROL_SIGNALS["emergency-stop"]:
            eng.emergency_stop()

    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, *CONTROL_SIGNALS.values()):
        signal.signal(s, handler)


@experimental.command("start")
@click.option("--input", "input_file", type=click.Path(path_type=Path, exists=True), default=None,
              help="Use a WAV file as the microphone (paced in real time)")
@click.option("--output", type=click.Choice(["virtual", "default", "null"]), default="virtual", show_default=True)
@click.option("--preset", default=None, help="Apply a preset before starting")
@click.option("--asr", "asr_name", default=None, help="Override ASR backend for this run")
@click.option("--duration", type=float, default=None, help="Stop after N seconds")
@click.option("--ptt", is_flag=True, help="Push-to-talk: `aivoice experimental control ptt-down|ptt-up`")
@click.option("--monitor", is_flag=True, help="Also play the virtual mic on your default output (use headphones)")
@click.option("--no-warmup", is_flag=True)
@click.option("--quiet", is_flag=True, help="No live transcript output")
@click.option("--consent-ack", is_flag=True)
def exp_start(input_file, output, preset, asr_name, duration, ptt, monitor, no_warmup, quiet, consent_ack):
    """Run the real-time pipeline in the foreground (Ctrl-C to stop)."""
    from .consent import require_consent
    from .realtime.build import BuildError, build_engine, pid_path
    from .realtime.presets import apply_preset
    from .realtime.timeline import format_report

    require_consent(ack=consent_ack)
    if not st.get("experimental.enabled"):
        raise click.ClickException("experimental mode is disabled: run `aivoice experimental enable` first")
    if _engine_pid():
        raise click.ClickException(f"an engine is already running (pid {_engine_pid()}); `aivoice experimental stop`")
    if preset:
        try:
            apply_preset(preset)
        except KeyError as e:
            raise _fail(e) from e
    cfg = st.load()
    if asr_name:
        cfg["asr"]["backend"] = asr_name
    if ptt:
        cfg["audio"]["input_mode"] = "ptt"
    if monitor:
        cfg["audio"]["monitor"] = True
    try:
        eng, report = build_engine(cfg, input_file=input_file, output=output)
    except (BuildError, Exception) as e:  # noqa: BLE001
        raise _fail(e) from e
    click.echo("Experimental real-time mode\n")
    for line in report.lines():
        click.echo(line)
    if input_file is None and cfg["audio"].get("monitor") and output == "virtual":
        from .virtualmic import enable_monitor

        try:
            enable_monitor(cfg["audio"].get("virtual_mic_name", "aivoice"))
            click.echo("Monitor  on (virtual mic -> default output; use headphones to avoid feedback)")
        except RuntimeError as e:
            click.echo(f"warning: monitor not enabled: {e}", err=True)

    def on_event(kind: str, data: dict[str, Any]) -> None:
        if quiet:
            return
        if kind == "partial":
            click.echo(f"\r\033[K  … {data['text'][-110:]}", nl=False, err=True)
        elif kind == "final":
            click.echo(f"\r\033[K  ✓ {data['text']}", err=True)
        elif kind == "commit":
            click.echo(f"\r\033[K  ♪ {data['text']}", err=True)
        elif kind == "error":
            click.echo(f"\r\033[K  ! {data.get('stage')}: {data.get('error')}", err=True)
        elif kind == "state" and data["state"] in ("MUTED", "LISTENING"):
            click.echo(f"\r\033[K  [{data['state'].lower()}]", err=True)

    eng.on_event = on_event
    stop = threading.Event()
    _install_signal_handlers(eng, stop)
    pid_path().parent.mkdir(parents=True, exist_ok=True)
    pid_path().write_text(str(os.getpid()))
    try:
        click.echo("\nloading models / warming up..." if not no_warmup else "\nloading models...")
        eng.prepare(warmup=not no_warmup)
        if eng.warmup_ms is not None:
            click.echo(f"TTS warmup: first audio in {eng.warmup_ms:.0f} ms")
        eng.start(warmup=False)
        click.echo("listening (Ctrl-C to stop; `aivoice experimental control toggle-mute|interrupt`)\n")
        t_end = time.monotonic() + duration if duration else None
        while not stop.is_set():
            if t_end and time.monotonic() >= t_end:
                break
            if input_file is not None and eng.wait_idle(timeout=0.2):
                break
            stop.wait(0.2)
    except Exception as e:  # noqa: BLE001
        raise _fail(e) from e
    finally:
        eng.stop()
        for fn in report.cleanup:
            try:
                fn()
            except Exception:  # noqa: BLE001
                pass
        pid_path().unlink(missing_ok=True)
        if not quiet:
            click.echo("\n" + format_report(eng.timelines.report()))


@experimental.command("stop")
def exp_stop():
    """Stop the running engine (SIGTERM; it finishes cleanly and removes its virtual mic if it created it)."""
    pid = _engine_pid()
    if not pid:
        click.echo("no engine running")
        return
    os.kill(pid, signal.SIGTERM)
    for _ in range(50):
        if not _engine_pid():
            click.echo(f"stopped (pid {pid})")
            return
        time.sleep(0.1)
    raise click.ClickException(f"engine pid {pid} did not stop within 5 s")


@experimental.command("control")
@click.argument("action", type=click.Choice(sorted(CONTROL_SIGNALS)))
def exp_control(action):
    """Send a live control to the running engine (bind these to compositor hotkeys)."""
    pid = _engine_pid()
    if not pid:
        raise click.ClickException("no engine running")
    os.kill(pid, CONTROL_SIGNALS[action])
    click.echo(f"{action} -> pid {pid}")


def _status_text(s: dict[str, Any]) -> str:
    from .realtime.timeline import format_report

    lines = [
        f"state     {s['state']}{' (muted)' if s.get('muted') else ''}  pid {s['pid']}  up {s['uptime_s']:.0f}s",
        f"asr       {s['asr']['backend']} [{s['asr']['streaming']}]",
        f"tts       {s['tts']['name']} streaming={s['tts']['streaming']} remote={s['tts']['remote']}",
        f"llm       {s['llm']['mode']} {s['llm']['backend'] or ''}".rstrip(),
        f"voice     {s.get('voice') or '(default)'}",
        f"queues    transform {s['queues']['transform']}  synth {s['queues']['synth']}  jobs {s['queues']['jobs']}",
        f"player    {s['player']['rate']} Hz  buffered {s['player']['buffered_ms']:.0f} ms  underruns {s['player']['underruns']}",
        f"capture   level {s['capture']['level']:.2f}  dropped {s['capture']['dropped_blocks']}",
        f"heard     {s['last']['final'][-100:]}",
        f"spoke     {s['last']['spoken'][-100:]}",
        "",
        format_report(s["latency"]),
    ]
    if s.get("errors"):
        lines += ["", "recent errors:"] + [f"  {e}" for e in s["errors"]]
    return "\n".join(lines)


@experimental.command("status")
@click.option("--json", "as_json", is_flag=True)
@click.option("--watch", is_flag=True, help="Refresh every second")
def exp_status(as_json, watch):
    """Configuration, availability, and the running engine's live state."""
    from .asr import available_backends
    from .hardware import detect

    while True:
        s = _read_status() if _engine_pid() else None
        if as_json:
            _echo_json({"enabled": st.get("experimental.enabled"), "engine": s, "settings": st.load()})
            return
        if watch:
            click.clear()
        cfg = st.load()
        click.echo(f"experimental mode: {'enabled' if cfg['experimental']['enabled'] else 'disabled'}  preset: {cfg['experimental'].get('preset')}")
        click.echo(f"asr: {cfg['asr']['backend']} (policy {cfg['asr']['policy']})  tts: {cfg['tts']['backend']} @ {cfg['tts']['server_url']}")
        click.echo(f"llm: {'on' if cfg['llm']['enabled'] else 'off'} mode={cfg['llm']['mode']} model={cfg['llm']['model']}")
        click.echo(f"voice: {cfg['voice']['active'] or '(default)'}  barge-in: {cfg['audio']['barge_in']}  input: {cfg['audio']['input_mode']}")
        consent = cfg["privacy"]["remote_consent"]
        click.echo("remote processing allowed for: " + (", ".join(k for k, v in consent.items() if v) or "nothing (local only)"))
        if not watch:
            click.echo("\nASR runtimes:")
            for k, why in available_backends().items():
                click.echo(f"  {k:<18} {'ready to load' if why is None else why}")
            hw = detect()
            click.echo(f"\nGPU: {', '.join(g.name for g in hw.gpus) or 'none'}  CPU threads: {hw.cpu_threads}")
        click.echo("")
        click.echo(_status_text(s) if s else "engine: not running")
        if not watch:
            return
        time.sleep(1)


@experimental.command("latency")
@click.option("--json", "as_json", is_flag=True)
def exp_latency(as_json):
    """Latency percentiles from the running engine, else the newest pipeline benchmark."""
    from .paths import benchmarks_dir
    from .realtime.timeline import format_report

    s = _read_status() if _engine_pid() else None
    if s:
        _echo_json(s["latency"]) if as_json else click.echo(format_report(s["latency"]))
        return
    files = sorted(benchmarks_dir().glob("*-pipeline.json")) if benchmarks_dir().is_dir() else []
    if not files:
        raise click.ClickException("no running engine and no pipeline benchmark yet (`aivoice experimental benchmark`)")
    data = json.loads(files[-1].read_text())
    if as_json:
        _echo_json(data["pipeline"]["aggregate"])
    else:
        from .realtime.bench import format_bench

        click.echo(f"from {files[-1]}")
        click.echo(format_bench(data))


@experimental.command("benchmark")
@click.option("--clips", type=click.Path(path_type=Path, exists=True, file_okay=False), default=None,
              help="Directory of name.wav + name.txt pairs (default: generated with espeak-ng)")
@click.option("--mock", is_flag=True, help="Pipeline with scripted ASR + tone TTS (measures aivoice overhead only)")
@click.option("--asr/--no-asr", "do_asr", default=True)
@click.option("--tts/--no-tts", "do_tts", default=True)
@click.option("--pipeline/--no-pipeline", "do_pipe", default=True)
@click.option("--scenarios/--no-scenarios", default=True, help="Add quiet/noise/interruption variants")
@click.option("--consent-ack", is_flag=True)
def exp_benchmark(clips, mock, do_asr, do_tts, do_pipe, scenarios, consent_ack):
    """Measure ASR, TTS and end-to-end latency on this machine; writes JSON + text reports."""
    from .consent import require_consent
    from .realtime import bench

    require_consent(ack=consent_ack)
    data = bench.load_clip_dir(clips) if clips else bench.espeak_clips()
    if not data:
        raise click.ClickException("no test clips: pass --clips DIR or install espeak-ng")
    if scenarios:
        data = bench.derive_scenarios(data)
    click.echo(f"{len(data)} clips ({sorted({c['scenario'] for c in data})})")
    out: dict[str, Any] = {}
    if do_asr and not mock:
        out["asr"] = _bench_asr_all(data)
    if do_tts and not mock:
        try:
            from .tts import create_tts

            out["tts"] = bench.bench_tts(create_tts())
        except Exception as e:  # noqa: BLE001
            out["tts"] = {"backend": "?", "streaming": None, "remote": None, "ttfa_ms": {}, "errors": [str(e)]}
            click.echo(f"TTS benchmark skipped: {e}", err=True)
    if do_pipe:
        try:
            out["pipeline"] = bench.bench_pipeline(data[:6], mock=mock)
        except Exception as e:  # noqa: BLE001
            click.echo(f"pipeline benchmark failed: {e}", err=True)
    j, t = bench.save_report("pipeline" if do_pipe and "pipeline" in out else "components", out)
    click.echo(t.read_text())
    click.echo(f"reports: {j}\n         {t}")


def _bench_asr_all(clips, names: list[str] | None = None) -> dict[str, Any]:
    from .asr import available_backends, create_backend
    from .realtime import bench

    cfg = st.load()
    avail = available_backends()
    backends = {}
    mgr = None
    try:
        mgr = _manager()
    except Exception:  # noqa: BLE001
        pass
    for name in names or list(avail):
        if avail.get(name) is not None:
            click.echo(f"skip {name}: {avail[name]}", err=True)
            continue
        try:
            backends[name] = create_backend(name, settings=cfg["asr"], registry=mgr)
        except Exception as e:  # noqa: BLE001
            click.echo(f"skip {name}: {e}", err=True)
    if not backends:
        return {}
    return bench.bench_asr(backends, clips)


@experimental.group("asr")
def exp_asr():
    """ASR backend selection."""


@exp_asr.command("list")
def exp_asr_list():
    from .asr import available_backends, backend_config
    from .asr.selector import ASRSelector

    avail = available_backends()
    ms = ASRSelector().measurements({k: backend_config(k, st.get("asr")) for k in avail})
    click.echo(f"{'BACKEND':<18} {'STREAMING':<20} {'RUNTIME':<14} MEASURED (this machine)")
    streaming = {"parakeet": "chunked re-decode"}
    for k, why in avail.items():
        m = ms.get(k)
        meas = f"final p50 {m.final_p50_ms:.0f} ms, WER {m.wer:.1%}" if m and m.wer is not None else (
            f"final p50 {m.final_p50_ms:.0f} ms" if m else "-")
        click.echo(f"{k:<18} {streaming.get(k, 'native'):<20} {'installed' if why is None else 'missing':<14} {meas}")
    click.echo(f"\ncurrent: {st.get('asr.backend')} (policy {st.get('asr.policy')})")


@exp_asr.command("use")
@click.argument("name")
@click.option("--policy", type=click.Choice(["fastest", "accurate", "balanced", "low-memory"]), default=None,
              help="With NAME=auto: selection policy")
def exp_asr_use(name, policy):
    from .asr import BACKENDS

    if name != "auto" and name not in BACKENDS:
        raise click.ClickException(f"unknown backend {name!r}; choose from auto, {', '.join(BACKENDS)}")
    st.update("asr", backend=name, **({"policy": policy} if policy else {}))
    if name == "auto":
        from .realtime.build import choose_asr

        try:
            chosen, sel = choose_asr(st.load())
            click.echo(f"auto -> {chosen}: {sel.reason if sel else ''}")
        except Exception as e:  # noqa: BLE001
            click.echo(f"auto selected, but: {e}")
    else:
        st.update("asr", policy="manual")
        click.echo(f"asr backend = {name} (takes effect at the next utterance boundary / start)")


@exp_asr.command("benchmark")
@click.option("--clips", type=click.Path(path_type=Path, exists=True, file_okay=False), default=None)
@click.option("--backend", "names", multiple=True, help="Limit to these backends")
def exp_asr_benchmark(clips, names):
    """Benchmark every installed ASR backend on the same clips (feeds automatic selection)."""
    from .realtime import bench

    data = bench.load_clip_dir(clips) if clips else bench.espeak_clips()
    if not data:
        raise click.ClickException("no test clips: pass --clips DIR or install espeak-ng")
    data = bench.derive_scenarios(data)
    res = _bench_asr_all(data, list(names) or None)
    if not res:
        raise click.ClickException("no ASR backend could run (see skips above)")
    j, t = bench.save_report("asr", {"asr": res})
    click.echo(t.read_text())
    click.echo(f"reports: {j}")


@experimental.group("tts")
def exp_tts():
    """TTS backend (Fish Drama 3 / S2-Pro server, hosted API, local fallback)."""


@exp_tts.command("use")
@click.argument("backend", type=click.Choice(["auto", "fish-server", "fish-api", "local", "espeak", "piper", "command"]))
@click.option("--url", default=None, help="fish-speech server URL (fish-server)")
@click.option("--model", default=None, help="drama-3 | s2-pro (server: tag dialect; api: model header)")
def exp_tts_use(backend, url, model):
    vals: dict[str, Any] = {"backend": backend}
    if url:
        vals["server_url"] = url
    if model:
        vals["server_model" if backend != "fish-api" else "api_model"] = model
    st.update("tts", **vals)
    if url and not st.is_local_url(url) and not st.has_remote_consent(url):
        click.echo(f"note: {url} is remote; allow it with `aivoice experimental remote allow {url}`")
    if backend == "fish-api" and not st.has_remote_consent("fish-api"):
        click.echo("note: allow the hosted API with `aivoice experimental remote allow fish-api`")
    click.echo(f"tts backend = {backend}")


@exp_tts.command("health")
def exp_tts_health():
    from .tts import create_tts

    try:
        tts = create_tts()
    except Exception as e:  # noqa: BLE001
        raise _fail(e) from e
    ok, detail = tts.health()
    caps = tts.capabilities()
    click.echo(f"{caps.name}: {'ok' if ok else 'NOT OK'} {detail}")
    click.echo(f"streaming={caps.streaming} dialect={caps.dialect} remote={caps.remote} {caps.notes}")
    if not ok:
        raise click.ClickException("TTS backend unhealthy")


@exp_tts.command("say")
@click.argument("text")
@click.option("-o", "--out", type=click.Path(path_type=Path), default=None, help="Write WAV instead of timing only")
@click.option("--voice", default=None)
def exp_tts_say(text, out, voice):
    """Synthesize TEXT once with the configured backend (through the expression compiler)."""
    import numpy as np

    from .expression import ExpressionCompiler
    from .tts import create_tts
    from .tts_voices import load_profile

    tts = create_tts()
    v = load_profile(voice) if voice else None
    compiled = ExpressionCompiler(tts.capabilities().dialect).validate(text)
    if compiled.removed:
        click.echo(f"removed unsupported markup: {compiled.removed}", err=True)
    t0 = time.perf_counter()
    first = None
    parts, sr = [], 24000
    for ch in tts.synthesize(compiled.text, voice=v):
        first = first or (time.perf_counter() - t0) * 1000
        parts.append(ch.pcm)
        sr = ch.sample_rate
    total = (time.perf_counter() - t0) * 1000
    audio = np.concatenate(parts) if parts else np.zeros(0, np.float32)
    click.echo(f"{tts.capabilities().name}: first audio {first or 0:.0f} ms, total {total:.0f} ms, {audio.size / sr:.2f}s audio")
    if out:
        from .audio_io import write_wav

        write_wav(out, audio, sr)
        click.echo(str(out))


@experimental.group("preset")
def exp_preset():
    """Built-in and saved presets."""


@exp_preset.command("list")
def exp_preset_list():
    from .realtime.presets import PRESETS, all_presets

    cur = st.get("experimental.preset")
    for name in sorted(all_presets()):
        tag = "built-in" if name in PRESETS else "saved"
        click.echo(f"{'*' if name == cur else ' '} {name:<20} {tag}")


@exp_preset.command("use")
@click.argument("name")
def exp_preset_use(name):
    from .realtime.presets import apply_preset

    try:
        apply_preset(name)
    except KeyError as e:
        raise _fail(e) from e
    click.echo(f"preset {name} applied")


@exp_preset.command("save")
@click.argument("name")
def exp_preset_save(name):
    from .realtime.presets import save_preset

    try:
        save_preset(name)
    except ValueError as e:
        raise _fail(e) from e
    click.echo(f"saved current settings as preset {name}")


@exp_preset.command("show")
@click.argument("name")
def exp_preset_show(name):
    from .realtime.presets import all_presets

    p = all_presets().get(name)
    if p is None:
        raise click.ClickException(f"unknown preset {name!r}")
    _echo_json(p)


@exp_preset.command("delete")
@click.argument("name")
def exp_preset_delete(name):
    from .realtime.presets import delete_preset

    try:
        delete_preset(name)
    except KeyError as e:
        raise _fail(e) from e
    click.echo(f"deleted {name}")


@experimental.group("remote")
def exp_remote():
    """Opt in/out of sending transcripts or reference audio to a remote TTS."""


@exp_remote.command("allow")
@click.argument("destination")
@click.option("--yes", is_flag=True, help="Skip the confirmation prompt")
def exp_remote_allow(destination, yes):
    """DESTINATION is `fish-api` or a server URL (http://gpu-box:8080)."""
    msg = (
        f"Allow {destination} to receive the text aivoice recognises from your microphone "
        "(and the reference audio of local voice profiles)? Microphone audio itself is never sent."
    )
    if not yes and not click.confirm(msg):
        raise click.Abort()
    st.grant_remote_consent(destination, True)
    click.echo(f"remote processing allowed: {destination}")


@exp_remote.command("revoke")
@click.argument("destination")
def exp_remote_revoke(destination):
    st.grant_remote_consent(destination, False)
    click.echo(f"revoked: {destination}")


@exp_remote.command("list")
def exp_remote_list():
    c = st.get("privacy.remote_consent") or {}
    click.echo("\n".join(k for k, v in c.items() if v) or "(none: local processing only)")


@experimental.group("config")
def exp_config():
    """Read/write settings.json (no secrets accepted)."""


@exp_config.command("show")
def exp_config_show():
    _echo_json(st.load())


@exp_config.command("get")
@click.argument("key")
def exp_config_get(key):
    _echo_json(st.get(key))


@exp_config.command("set")
@click.argument("key")
@click.argument("value")
def exp_config_set(key, value):
    try:
        st.set_value(key, st.parse_value(value))
    except st.SettingsError as e:
        raise _fail(e) from e
    click.echo(f"{key} = {st.get(key)!r}")


@experimental.command("plan")
@click.option("--mock-4090", is_flag=True, help="Plan for an RTX 4090 / i9-14900K / 32 GB host")
def exp_plan(mock_4090):
    """Where each configured model would run, and whether it fits (estimates)."""
    from .hardware import detect, mock_rtx4090
    from .hub.registry import find_known
    from .resources import Requirement, ResourceScheduler, format_placement

    hw = mock_rtx4090() if mock_4090 else detect()
    cfg = st.load()
    reqs = []
    asr = cfg["asr"]["backend"] if cfg["asr"]["backend"] != "auto" else "moonshine-small"
    for key, prio, prefer in ((asr, 1, "cpu"), ((cfg["tts"].get("server_model") or "s2-pro").replace("auto", "s2-pro"), 3, "gpu")):
        k = find_known(key)
        if k:
            reqs.append(Requirement(k.key, k.est_vram_mib, k.est_ram_mib, prefer=prefer, priority=prio))
    if cfg["llm"]["enabled"]:
        k = find_known(cfg["llm"].get("model") or "qwen2.5-1.5b") or find_known("qwen2.5-1.5b")
        reqs.append(Requirement(k.key, k.est_vram_mib, k.est_ram_mib, prefer="cpu", priority=0))
    reqs.append(Requirement("drama-3 (reference)", find_known("drama-3").est_vram_mib, None, priority=-1))
    plan = ResourceScheduler(hw).plan(reqs)
    click.echo(f"host: {hw.cpu_model}, {hw.ram_total_mib // 1024} GiB RAM, GPU: {', '.join(g.name for g in hw.gpus) or 'none'}")
    click.echo(format_placement(plan))
    click.echo("\n(memory figures are estimates from the registry; Drama 3 is listed for reference only)")
    if cfg["tts"]["backend"] in ("fish-server", "auto") and not st.is_local_url(cfg["tts"]["server_url"]):
        click.echo(f"TTS runs remotely at {cfg['tts']['server_url']}: no local GPU needed for it")


@click.command("hardware")
@click.option("--json", "as_json", is_flag=True)
def hardware_cmd(as_json):
    """Detected CPU/GPU/RAM/disk and installed inference runtimes."""
    from .hardware import detect, fingerprint, format_hardware

    hw = detect(probe_cuda_runtime=True)
    if as_json:
        _echo_json({**hw.to_dict(), "fingerprint": fingerprint(hw)})
        return
    click.echo(format_hardware(hw))
    click.echo(f"Benchmark fingerprint {fingerprint(hw)}")


# =============================================================================
# llm
# =============================================================================


@click.group("llm")
def llm_group():
    """Optional local LLM transformation of recognised speech (llama.cpp / GGUF)."""


@llm_group.command("enable")
@click.option("--model", default=None, help="GGUF path or registered model (e.g. qwen2.5-1.5b)")
def llm_enable(model):
    vals: dict[str, Any] = {"enabled": True}
    if model:
        vals["model"] = model
    if st.get("llm.mode") == "verbatim":
        vals["mode"] = "hybrid"
    st.update("llm", **vals)
    click.echo(f"llm enabled (mode {st.get('llm.mode')}, model {st.get('llm.model')})")


@llm_group.command("disable")
def llm_disable():
    st.update("llm", enabled=False)
    click.echo("llm disabled (verbatim/rule-based paths only)")


@llm_group.command("model")
@click.argument("name", required=False)
def llm_model(name):
    if not name:
        click.echo(st.get("llm.model") or "(none)")
        return
    st.update("llm", model=name)
    click.echo(f"llm model = {name}")


@llm_group.command("style")
@click.argument("mode", required=False, type=click.Choice(["verbatim", "cleanup", "natural", "expressive", "character", "custom", "hybrid"]))
def llm_style(mode):
    if not mode:
        click.echo(st.get("llm.mode"))
        return
    st.update("llm", mode=mode)
    click.echo(f"llm mode = {mode}")


@llm_group.command("prompt")
@click.argument("text", required=False)
def llm_prompt(text):
    """Custom transformation instruction (used by mode=custom)."""
    if text is None:
        click.echo(st.get("llm.prompt") or "(none)")
        return
    st.update("llm", prompt=text)
    click.echo("custom prompt saved")


@llm_group.command("character")
@click.argument("text", required=False)
def llm_character(text):
    """Personality description for mode=character."""
    if text is None:
        click.echo(st.get("llm.character") or "(none)")
        return
    st.update("llm", character=text)
    click.echo("character saved")


@llm_group.command("intensity")
@click.argument("value", type=click.FloatRange(0, 1), required=False)
@click.option("--creativity", type=click.FloatRange(0, 1), default=None)
@click.option("--timeout-ms", type=int, default=None)
def llm_intensity(value, creativity, timeout_ms):
    vals: dict[str, Any] = {}
    if value is not None:
        vals["intensity"] = value
    if creativity is not None:
        vals["creativity"] = creativity
    if timeout_ms is not None:
        vals["timeout_ms"] = timeout_ms
    if vals:
        st.update("llm", **vals)
    llm = st.get("llm")
    click.echo(f"intensity {llm['intensity']}  creativity {llm['creativity']}  timeout {llm['timeout_ms']} ms")


@llm_group.command("test")
@click.argument("text")
def llm_test(text):
    """Run TEXT through the configured transformation and show the result and timing."""
    from .realtime.build import BuildReport, build_llm

    rep = BuildReport()
    eng = build_llm(st.load(), None, rep)
    for w in rep.warnings:
        click.echo(f"warning: {w}", err=True)
    if eng is None:
        click.echo(f"verbatim: {text}")
        return
    r = eng.transform(text)
    _echo_json(r.to_dict())
    eng.close()


# =============================================================================
# voice (TTS voice profiles)
# =============================================================================


@click.group("voice")
def voice_group():
    """TTS voice profiles for experimental mode (local voice, local voice id, hosted Fish id).

    Distinct from `aivoice voices`, which holds MeanVC2 conversion voices; a
    library voice can seed a profile with `voice add --from-library`.
    """


@voice_group.command("list")
@click.option("--json", "as_json", is_flag=True)
def voice_list(as_json):
    from .tts_voices import list_profiles

    items = list_profiles()
    if as_json:
        _echo_json([p.to_dict() for p in items])
        return
    active = st.get("voice.active")
    if not items:
        click.echo("(no voice profiles)  add one: aivoice voice add NAME --audio ref.wav --text '...' --i-have-rights")
        return
    for p in items:
        mark = "*" if active in (p.id, p.name) else " "
        target = p.reference_id or (Path(p.reference_audio).name if p.reference_audio else "")
        click.echo(f"{mark} {p.name:<24} {p.kind:<16} {target}")


@voice_group.command("add")
@click.argument("name")
@click.option("--audio", type=click.Path(path_type=Path, exists=True), default=None)
@click.option("--text", default=None, help="Exact transcript of the reference audio")
@click.option("--from-library", "library", default=None, help="Use a MeanVC2 library voice's reference clip")
@click.option("--character", default="", help="Personality prompt for CHARACTER mode")
@click.option("--i-have-rights", is_flag=True, help="Attest you may reproduce this voice")
def voice_add(name, audio, text, library, character, i_have_rights):
    """Create a LOCAL VOICE from authorized reference audio + transcript."""
    from .tts_voices import VoiceProfileError, add_local_voice, validate_reference

    src_voice = None
    if library:
        from .voices import load_voice

        try:
            lv = load_voice(library)
        except FileNotFoundError as e:
            raise _fail(e) from e
        if not lv.reference:
            raise click.ClickException(f"library voice {library!r} has no reference clip")
        audio, src_voice = Path(lv.reference), lv.id
    if audio is None or not text:
        raise click.ClickException("need --audio (or --from-library) and --text (exact words in the clip)")
    try:
        p = add_local_voice(name, audio, text, i_have_rights=i_have_rights, character=character, source_voice=src_voice)
    except VoiceProfileError as e:
        raise _fail(e) from e
    for issue in validate_reference(Path(p.reference_audio), p.reference_text):
        click.echo(f"note: {issue}", err=True)
    click.echo(f"created local voice {p.name} ({p.id})")


@voice_group.command("add-id")
@click.argument("name")
@click.option("--reference-id", required=True)
@click.option("--server", "server_url", default=None, help="fish-speech server (default: tts.server_url)")
@click.option("--register", is_flag=True, help="Upload --audio/--text to the server as this reference id")
@click.option("--audio", type=click.Path(path_type=Path, exists=True), default=None)
@click.option("--text", default=None)
@click.option("--i-have-rights", is_flag=True)
def voice_add_id(name, reference_id, server_url, register, audio, text, i_have_rights):
    """Create a LOCAL VOICE ID (a reference saved on your fish-speech server)."""
    from .tts_voices import VoiceProfileError, add_local_voice_id

    url = server_url or st.get("tts.server_url")
    if register and not st.is_local_url(url) and not st.has_remote_consent(url):
        raise click.ClickException(f"{url} is remote; allow it first: aivoice experimental remote allow {url}")
    try:
        p = add_local_voice_id(name, reference_id, server_url=url, i_have_rights=i_have_rights, audio=audio, text=text, register=register)
    except (VoiceProfileError, Exception) as e:  # noqa: BLE001
        raise _fail(e) from e
    click.echo(f"created local voice id {p.name} -> {p.reference_id} @ {p.server_url}")


@voice_group.command("add-hosted")
@click.argument("name")
@click.option("--reference-id", required=True, help="fish.audio voice model id")
@click.option("--check", is_flag=True, help="Look the id up through the official API")
@click.option("--i-have-rights", is_flag=True)
def voice_add_hosted(name, reference_id, check, i_have_rights):
    """Create a HOSTED FISH AUDIO ID profile (works only with tts.backend=fish-api)."""
    from .tts_voices import VoiceProfileError, add_hosted

    notes = ""
    if check:
        from .tts.fish import FishAPIBackend

        try:
            info = FishAPIBackend().get_voice(reference_id)
            notes = f"hosted title: {info.get('title')!r}, state: {info.get('state')}"
        except Exception as e:  # noqa: BLE001
            raise _fail(e) from e
    try:
        p = add_hosted(name, reference_id, i_have_rights=i_have_rights, notes=notes)
    except VoiceProfileError as e:
        raise _fail(e) from e
    click.echo(f"created hosted profile {p.name} -> {p.reference_id}")
    click.echo("note: hosted ids cannot be used by a local Drama 3 server (see docs/voice-profiles.md)")


@voice_group.command("remove")
@click.argument("name")
@click.option("--server-delete", is_flag=True, help="Also delete a local voice id from its server")
@click.option("--yes", is_flag=True)
def voice_remove(name, server_delete, yes):
    from .tts_voices import load_profile, remove_profile

    try:
        p = load_profile(name)
    except FileNotFoundError as e:
        raise _fail(e) from e
    if not yes and not click.confirm(f"Remove voice profile {p.name!r}?"):
        raise click.Abort()
    if server_delete and p.kind == "local_voice_id":
        from .tts.fish import FishServerClient

        try:
            FishServerClient(p.server_url).delete_reference(p.reference_id)
        except Exception as e:  # noqa: BLE001
            click.echo(f"warning: server delete failed: {e}", err=True)
    remove_profile(p.id)
    if st.get("voice.active") in (p.id, p.name):
        st.set_value("voice.active", None)
    click.echo(f"removed {p.name}")


@voice_group.command("use")
@click.argument("name")
def voice_use(name):
    from .tts_voices import load_profile

    try:
        p = load_profile(name)
    except FileNotFoundError as e:
        raise _fail(e) from e
    st.set_value("voice.active", p.id)
    backend = st.get("tts.backend")
    if p.kind == "hosted_fish_id" and backend != "fish-api":
        click.echo("note: hosted ids need `aivoice experimental tts use fish-api`")
    if p.kind in ("local_voice_id",) and backend == "fish-api":
        click.echo("note: local voice ids need a fish-speech server backend")
    click.echo(f"active voice = {p.name} ({p.kind})")


@voice_group.command("info")
@click.argument("name")
def voice_info(name):
    from .tts_voices import load_profile, verify_profile

    try:
        p = load_profile(name)
    except FileNotFoundError as e:
        raise _fail(e) from e
    _echo_json(p.to_dict())
    issues = verify_profile(p)
    click.echo("ok" if not issues else "\n".join(f"issue: {i}" for i in issues))


@voice_group.command("preview")
@click.argument("name")
@click.option("--text", default="Hi! This is a quick preview of my voice.")
@click.option("-o", "--out", type=click.Path(path_type=Path), default=None)
def voice_preview(name, text, out):
    """Synthesize a preview with the configured TTS and save it as WAV."""
    import numpy as np

    from .audio_io import write_wav
    from .tts import create_tts
    from .tts_voices import load_profile

    try:
        p = load_profile(name)
        tts = create_tts()
        parts, sr = [], 24000
        t0 = time.perf_counter()
        first = None
        for ch in tts.synthesize(text, voice=p):
            first = first or (time.perf_counter() - t0) * 1000
            parts.append(ch.pcm)
            sr = ch.sample_rate
    except Exception as e:  # noqa: BLE001
        raise _fail(e) from e
    dest = out or (p.dir() / "preview.wav")
    write_wav(Path(dest), np.concatenate(parts) if parts else np.zeros(1, np.float32), sr)
    click.echo(f"{dest}  (first audio {first or 0:.0f} ms via {tts.capabilities().name})")


@voice_group.command("export")
@click.argument("name")
@click.argument("dest", type=click.Path(path_type=Path))
def voice_export(name, dest):
    from .tts_voices import export_profile, load_profile

    p = load_profile(name)
    click.echo(str(export_profile(p, dest)))


@voice_group.command("import")
@click.argument("archive", type=click.Path(path_type=Path, exists=True))
@click.option("--name", default=None)
@click.option("--i-have-rights", is_flag=True)
def voice_import(archive, name, i_have_rights):
    from .tts_voices import VoiceProfileError, import_profile

    try:
        p = import_profile(archive, i_have_rights=i_have_rights, name=name)
    except (VoiceProfileError, Exception) as e:  # noqa: BLE001
        raise _fail(e) from e
    click.echo(f"imported {p.name} ({p.kind})")


# =============================================================================
# credentials
# =============================================================================


@click.group("credentials")
def credentials_group():
    """API keys / tokens (environment or system keyring; never settings.json)."""


@credentials_group.command("status")
def cred_status():
    from .credentials import describe

    for k, v in describe().items():
        click.echo(f"{k:<22} {v}")


@credentials_group.command("set")
@click.argument("service", type=click.Choice(["fish-audio"]))
def cred_set(service):
    from .credentials import CredentialError, store_fish_api_key

    value = click.prompt("Fish Audio API key", hide_input=True)
    try:
        where = store_fish_api_key(value.strip())
    except CredentialError as e:
        raise _fail(e) from e
    click.echo(f"stored in {where}")


@credentials_group.command("delete")
@click.argument("service", type=click.Choice(["fish-audio"]))
def cred_delete(service):
    from .credentials import delete_fish_api_key

    delete_fish_api_key()
    click.echo("deleted from keyring (environment variables are untouched)")


# =============================================================================
# models (Hugging Face manager) - added to the existing `models` group
# =============================================================================


def _size(n) -> str:
    from .hub.manager import fmt_bytes

    return fmt_bytes(n)


@click.command("search")
@click.argument("query", required=False)
@click.option("--author", default=None)
@click.option("--category", type=click.Choice(["asr", "tts", "llm", "vad", "audio"]), default=None)
@click.option("--library", default=None, help="e.g. nemo, transformers, gguf")
@click.option("--language", default=None)
@click.option("--license", "license_", default=None)
@click.option("--quant", default=None, help="gguf | int4 | fp8 | ...")
@click.option("--sort", type=click.Choice(["downloads", "likes", "last_modified", "trending_score"]), default="downloads")
@click.option("--limit", default=20, show_default=True)
@click.option("--json", "as_json", is_flag=True)
def models_search(query, author, category, library, language, license_, quant, sort, limit, as_json):
    """Search the Hugging Face Hub (network)."""
    try:
        rows = _manager().search(query, author=author, category=category, library=library, language=language,
                                 license=license_, quantization=quant, sort=sort, limit=limit)
    except Exception as e:  # noqa: BLE001
        raise _fail(e) from e
    if as_json:
        _echo_json([r.to_dict() for r in rows])
        return
    click.echo(f"{'REPO':<52} {'DOWNLOADS':>10} {'PIPELINE':<28} NOTES")
    for r in rows:
        notes = []
        if r.gated:
            notes.append("gated")
        if r.known:
            notes.append(f"aivoice:{r.known}")
        click.echo(f"{r.repo_id[:52]:<52} {r.downloads or 0:>10} {(r.pipeline_tag or '-')[:28]:<28} {' '.join(notes)}")


@click.command("info")
@click.argument("repo_id")
@click.option("--revision", default=None)
@click.option("--files", "show_files", is_flag=True)
@click.option("--json", "as_json", is_flag=True)
def models_info(repo_id, revision, show_files, as_json):
    """Metadata, compatibility and risks for a repository (nothing is downloaded or executed)."""
    from .hub.registry import find_known

    known = find_known(repo_id)
    if known and known.repo_id is None:
        _echo_json(known.__dict__) if as_json else _print_known(known)
        return
    try:
        d = _manager().info(repo_id, revision=revision)
    except Exception as e:  # noqa: BLE001
        if known:
            click.echo(f"(Hub unavailable: {e}; showing aivoice's registry entry)\n", err=True)
            _print_known(known)
            return
        raise _fail(e) from e
    if as_json:
        _echo_json(d.to_dict())
        return
    a = d.assessment
    click.echo(f"repo        {d.repo_id}")
    click.echo(f"revision    {d.commit}")
    click.echo(f"license     {d.license or 'unknown'}")
    click.echo(f"gated       {d.gated or False}")
    click.echo(f"pipeline    {d.pipeline_tag or '-'}   library {d.library_name or '-'}")
    click.echo(f"params      {d.params if d.params else 'not reported by the Hub'}" + (" (verified: safetensors metadata)" if d.params else ""))
    click.echo(f"size        {_size(d.total_bytes)} in {len(d.files)} files (verified: Hub file metadata)")
    click.echo(f"runtime     {a.runtime or 'unknown'}   adapter {a.adapter or 'none (inspect only)'}   quant {a.quantization or '-'}")
    click.echo(f"loadable    {'yes' if a.usable else 'no'}")
    for r in a.reasons:
        click.echo(f"  - {r}")
    for w in a.warnings:
        click.echo(f"  ! {w}")
    if d.known:
        k = d.known
        click.echo(f"aivoice     {k.key}: VRAM ~{k.est_vram_mib or '-'} MiB, RAM ~{k.est_ram_mib or '-'} MiB (estimates)")
        if k.notes:
            click.echo(f"            {k.notes}")
    click.echo(f"installed   {'yes @ ' + d.installed['commit'][:10] if d.installed and d.installed.get('commit') else 'no'}")
    if show_files:
        for f in d.files:
            click.echo(f"  {_size(f.size):>12}  {f.path}")


def _print_known(k) -> None:
    click.echo(f"{k.key}  [{k.category}] runtime={k.runtime} adapter={k.adapter}")
    click.echo(f"  source   {k.repo_id or k.source}")
    click.echo(f"  params   {k.params}   license {k.license}")
    click.echo(f"  memory   VRAM ~{k.est_vram_mib or '-'} MiB, RAM ~{k.est_ram_mib or '-'} MiB (estimates)")
    for fact, src in k.facts.items():
        click.echo(f"  fact     {fact}  [{src}]")
    if k.notes:
        click.echo(f"  notes    {k.notes}")


@click.command("download")
@click.argument("repo_id")
@click.option("--revision", default=None, help="Branch, tag or commit to pin")
@click.option("--include", "patterns", multiple=True, help="Only files matching (glob); default from registry")
@click.option("--yes", is_flag=True, help="Confirm large downloads")
@click.option("--dry-run", is_flag=True, help="Show the plan only")
def models_download(repo_id, revision, patterns, yes, dry_run):
    """Download (resumable) + verify a snapshot into the Hugging Face cache, pinned by commit."""
    from .hub.manager import ConfirmationRequired, DownloadCancelled
    from .hub.registry import find_known

    mgr = _manager()
    known = find_known(repo_id)
    if known and known.repo_id is None:
        try:
            e = mgr.download(repo_id, confirm=yes)
        except Exception as ex:  # noqa: BLE001
            raise _fail(ex) from ex
        click.echo(f"{known.key}: ready at {e['path']}")
        return
    try:
        plan = mgr.plan(repo_id, revision=revision, allow_patterns=list(patterns) or None)
    except Exception as e:  # noqa: BLE001
        raise _fail(e) from e
    click.echo(f"{plan.repo_id} @ {plan.commit or plan.revision}")
    click.echo(f"  files     {len(plan.files)}" + (f" matching {plan.allow_patterns}" if plan.allow_patterns else ""))
    click.echo(f"  total     {_size(plan.total_bytes)}   to download {_size(plan.download_bytes)}")
    click.echo(f"  disk free {_size(plan.disk_free_bytes)}")
    for w in plan.warnings:
        click.echo(f"  ! {w}")
    if dry_run:
        return
    if known and known.est_vram_mib and known.est_ram_mib is None:
        from .hardware import detect

        gpus = detect().gpus
        if not gpus or max(g.total_mib for g in gpus) < known.est_vram_mib:
            click.echo(f"  ! {known.key} needs ~{known.est_vram_mib // 1024} GiB of GPU memory; this machine "
                       f"{'has none' if not gpus else 'has ' + str(max(g.total_mib for g in gpus) // 1024) + ' GiB'}. "
                       "Downloading it here only makes sense to serve it from a bigger GPU later.")
    last = [0.0]

    def progress(p):
        now = time.monotonic()
        if now - last[0] < 0.25:
            return
        last[0] = now
        tot = p["total"] or plan.download_bytes or 0
        pct = f"{100 * p['done'] / tot:5.1f}%" if tot else ""
        eta = f"eta {p['eta_s']:.0f}s" if p.get("eta_s") else ""
        click.echo(f"\r  {pct} {_size(p['done'])} @ {_size(p['speed'])}/s {eta}   ", nl=False, err=True)

    cancel = threading.Event()
    prev = signal.signal(signal.SIGINT, lambda *a: cancel.set())
    try:
        entry = mgr.download(repo_id, revision=revision, allow_patterns=list(patterns) or None, confirm=yes,
                             progress=progress, cancel=cancel, plan=plan)
    except ConfirmationRequired as e:
        raise _fail(e) from e
    except DownloadCancelled as e:
        raise click.ClickException("cancelled; partial files are kept and the next download resumes") from e
    except Exception as e:  # noqa: BLE001
        raise _fail(e) from e
    finally:
        signal.signal(signal.SIGINT, prev)
    click.echo("", err=True)
    click.echo(f"installed {entry['repo_id']} @ {entry['commit'][:10] if entry.get('commit') else '?'} -> {entry['path']}")
    click.echo(f"verified: {entry.get('verified')}" + (f"  problems: {entry.get('problems')}" if entry.get("problems") else ""))
    if not entry.get("adapter"):
        click.echo("note: no aivoice adapter for this repository; it is installed but cannot be loaded by aivoice")


@click.command("installed")
@click.option("--json", "as_json", is_flag=True)
def models_installed(as_json):
    """Models in aivoice's registry (Hub snapshots, imports, runtime-managed)."""
    reg = _manager().installed()
    if as_json:
        _echo_json(reg)
        return
    if not reg:
        click.echo("(none) - try: aivoice models known")
        return
    click.echo(f"{'MODEL':<44} {'CAT':<5} {'RUNTIME':<20} {'REV':<11} {'VERIFIED':<9} SIZE")
    for k, e in reg.items():
        click.echo(f"{k[:44]:<44} {(e.get('category') or '-'):<5} {(e.get('runtime') or '-'):<20} "
                   f"{(e.get('commit') or '-')[:10]:<11} {str(e.get('verified')):<9} {_size(e.get('size_bytes'))}")


@click.command("known")
@click.option("--category", type=click.Choice(["asr", "tts", "llm", "vad"]), default=None)
def models_known(category):
    """Models aivoice has adapters for, with what is verified vs estimated."""
    from .hub.registry import KNOWN_MODELS

    for k in KNOWN_MODELS:
        if category and k.category != category:
            continue
        mem = f"VRAM~{k.est_vram_mib or '-'} RAM~{k.est_ram_mib or '-'} MiB"
        click.echo(f"{k.key:<16} {k.category:<4} {k.runtime:<19} {(k.repo_id or k.source)[:44]:<44} {mem}")


@click.command("loaded")
def models_loaded():
    """Models held by the running experimental engine."""
    s = _read_status() if _engine_pid() else None
    if not s:
        click.echo("no engine running; models are only held in memory by `aivoice experimental start`")
        return
    click.echo(f"ASR  {s['asr']['backend']}")
    click.echo(f"TTS  {s['tts']['name']} ({'remote ' + str(s['tts']['destination']) if s['tts']['remote'] else 'local'})")
    click.echo(f"LLM  {s['llm']['backend'] or 'none'} (mode {s['llm']['mode']})")


def _entry_or_fail(mgr, name):
    e = mgr.get_entry(name)
    if not e:
        raise click.ClickException(f"not installed: {name} (aivoice models download {name})")
    return e


def _lifecycle():
    from .hub.lifecycle import LoadedModel, ModelLifecycleManager

    lm = ModelLifecycleManager()

    def asr_loader(entry, device):
        from .asr import create_backend

        name = entry.get("key") or ("parakeet" if entry.get("adapter") == "parakeet" else entry.get("repo_id"))
        cfg = {**st.get("asr"), "parakeet_device": "cuda" if device.startswith("cuda") else "cpu"}
        be = create_backend(name, settings=cfg, registry=_manager())
        be.load_model()

        def warm():
            import numpy as np

            from .asr.base import streaming_view

            t0 = time.perf_counter()
            v = streaming_view(be)
            v.transcribe_stream(np.zeros(16000, np.float32), 16000)
            v.finalize_utterance()
            return (time.perf_counter() - t0) * 1000

        return LoadedModel(name, be, device, 0, unload=be.unload_model, warmup=warm)

    def llm_loader(entry, device):
        from .llm import LlamaCppBackend, resolve_model_path

        path = resolve_model_path(entry.get("repo_id") or entry.get("key"), _manager())
        if not path:
            raise RuntimeError("no .gguf file in the snapshot")
        be = LlamaCppBackend(path, n_gpu_layers=-1 if device.startswith("cuda") else 0)

        def warm():
            t0 = time.perf_counter()
            be.complete("Reply OK.", "hi", max_tokens=4, temperature=0.0, schema=None, cancel=threading.Event())
            return (time.perf_counter() - t0) * 1000

        return LoadedModel(path, be, device, 0, unload=be.close, warmup=warm)

    def fish_loader(entry, device):
        from .tts.fish import FishServerBackend

        b = FishServerBackend(st.get("tts.server_url"))
        ok, detail = b.health()
        if not ok:
            raise RuntimeError(
                f"{entry.get('repo_id')} is served by a separate fish-speech process; start it "
                f"(see docs/drama3.md). Server check: {detail}"
            )
        return LoadedModel(entry.get("repo_id"), b, "server", 0, warmup=lambda: b.warmup() or 0.0)

    def silero_loader(entry, device):
        from .vad import SileroVAD

        v = SileroVAD()
        return LoadedModel("silero-vad", v, "cpu", 0)

    lm.register_loader("moonshine", asr_loader)
    lm.register_loader("parakeet", asr_loader)
    lm.register_loader("llama-cpp", llm_loader)
    lm.register_loader("fish-server", fish_loader)
    lm.register_loader("silero-onnx", silero_loader)
    return lm


@click.command("load")
@click.argument("name")
@click.option("--device", default=None, help="cpu | cuda:0 (default: scheduler decides)")
def models_load(name, device):
    """Validate + load + warm up NAME in this process, report timings, then release it.

    The experimental engine loads its own models at start; this command proves a
    model is loadable on this machine and measures it.
    """
    mgr = _manager()
    entry = _entry_or_fail(mgr, name)
    if entry.get("category") == "tts" and entry.get("runtime") == "fish-speech-server":
        click.echo("note: Fish models run in a separate fish-speech server; this checks that server.")
    lm = _lifecycle()
    try:
        m = lm.load(name, entry, device=device)
    except Exception as e:  # noqa: BLE001
        raise _fail(e) from e
    click.echo(f"READY  {name} on {m.device}  load {m.load_ms or 0:.0f} ms  warmup {m.warmup_ms or 0:.0f} ms  reserved ~{m.reserved_mib} MiB")
    lm.shutdown()
    click.echo("released (one-shot process)")


@click.command("unload")
@click.argument("name")
def models_unload(name):
    """Models are released when the process holding them stops; for the engine: `experimental stop`."""
    if _engine_pid():
        click.echo("the running engine holds its models until `aivoice experimental stop` (or switch with `experimental asr use`)")
    else:
        click.echo(f"{name}: not loaded (no engine running)")


@click.command("remove")
@click.argument("name")
@click.option("--keep-files", is_flag=True, help="Only unregister; leave the HF cache untouched")
@click.option("--yes", is_flag=True)
def models_remove(name, keep_files, yes):
    """Unregister NAME and delete its cache revision (never shared/unmanaged files)."""
    mgr = _manager()
    e = _entry_or_fail(mgr, name)
    if not yes and not click.confirm(f"Remove {name} ({_size(e.get('size_bytes'))})?"):
        raise click.Abort()
    try:
        r = mgr.remove(name, delete_files=not keep_files)
    except Exception as ex:  # noqa: BLE001
        raise _fail(ex) from ex
    click.echo(f"removed {r['removed']}; freed {_size(r['freed_bytes'])}" + (f" ({r['kept_files_reason']})" if r["kept_files_reason"] else ""))


@click.command("update")
@click.argument("name")
@click.option("--yes", is_flag=True)
@click.option("--remove-old", is_flag=True, help="Delete the previous revision after the new one verifies")
def models_update(name, yes, remove_old):
    """Move a Hub model to the newest commit of its requested revision (keeps the old one until verified)."""
    mgr = _manager()
    try:
        old, new = mgr.check_update(name)
        if old == new:
            click.echo(f"{name} is up to date ({(old or '')[:10]})")
            return
        click.echo(f"{name}: {(old or '?')[:10]} -> {(new or '?')[:10]}")
        e = mgr.update(name, confirm=yes, remove_old=remove_old)
    except Exception as ex:  # noqa: BLE001
        raise _fail(ex) from ex
    click.echo(f"updated -> {e.get('commit', '')[:10]} verified={e.get('verified')}")


@click.command("verify")
@click.argument("name")
@click.option("--quick", is_flag=True, help="Sizes only, skip SHA-256")
def models_verify(name, quick):
    """Check sizes and LFS SHA-256 of an installed snapshot against the recorded Hub manifest."""
    try:
        r = _manager().verify(name, hash_files=not quick)
    except Exception as e:  # noqa: BLE001
        raise _fail(e) from e
    click.echo(f"{name}: {'ok' if r.ok else 'FAILED'} ({r.checked} files, {r.hashed} hashed)")
    for p in r.problems:
        click.echo(f"  {p}")
    if not r.ok:
        raise click.ClickException("verification failed")


@click.command("benchmark")
@click.argument("name")
@click.option("--clips", type=click.Path(path_type=Path, exists=True, file_okay=False), default=None)
def models_benchmark(name, clips):
    """Benchmark an installed ASR/TTS model on this machine."""
    from .hub.registry import find_known
    from .realtime import bench

    k = find_known(name)
    cat = k.category if k else (_manager().get_entry(name) or {}).get("category")
    if cat == "asr":
        data = bench.load_clip_dir(clips) if clips else bench.espeak_clips()
        if not data:
            raise click.ClickException("no clips: pass --clips DIR or install espeak-ng")
        res = _bench_asr_all(data, [k.key if k else name])
        j, t = bench.save_report("asr", {"asr": res})
    elif cat == "tts":
        from .tts import create_tts

        j, t = bench.save_report("tts", {"tts": bench.bench_tts(create_tts())})
    else:
        raise click.ClickException(f"no benchmark for category {cat!r}")
    click.echo(t.read_text())


@click.command("cache")
@click.option("--json", "as_json", is_flag=True)
def models_cache(as_json):
    """Hugging Face cache usage (shared with other tools; aivoice-managed entries marked)."""
    try:
        r = _manager().cache_report()
    except Exception as e:  # noqa: BLE001
        raise _fail(e) from e
    if as_json:
        _echo_json(r)
        return
    click.echo(f"cache {r['cache_dir']}  total {_size(r['size'])}")
    for repo in r["repos"]:
        click.echo(f"  {'*' if repo['managed'] else ' '} {_size(repo['size']):>10}  {repo['repo_id']}  ({len(repo['revisions'])} rev)")
    click.echo("  (* = registered by aivoice)")


@click.command("cleanup")
@click.option("--dry-run/--execute", default=True, show_default=True)
def models_cleanup(dry_run):
    """Delete unpinned old revisions of aivoice-managed repos. Other repos are never touched."""
    try:
        r = _manager().cleanup(dry_run=dry_run)
    except Exception as e:  # noqa: BLE001
        raise _fail(e) from e
    for c in r["candidates"]:
        click.echo(f"  {'would delete' if dry_run else 'deleted'} {c['repo_id']}@{c['commit'][:10]} ({_size(c['size'])})")
    if not r["candidates"]:
        click.echo("nothing to clean")
    if r["left_alone"]:
        click.echo(f"left alone (not managed by aivoice): {len(r['left_alone'])} repos")


@click.command("import")
@click.argument("path", type=click.Path(path_type=Path, exists=True))
@click.option("--category", type=click.Choice(["asr", "tts", "llm", "vad"]), default=None)
@click.option("--name", default=None)
def models_import(path, category, name):
    """Register an already-downloaded model directory or file (never copied, never deleted)."""
    try:
        e = _manager().import_path(path, category=category, name=name)
    except Exception as ex:  # noqa: BLE001
        raise _fail(ex) from ex
    click.echo(f"imported {e['key']}: runtime={e['runtime']} adapter={e['adapter'] or 'none (inspect only)'}")
    for w in e.get("warnings", []):
        click.echo(f"  ! {w}")


@click.command("default")
@click.argument("category", type=click.Choice(["asr", "tts", "llm", "vad"]))
@click.argument("name", required=False)
def models_default(category, name):
    """Show or set the default model for a category (persists in settings)."""
    if not name:
        click.echo(st.get(f"defaults.{category}") or "(none)")
        return
    st.set_value(f"defaults.{category}", name)
    if category == "asr":
        from .asr import BACKENDS

        if name in BACKENDS:
            st.update("asr", backend=name, policy="manual")
    elif category == "llm":
        st.update("llm", model=name)
    elif category == "tts" and name in ("drama-3", "s2-pro", "fishaudio/drama-3", "fishaudio/s2-pro"):
        st.update("tts", server_model=name.split("/")[-1])
    click.echo(f"default {category} = {name}")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(0)
