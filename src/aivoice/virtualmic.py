"""PipeWire / Pulse virtual microphone helpers.

The virtual mic is a null sink (aivoice plays into it) plus a remap source on
its monitor (apps record from it, so Discord/OBS/VRChat list it as a normal
microphone). Works on PipeWire through pipewire-pulse and on PulseAudio.

* Idempotent: an existing sink/source with the same name is reused, never
  duplicated on every launch.
* Tracked: module ids aivoice loaded are recorded in the runtime dir, so
  :func:`cleanup_virtual_mic` unloads exactly those and nothing else.
* Feedback-safe: :func:`feedback_risk` refuses to capture from the virtual mic
  or its monitor (which would feed aivoice's own output back into ASR).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .paths import runtime_dir


@dataclass
class VirtualMicInfo:
    name: str
    backend: str
    hint: str
    sink: str = ""
    source: str = ""
    reused: bool = False
    module_ids: list[int] = field(default_factory=list)


def _pactl(*args: str, timeout: float = 5) -> subprocess.CompletedProcess:
    return subprocess.run(["pactl", *args], capture_output=True, text=True, timeout=timeout, check=False)


def _short(kind: str) -> list[list[str]]:
    r = _pactl("list", "short", kind)
    return [line.split("\t") for line in r.stdout.splitlines() if line.strip()]


def _names(kind: str) -> set[str]:
    return {row[1] for row in _short(kind) if len(row) > 1}


def _state_file(name: str) -> Path:
    return runtime_dir() / f"virtualmic-{name}.json"


def _load_state(name: str) -> dict:
    try:
        return json.loads(_state_file(name).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _save_state(name: str, data: dict) -> None:
    p = _state_file(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def sink_name(name: str) -> str:
    return f"{name}_sink"


def source_name(name: str) -> str:
    return f"{name}_mic"


def ensure_virtual_mic(name: str = "aivoice") -> VirtualMicInfo:
    """Create (or reuse) ``<name>_sink`` + ``<name>_mic``."""
    if not shutil.which("pactl"):
        if shutil.which("pw-cli"):
            return VirtualMicInfo(
                name=name,
                backend="pipewire",
                hint=(
                    "PipeWire without pipewire-pulse (pactl) detected. Install pipewire-pulse, or add the "
                    "persistent virtual mic from docs/nixos-pipewire.md to your PipeWire config."
                ),
            )
        raise RuntimeError("Neither pactl nor pw-cli found. Install PipeWire (with pipewire-pulse) or PulseAudio.")
    sink, source = sink_name(name), source_name(name)
    state = _load_state(name)
    loaded: list[int] = list(state.get("module_ids", []))
    reused = True
    if sink not in _names("sinks"):
        reused = False
        r = _pactl(
            "load-module",
            "module-null-sink",
            f"sink_name={sink}",
            "channel_map=mono",
            f"sink_properties=device.description={name}-output",
        )
        if r.returncode != 0:
            raise RuntimeError(f"could not create null sink: {r.stderr.strip() or r.stdout.strip()}")
        loaded.append(int(r.stdout.strip()))
    if source not in _names("sources"):
        reused = False
        r = _pactl(
            "load-module",
            "module-remap-source",
            f"master={sink}.monitor",
            f"source_name={source}",
            "channel_map=mono",
            f"source_properties=device.description={name}-virtual-mic",
        )
        if r.returncode != 0:
            raise RuntimeError(f"could not create virtual mic source: {r.stderr.strip() or r.stdout.strip()}")
        loaded.append(int(r.stdout.strip()))
    _save_state(name, {"module_ids": loaded, "sink": sink, "source": source})
    return VirtualMicInfo(
        name=source,
        backend="pulse" if not shutil.which("pw-cli") else "pipewire-pulse",
        hint=f"Set your app's input to '{name}-virtual-mic' (source {source}). aivoice plays into sink '{sink}'.",
        sink=sink,
        source=source,
        reused=reused,
        module_ids=loaded,
    )


def create_null_sink(name: str = "aivoice") -> VirtualMicInfo:
    """Backward-compatible name for :func:`ensure_virtual_mic`."""
    return ensure_virtual_mic(name)


def enable_monitor(name: str = "aivoice", target: str = "@DEFAULT_SINK@", latency_ms: int = 40) -> int:
    """Hear the virtual mic on your headphones (module-loopback). Use headphones: speakers feed back."""
    state = _load_state(name)
    if state.get("monitor_module"):
        return int(state["monitor_module"])
    r = _pactl(
        "load-module",
        "module-loopback",
        f"source={sink_name(name)}.monitor",
        f"sink={target}",
        f"latency_msec={latency_ms}",
        "source_dont_move=true",
    )
    if r.returncode != 0:
        raise RuntimeError(f"could not enable monitor: {r.stderr.strip()}")
    mid = int(r.stdout.strip())
    state["monitor_module"] = mid
    state.setdefault("module_ids", []).append(mid)
    _save_state(name, state)
    return mid


def disable_monitor(name: str = "aivoice") -> bool:
    state = _load_state(name)
    mid = state.pop("monitor_module", None)
    if mid is None:
        return False
    _pactl("unload-module", str(mid))
    state["module_ids"] = [m for m in state.get("module_ids", []) if m != mid]
    _save_state(name, state)
    return True


def cleanup_virtual_mic(name: str = "aivoice") -> list[int]:
    """Unload only the modules aivoice itself loaded for *name* (reverse order)."""
    state = _load_state(name)
    ids = list(state.get("module_ids", []))
    if not ids or not shutil.which("pactl"):
        _state_file(name).unlink(missing_ok=True)
        return []
    live = {int(row[0]) for row in _short("modules") if row and row[0].isdigit()}
    done = []
    for mid in reversed(ids):
        if mid in live:
            _pactl("unload-module", str(mid))
            done.append(mid)
    _state_file(name).unlink(missing_ok=True)
    return done


def feedback_risk(input_device: str | None, name: str = "aivoice") -> str | None:
    """Why capturing from *input_device* would loop aivoice's output back into itself, if it would."""
    if not input_device:
        return None
    dev = str(input_device)
    bad = {source_name(name), f"{sink_name(name)}.monitor"}
    if dev in bad or dev.endswith(f"{sink_name(name)}.monitor") or "virtual-mic" in dev and name in dev:
        return (
            f"input {dev!r} is aivoice's own virtual mic/monitor: that would feed synthesized speech back "
            "into recognition. Pick your real microphone."
        )
    return None


def status(name: str = "aivoice") -> dict:
    out = {"pactl": bool(shutil.which("pactl")), "pw-cli": bool(shutil.which("pw-cli")), "state": _load_state(name)}
    if out["pactl"]:
        out["sink_present"] = sink_name(name) in _names("sinks")
        out["source_present"] = source_name(name) in _names("sources")
    return out


def info_dict(info: VirtualMicInfo) -> dict:
    return asdict(info)
