"""Enumerate PipeWire / Pulse / ALSA audio devices."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class AudioDevice:
    id: str
    name: str
    direction: str  # input | output | duplex | unknown
    backend: str  # pipewire | pulse | alsa | mock


def _run(cmd: list[str]) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=False)
        return r.stdout or ""
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return ""


def list_pipewire() -> list[AudioDevice]:
    if not shutil.which("pw-cli") and not shutil.which("wpctl"):
        return []
    out: list[AudioDevice] = []
    text = _run(["wpctl", "status"])
    if text:
        section = None
        for line in text.splitlines():
            if "Audio" in line and "Sinks" in line:
                section = "output"
            elif "Sources" in line:
                section = "input"
            m = re.search(r"(\d+)\.\s+(.+?)(?:\s+\[|$)", line)
            if m and section in ("input", "output"):
                out.append(
                    AudioDevice(
                        id=m.group(1),
                        name=m.group(2).strip(),
                        direction=section,
                        backend="pipewire",
                    )
                )
        if out:
            return out
    # Fallback: pw-dump is heavy; keep empty if wpctl missing details
    return out


def list_pulse() -> list[AudioDevice]:
    if not shutil.which("pactl"):
        return []
    out: list[AudioDevice] = []
    for kind, direction in (("sources", "input"), ("sinks", "output")):
        text = _run(["pactl", "list", "short", kind])
        for line in text.splitlines():
            parts = line.split("\t")
            if len(parts) >= 2:
                out.append(
                    AudioDevice(id=parts[0], name=parts[1], direction=direction, backend="pulse")
                )
    return out


def list_alsa() -> list[AudioDevice]:
    out: list[AudioDevice] = []
    text = _run(["arecord", "-l"])
    for m in re.finditer(r"card\s+(\d+):.*?\[(.+?)\].*?device\s+(\d+):.*?\[(.+?)\]", text, re.I):
        out.append(
            AudioDevice(
                id=f"hw:{m.group(1)},{m.group(3)}",
                name=f"{m.group(2)} — {m.group(4)}",
                direction="input",
                backend="alsa",
            )
        )
    text = _run(["aplay", "-l"])
    for m in re.finditer(r"card\s+(\d+):.*?\[(.+?)\].*?device\s+(\d+):.*?\[(.+?)\]", text, re.I):
        out.append(
            AudioDevice(
                id=f"hw:{m.group(1)},{m.group(3)}",
                name=f"{m.group(2)} — {m.group(4)}",
                direction="output",
                backend="alsa",
            )
        )
    return out


def list_devices() -> list[AudioDevice]:
    if os.environ.get("AIVOICE_MOCK_DEVICES") == "1":
        return [
            AudioDevice("mock0", "Mock Mic", "input", "mock"),
            AudioDevice("mock1", "Mock Speaker", "output", "mock"),
            AudioDevice("virtmic0", "Mock Virtual Mic", "output", "mock"),
        ]
    devices = list_pipewire()
    if devices:
        return devices
    devices = list_pulse()
    if devices:
        return devices
    return list_alsa()


def format_devices(devices: list[AudioDevice] | None = None) -> str:
    devices = devices if devices is not None else list_devices()
    if not devices:
        return (
            "(no audio devices found via PipeWire/Pulse/ALSA)\n"
            "Hints: install pipewire / pipewire-pulse / alsa-utils; "
            "check `wpctl status` or `pactl list short sources`."
        )
    lines = [f"{d.backend}\t{d.direction}\t{d.id}\t{d.name}" for d in devices]
    return "\n".join(lines)


def detect_audio_stack() -> str:
    if shutil.which("pw-cli") or shutil.which("wpctl"):
        return "pipewire"
    if shutil.which("pactl"):
        return "pulse"
    if shutil.which("arecord"):
        return "alsa"
    return "none"
