"""PipeWire / Pulse virtual microphone helpers."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass


@dataclass
class VirtualMicInfo:
    name: str
    backend: str
    hint: str


def create_null_sink(name: str = "aivoice") -> VirtualMicInfo:
    """Create a PipeWire-Pulse null sink that apps can use as a mic (remap source)."""
    if shutil.which("pactl"):
        # Create null sink + remap source
        sink = f"{name}_sink"
        source = f"{name}_mic"
        subprocess.run(
            ["pactl", "load-module", "module-null-sink", f"sink_name={sink}", f"sink_properties=device.description={name}"],
            check=False,
            capture_output=True,
        )
        subprocess.run(
            [
                "pactl",
                "load-module",
                "module-remap-source",
                f"master={sink}.monitor",
                f"source_name={source}",
                f"source_properties=device.description={name} Virtual Mic",
            ],
            check=False,
            capture_output=True,
        )
        return VirtualMicInfo(
            name=source,
            backend="pulse",
            hint=f"Set app input to '{name} Virtual Mic' (source {source}). "
            f"Play converted audio into sink '{sink}'.",
        )
    if shutil.which("pw-cli"):
        return VirtualMicInfo(
            name=name,
            backend="pipewire",
            hint=(
                "PipeWire detected but auto graph not configured. "
                "Example: use `pw-loopback` / Helvum / qpwgraph to route aivoice output "
                "into a null sink monitor used as mic."
            ),
        )
    raise RuntimeError(
        "Neither pactl nor pw-cli found. Install PipeWire or PulseAudio to use virtualmic."
    )
