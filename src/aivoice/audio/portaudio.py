"""Import sounddevice on systems where ctypes cannot find PortAudio (NixOS).

``sounddevice`` locates libportaudio with ``ctypes.util.find_library``, which on NixOS searches neither
LD_LIBRARY_PATH nor the Nix store, so the import fails with "PortAudio library not found" even when the
library is on LD_LIBRARY_PATH. Look there (and in $AIVOICE_PORTAUDIO) ourselves and point find_library at it.
"""

from __future__ import annotations

import ctypes.util
import os
from pathlib import Path


def find_portaudio() -> str | None:
    explicit = os.environ.get("AIVOICE_PORTAUDIO")
    if explicit and Path(explicit).is_file():
        return explicit
    for d in filter(None, os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)):
        for name in ("libportaudio.so.2", "libportaudio.so"):
            p = Path(d) / name
            if p.is_file():
                return str(p)
    return None


def import_sounddevice():
    try:
        import sounddevice  # type: ignore

        return sounddevice
    except OSError as e:
        if "PortAudio" not in str(e):
            raise
        lib = find_portaudio()
        if lib is None:
            raise OSError("PortAudio library not found (set AIVOICE_PORTAUDIO or put libportaudio on LD_LIBRARY_PATH)") from e
    orig = ctypes.util.find_library
    ctypes.util.find_library = lambda name: lib if name == "portaudio" else orig(name)
    try:
        import sounddevice  # type: ignore

        return sounddevice
    finally:
        ctypes.util.find_library = orig
