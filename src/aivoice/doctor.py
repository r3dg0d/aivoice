"""Environment / dependency diagnostics."""

from __future__ import annotations

import json
import shutil
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any

from .models import is_ready, list_models
from .paths import cache_home, data_home, downloads_dir, vendor_dir, voices_dir
from .providers import get_provider, list_providers


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


def run_doctor(*, check_network: bool = True) -> list[Check]:
    checks: list[Check] = []

    # MeanVC2 vendor + weights
    vendor = vendor_dir()
    checks.append(
        Check(
            "MeanVC2 vendor",
            (vendor / "runtime" / "run_rt.py").is_file() or (vendor / "src" / "infer" / "infer_e2e.py").is_file(),
            str(vendor),
        )
    )
    checks.append(Check("MeanVC2 catalog", is_ready("meanvc2"), "aivoice models install meanvc2 --yes"))

    # CUDA / torch
    try:
        import torch

        cuda = bool(torch.cuda.is_available())
        name = torch.cuda.get_device_name(0) if cuda else "cpu"
        checks.append(Check("PyTorch", True, f"{torch.__version__}"))
        checks.append(Check("CUDA", cuda, name))
    except ImportError:
        checks.append(Check("PyTorch", False, "not installed"))
        checks.append(Check("CUDA", False, "needs PyTorch"))

    # PipeWire / pactl
    checks.append(Check("PipeWire/Pulse (pactl)", bool(shutil.which("pactl")), shutil.which("pactl") or "missing"))
    checks.append(Check("pw-cli", bool(shutil.which("pw-cli")), shutil.which("pw-cli") or "missing"))
    checks.append(Check("FFmpeg", bool(shutil.which("ffmpeg")), shutil.which("ffmpeg") or "optional for audio prep"))

    # Voice library paths
    try:
        voices_dir().mkdir(parents=True, exist_ok=True)
        downloads_dir().mkdir(parents=True, exist_ok=True)
        checks.append(Check("Voice library", True, str(voices_dir())))
        checks.append(Check("Download cache", True, str(downloads_dir())))
        checks.append(Check("Data home", True, str(data_home())))
        checks.append(Check("Cache home", True, str(cache_home())))
    except OSError as e:
        checks.append(Check("Voice library", False, str(e)))

    # Disk
    usage = shutil.disk_usage(str(data_home()))
    checks.append(
        Check(
            "Disk space",
            usage.free > 2 * 1024**3,
            f"{usage.free // (1024**3)} GiB free",
        )
    )

    # Providers
    for name, prov in list_providers().items():
        caps = prov.capabilities()
        detail = caps.notes[:120]
        ok = True
        if check_network and caps.search:
            try:
                # light HEAD to site
                if name == "voice-models":
                    req = urllib.request.Request(
                        "https://voice-models.com/",
                        method="HEAD",
                        headers={"User-Agent": "aivoice/0.2 doctor"},
                    )
                    with urllib.request.urlopen(req, timeout=10) as resp:
                        ok = 200 <= getattr(resp, "status", 200) < 400
                        detail = f"HTTP {getattr(resp, 'status', '?')} — search={caps.search} download={caps.download}"
            except Exception as e:  # noqa: BLE001
                ok = False
                detail = f"unreachable: {e}"
        checks.append(Check(f"provider:{name}", ok, detail))

    # Optional MeanVC2 component files if vendor present
    for label, rel in (
        ("WavLM/ECAPA (vendor)", "N/A — loaded by upstream at runtime"),
        ("Vocos (vendor)", "N/A — loaded by upstream at runtime"),
    ):
        checks.append(Check(label, True, rel))

    return checks


def format_doctor(checks: list[Check], *, as_json: bool = False) -> str:
    if as_json:
        return json.dumps([asdict(c) for c in checks], indent=2)
    lines = ["aivoice doctor", ""]
    for c in checks:
        mark = "✓" if c.ok else "✗"
        lines.append(f"{c.name:<22} {mark}  {c.detail}")
    return "\n".join(lines)
