"""Hardware / runtime capability detection.

Everything here is a cheap probe: no model is loaded and torch is only imported
when it is already installed. The fingerprint is what benchmark caches key on,
so a driver, runtime, or GPU change invalidates old measurements.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass, field

from .paths import cache_home

# Python packages that back a runtime adapter. Detected with find_spec, so
# nothing heavy is imported just to answer "is it installed?".
RUNTIME_PACKAGES = {
    "nemo": "nemo_toolkit",
    "moonshine_voice": "moonshine-voice",
    "onnxruntime": "onnxruntime",
    "llama_cpp": "llama-cpp-python",
    "torch": "torch",
    "transformers": "transformers",
    "huggingface_hub": "huggingface_hub",
    "sounddevice": "sounddevice",
    "soundfile": "soundfile",
    "fish_speech": "fish-speech",
    "msgpack": "msgpack",
    "ormsgpack": "ormsgpack",
    "keyring": "keyring",
}


@dataclass
class GpuInfo:
    index: int
    name: str
    total_mib: int
    free_mib: int
    driver: str = ""
    compute_capability: str = ""


@dataclass
class HardwareInfo:
    cpu_model: str
    cpu_arch: str
    cpu_threads: int
    ram_total_mib: int
    ram_available_mib: int
    gpus: list[GpuInfo] = field(default_factory=list)
    cuda_runtime: str | None = None
    runtimes: dict[str, str | None] = field(default_factory=dict)
    disk_free_mib: int = 0
    os: str = ""

    @property
    def has_cuda_gpu(self) -> bool:
        return bool(self.gpus)

    def to_dict(self) -> dict:
        return asdict(self)


def _meminfo() -> tuple[int, int]:
    total = avail = 0
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                key, _, rest = line.partition(":")
                kib = int(rest.strip().split()[0])
                if key == "MemTotal":
                    total = kib // 1024
                elif key == "MemAvailable":
                    avail = kib // 1024
    except (OSError, ValueError, IndexError):
        pass
    return total, avail


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def query_nvidia_smi() -> list[GpuInfo]:
    """GPUs as reported by nvidia-smi (works without torch)."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        r = subprocess.run(
            [
                exe,
                "--query-gpu=index,name,memory.total,memory.free,driver_version,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    return parse_nvidia_smi(r.stdout)


def parse_nvidia_smi(text: str) -> list[GpuInfo]:
    gpus: list[GpuInfo] = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            gpus.append(
                GpuInfo(
                    index=int(parts[0]),
                    name=parts[1],
                    total_mib=int(float(parts[2])),
                    free_mib=int(float(parts[3])),
                    driver=parts[4] if len(parts) > 4 else "",
                    compute_capability=parts[5] if len(parts) > 5 else "",
                )
            )
        except ValueError:
            continue
    return gpus


def _cuda_runtime() -> str | None:
    if importlib.util.find_spec("torch") is None:
        return None
    try:
        import torch  # type: ignore

        return getattr(torch.version, "cuda", None)
    except Exception:  # noqa: BLE001
        return None


def runtime_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for module, dist in RUNTIME_PACKAGES.items():
        if importlib.util.find_spec(module) is None:
            out[module] = None
            continue
        try:
            out[module] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            out[module] = "installed"
    return out


def has_runtime(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def detect(*, probe_cuda_runtime: bool = False) -> HardwareInfo:
    """Collect hardware facts. ``probe_cuda_runtime`` imports torch (slow) for its CUDA version."""
    if os.environ.get("AIVOICE_MOCK_HARDWARE") == "rtx4090":
        return mock_rtx4090()
    total, avail = _meminfo()
    try:
        cache_home().mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(cache_home()).free // (1024 * 1024)
    except OSError:
        free = 0
    return HardwareInfo(
        cpu_model=_cpu_model(),
        cpu_arch=platform.machine(),
        cpu_threads=os.cpu_count() or 1,
        ram_total_mib=total,
        ram_available_mib=avail,
        gpus=query_nvidia_smi(),
        cuda_runtime=_cuda_runtime() if probe_cuda_runtime else None,
        runtimes=runtime_versions(),
        disk_free_mib=free,
        os=f"{platform.system()} {platform.release()}",
    )


def mock_rtx4090() -> HardwareInfo:
    """The primary target machine, for planning tests on hosts without a GPU."""
    return HardwareInfo(
        cpu_model="Intel(R) Core(TM) i9-14900K",
        cpu_arch="x86_64",
        cpu_threads=32,
        ram_total_mib=32 * 1024,
        ram_available_mib=26 * 1024,
        gpus=[GpuInfo(0, "NVIDIA GeForce RTX 4090", 24564, 23800, "mock", "8.9")],
        cuda_runtime="12.x (mock)",
        runtimes=runtime_versions(),
        disk_free_mib=500 * 1024,
        os="Linux (mock)",
    )


def fingerprint(hw: HardwareInfo | None = None, *, extra: dict | None = None) -> str:
    """Stable hash of what makes a benchmark result valid: CPU, GPU, driver, runtime versions."""
    hw = hw or detect()
    material = {
        "cpu": hw.cpu_model,
        "threads": hw.cpu_threads,
        "gpus": [(g.name, g.total_mib, g.driver) for g in hw.gpus],
        "runtimes": hw.runtimes,
        "extra": extra or {},
    }
    return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()[:16]


def format_hardware(hw: HardwareInfo) -> str:
    lines = [
        f"CPU       {hw.cpu_model} ({hw.cpu_arch}, {hw.cpu_threads} threads)",
        f"RAM       {hw.ram_available_mib} / {hw.ram_total_mib} MiB available",
    ]
    if hw.gpus:
        for g in hw.gpus:
            lines.append(
                f"GPU {g.index}     {g.name}  {g.free_mib}/{g.total_mib} MiB free  "
                f"driver {g.driver or '?'}  cc {g.compute_capability or '?'}"
            )
    else:
        lines.append("GPU       none detected (nvidia-smi missing or no NVIDIA GPU)")
    if hw.cuda_runtime:
        lines.append(f"CUDA      {hw.cuda_runtime} (torch)")
    lines.append(f"Disk      {hw.disk_free_mib // 1024} GiB free under cache")
    present = [f"{k}={v}" for k, v in hw.runtimes.items() if v]
    lines.append("Runtimes  " + (", ".join(present) or "none of the optional runtimes installed"))
    return "\n".join(lines)
