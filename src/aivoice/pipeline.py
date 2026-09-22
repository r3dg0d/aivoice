"""Live / file voice conversion orchestration."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from .backend.factory import create_backend
from .metrics import MetricsTracker, format_metrics
from .paths import vendor_dir
from .presets import Mode, get_mode


def convert_file(
    source: Path,
    reference: Path,
    output: Path,
    *,
    mode: str = "balanced",
    device: str = "cuda",
    backend: str | None = None,
) -> None:
    be = create_backend(backend, mode=mode, device=device)
    be.set_reference(reference)
    be.convert_file(source, output)


def run_live_subprocess(
    reference: Path,
    *,
    mode: str = "balanced",
    device: str = "cuda",
) -> int:
    """Spawn MeanVC2 runtime/run_rt.py --mode realtime when vendor present."""
    vendor = vendor_dir()
    rt = vendor / "runtime" / "run_rt.py"
    if not rt.is_file():
        raise RuntimeError(
            f"Missing {rt}. Install: aivoice models install meanvc2 --yes"
        )
    m: Mode = get_mode(mode)
    model_flag = "40ms" if m.meanvc2_model == "40ms" else "120ms"
    cmd = [
        sys.executable,
        str(rt),
        "--mode",
        "realtime",
        "--model",
        model_flag,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(vendor) + os.pathsep + env.get("PYTHONPATH", "")
    env["MEANVC2_TARGET_WAV"] = str(reference)
    env["AIVOICE_DEVICE"] = device
    print(f"spawning: {' '.join(cmd)}", flush=True)
    print(f"reference={reference} mode={mode} device={device}", flush=True)
    return subprocess.call(cmd, cwd=str(vendor), env=env)


def benchmark_overhead(*, frames: int = 50, mode: str = "balanced") -> str:
    import numpy as np

    m = get_mode(mode)
    mt = MetricsTracker(sample_rate=m.sample_rate)
    chunk = np.zeros(int(m.sample_rate * m.chunk_ms / 1000), dtype=np.float32)
    for _ in range(frames):
        t0 = time.perf_counter()
        # simulate buffer copy overhead only
        _ = chunk.copy()
        ms = (time.perf_counter() - t0) * 1000
        audio_ms = float(m.chunk_ms)
        metrics = mt.record(process_ms=ms, audio_ms=audio_ms, buffer_depth=m.buffer_chunks)
    return format_metrics(metrics) + "\n(Note: full VC RTF needs models install + CUDA preferred)"
