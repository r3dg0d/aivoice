"""MeanVC2 integration via vendor checkout subprocess / optional import."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from ..paths import models_dir, vendor_dir
from ..presets import Mode


class MeanVC2Backend:
    """Prefer subprocess to upstream CLI when full import stack is heavy."""

    name = "meanvc2"

    def __init__(self, mode: Mode, device: str = "cuda") -> None:
        self.mode = mode
        self.device = device
        self._ref: Path | None = None
        self.vendor = vendor_dir()
        if not self.vendor.is_dir():
            raise RuntimeError(
                f"MeanVC2 vendor missing at {self.vendor}. "
                "Run: aivoice models install meanvc2 --yes"
            )

    def set_reference(self, wav_path: Path) -> None:
        if not wav_path.is_file():
            raise FileNotFoundError(wav_path)
        self._ref = wav_path

    def convert_file(self, source: Path, dest: Path) -> None:
        if self._ref is None:
            raise RuntimeError("set_reference() required")
        infer = self.vendor / "src" / "infer" / "infer_e2e.py"
        rt = self.vendor / "runtime" / "run_rt.py"
        model_flag = "40ms" if self.mode.meanvc2_model == "40ms" else "120ms"
        if infer.is_file():
            cmd = [
                sys.executable,
                str(infer),
                "--model",
                model_flag,
                "--source-wav",
                str(source),
                "--target-wav",
                str(self._ref),
                "--output-wav",
                str(dest),
                "--steps",
                str(self.mode.steps),
            ]
        elif rt.is_file():
            cmd = [
                sys.executable,
                str(rt),
                "--mode",
                "file",
                "--input",
                str(source),
                "--output",
                str(dest),
                "--model",
                model_flag,
            ]
            # reference wiring varies by upstream version — pass env
            os.environ["MEANVC2_TARGET_WAV"] = str(self._ref)
        else:
            raise RuntimeError(
                f"Neither infer_e2e.py nor run_rt.py found under {self.vendor}. "
                "Re-run models install / check clone."
            )
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.vendor) + os.pathsep + env.get("PYTHONPATH", "")
        # Point at HF weights cache if present
        hf = models_dir() / "meanvc2" / "hf"
        if hf.is_dir():
            env["MEANVC2_CKPT_DIR"] = str(hf)
        r = subprocess.run(cmd, cwd=str(self.vendor), env=env, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(
                f"MeanVC2 failed (exit {r.returncode}):\n{r.stderr[-2000:] or r.stdout[-2000:]}"
            )

    def convert_chunk(self, pcm: np.ndarray, sample_rate: int) -> tuple[np.ndarray, float]:
        """Streaming chunk path — uses run_rt when available; else raises with guidance."""
        _ = sample_rate
        if self._ref is None:
            raise RuntimeError("set_reference() required")
        t0 = time.perf_counter()
        rt = self.vendor / "runtime" / "run_rt.py"
        if not rt.is_file():
            raise RuntimeError(
                "Live chunk conversion requires MeanVC2 runtime/run_rt.py. "
                "File mode works via convert_file(); for live, complete models install "
                "and upstream realtime deps (see STATUS.md)."
            )
        # Chunk-level Python API is upstream-internal; document subprocess live path
        # via pipeline using run_rt --mode realtime instead of per-chunk here.
        raise RuntimeError(
            "In-process chunk API not exposed by upstream. "
            "Use aivoice live (spawns run_rt --mode realtime) or aivoice file."
        )
