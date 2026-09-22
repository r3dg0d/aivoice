"""Clear errors / debug passthrough when MeanVC2 is not installed."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np


INSTALL_HINT = (
    "MeanVC2 backend not ready. Run:\n"
    "  aivoice models list\n"
    "  aivoice models install meanvc2 --yes\n"
    "Then ensure torch + upstream deps are available (see STATUS.md).\n"
    "License note: upstream README claims Apache-2.0 but GitHub LICENSE file was missing — "
    "our wrapper is Apache-2.0; we do not relicense upstream."
)


class MissingBackend:
    name = "missing"

    def set_reference(self, wav_path: Path) -> None:
        _ = wav_path
        raise RuntimeError(INSTALL_HINT)

    def convert_chunk(self, pcm: np.ndarray, sample_rate: int) -> tuple[np.ndarray, float]:
        _ = pcm, sample_rate
        raise RuntimeError(INSTALL_HINT)

    def convert_file(self, source: Path, dest: Path) -> None:
        _ = source, dest
        raise RuntimeError(INSTALL_HINT)


class PassthroughBackend:
    """Debug: copies audio unchanged (explicit opt-in)."""

    name = "passthrough"

    def __init__(self) -> None:
        self._ref: Path | None = None

    def set_reference(self, wav_path: Path) -> None:
        if not wav_path.is_file():
            raise FileNotFoundError(wav_path)
        self._ref = wav_path

    def convert_chunk(self, pcm: np.ndarray, sample_rate: int) -> tuple[np.ndarray, float]:
        _ = sample_rate
        t0 = time.perf_counter()
        out = pcm.copy()
        return out, (time.perf_counter() - t0) * 1000

    def convert_file(self, source: Path, dest: Path) -> None:
        shutil.copy2(source, dest)
