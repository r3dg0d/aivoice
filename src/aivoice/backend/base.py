from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np


class VoiceBackend(Protocol):
    name: str

    def set_reference(self, wav_path: Path) -> None: ...

    def convert_chunk(self, pcm: np.ndarray, sample_rate: int) -> tuple[np.ndarray, float]:
        """Return (pcm_out, process_ms)."""
        ...

    def convert_file(self, source: Path, dest: Path) -> None: ...
