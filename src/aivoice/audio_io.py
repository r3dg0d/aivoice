"""Audio file helpers + optional sounddevice streaming."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "soundfile required for wav I/O. pip install soundfile"
        ) from e
    data, sr = sf.read(str(path), always_2d=False)
    if getattr(data, "ndim", 1) > 1:
        data = data.mean(axis=1)
    return np.asarray(data, dtype=np.float32), int(sr)


def write_wav(path: Path, pcm: np.ndarray, sample_rate: int) -> None:
    import soundfile as sf  # type: ignore

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), pcm, sample_rate)


def have_sounddevice() -> bool:
    try:
        import sounddevice  # type: ignore  # noqa: F401

        return True
    except ImportError:
        return False
