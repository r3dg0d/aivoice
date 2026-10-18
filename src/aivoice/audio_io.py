"""Audio file helpers + optional sounddevice streaming."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _read_pcm_wav(path: Path) -> tuple[np.ndarray, int]:
    """Plain PCM WAV via the stdlib (used when soundfile is not installed)."""
    import wave

    try:
        with wave.open(str(path), "rb") as w:
            ch, width, sr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
            raw = w.readframes(n)
    except (wave.Error, EOFError) as e:
        raise RuntimeError(f"{path}: not a PCM WAV file ({e}); pip install soundfile for other formats") from e
    if width == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif width == 1:
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise RuntimeError(f"{path}: {8 * width}-bit WAV needs soundfile (pip install soundfile)")
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, int(sr)


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf  # type: ignore
    except ImportError:
        return _read_pcm_wav(Path(path))
    data, sr = sf.read(str(path), always_2d=False)
    if getattr(data, "ndim", 1) > 1:
        data = data.mean(axis=1)
    return np.asarray(data, dtype=np.float32), int(sr)


def write_wav(path: Path, pcm: np.ndarray, sample_rate: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import soundfile as sf  # type: ignore
    except ImportError:
        import wave

        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(int(sample_rate))
            w.writeframes((np.clip(np.asarray(pcm, np.float32), -1, 1) * 32767).astype("<i2").tobytes())
        return
    sf.write(str(path), pcm, sample_rate)


def have_sounddevice() -> bool:
    try:
        import sounddevice  # type: ignore  # noqa: F401

        return True
    except ImportError:
        return False
