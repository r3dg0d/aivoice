"""Reference-audio preparation for MeanVC2 profiles (light-touch)."""

from __future__ import annotations

import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PrepResult:
    output: Path
    duration_s: float | None
    sample_rate: int | None
    notes: list[str]


def prepare_reference(
    src: Path,
    dest: Path,
    *,
    target_sr: int = 16000,
    max_seconds: float = 30.0,
) -> PrepResult:
    """Copy/resample reference audio. Prefer ffmpeg when present; else wav copy.

    Does not invent denoising/VAD when tools are missing — keep clean audio intact.
    """
    notes: list[str] = []
    dest.parent.mkdir(parents=True, exist_ok=True)
    src = Path(src)
    if not src.is_file():
        raise FileNotFoundError(src)

    if shutil.which("ffmpeg"):
        # light: resample + trim silence lightly + cap duration
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-ac",
            "1",
            "-ar",
            str(target_sr),
            "-t",
            str(max_seconds),
            "-af",
            "silenceremove=start_periods=1:start_silence=0.3:start_threshold=-40dB:"
            "stop_periods=1:stop_silence=0.4:stop_threshold=-40dB,loudnorm=I=-16:TP=-1.5:LRA=11",
            str(dest),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            notes.append("ffmpeg prep failed; falling back to raw copy")
            shutil.copy2(src, dest)
        else:
            notes.append("ffmpeg: mono resample + light silence trim + loudnorm")
    else:
        notes.append("ffmpeg not found; copied original reference unchanged")
        shutil.copy2(src, dest)

    dur, sr = _probe_wav(dest)
    if dur is not None and dur < 0.5:
        notes.append("warning: reference shorter than 0.5s — MeanVC2 quality may suffer")
    if dur is not None and dur > 60:
        notes.append("warning: long reference; consider a cleaner 5–20s speech clip")
    return PrepResult(output=dest, duration_s=dur, sample_rate=sr, notes=notes)


def _probe_wav(path: Path) -> tuple[float | None, int | None]:
    try:
        with wave.open(str(path), "rb") as w:
            sr = w.getframerate()
            frames = w.getnframes()
            return frames / float(sr), sr
    except Exception:  # noqa: BLE001
        return None, None


def score_reference(path: Path) -> list[str]:
    """Heuristic quality notes — no fabricated scores."""
    notes: list[str] = []
    dur, sr = _probe_wav(path)
    if dur is None:
        notes.append("could not probe as WAV (ok if other format)")
        return notes
    if dur < 1.0:
        notes.append("short clip")
    if sr and sr < 16000:
        notes.append(f"low sample rate ({sr})")
    size = path.stat().st_size
    if size < 10_000:
        notes.append("very small file — may be silence or heavily compressed")
    return notes
