"""Synthetic, rights-free evaluation material: espeak-ng formant voices.

Each "speaker" is an espeak-ng voice variant (e.g. ``en-us+m3``), a synthetic
voice that imitates no real person, so its audio can be generated, measured and
published freely. Formant speech is far from natural speech: results on it
validate *mechanics* (silence handling, preprocessing, joins), not model quality.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HARVARD = [
    "The birch canoe slid on the smooth planks.",
    "Glue the sheet to the dark blue background.",
    "It's easy to tell the depth of a well.",
    "These days a chicken leg is a rare dish.",
    "Rice is often served in round bowls.",
    "The juice of lemons makes fine punch.",
    "The box was thrown beside the parked truck.",
    "The hogs were fed chopped corn and garbage.",
    "Four hours of steady work faced us.",
    "A large size in stockings is hard to sell.",
    "The boy was there when the sun rose.",
    "A rod is used to catch pink salmon.",
]

SPEAKERS = {
    "synth-a": {"voice": "en-us+m3", "wpm": 165, "pitch": 45},
    "synth-b": {"voice": "en+f4", "wpm": 175, "pitch": 60},
    "synth-c": {"voice": "en-us+m7", "wpm": 150, "pitch": 35},
    "synth-d": {"voice": "en+f2", "wpm": 185, "pitch": 70},
}


@dataclass
class Utterance:
    speaker: str
    text: str
    audio: np.ndarray
    sr: int


def available() -> bool:
    return shutil.which("espeak-ng") is not None


def speak(text: str, speaker: str) -> tuple[np.ndarray, int]:
    from ..audio_io import read_wav

    v = SPEAKERS[speaker]
    with tempfile.TemporaryDirectory(prefix="aivoice-synth-") as td:
        f = Path(td) / "u.wav"
        subprocess.run(["espeak-ng", "-v", v["voice"], "-s", str(v["wpm"]), "-p", str(v["pitch"]), "-w", str(f), text],
                       check=True, capture_output=True, timeout=60)
        x, sr = read_wav(f)
    return x, sr


def corpus(speakers: list[str] | None = None, sentences: list[str] | None = None) -> list[Utterance]:
    out = []
    for s in speakers or list(SPEAKERS):
        for t in sentences or HARVARD:
            x, sr = speak(t, s)
            out.append(Utterance(s, t, x, sr))
    return out


def pad_silence(x: np.ndarray, sr: int, lead_s: float, trail_s: float, noise_db: float = -60.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    amp = 10 ** (noise_db / 20)
    lead = (rng.normal(0, amp, int(lead_s * sr))).astype(np.float32)
    trail = (rng.normal(0, amp, int(trail_s * sr))).astype(np.float32)
    return np.concatenate([lead, x, trail])
