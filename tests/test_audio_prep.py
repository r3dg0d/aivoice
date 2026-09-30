"""prepare_reference must keep the whole utterance, not just the first phrase."""

from __future__ import annotations

import math
import shutil
import struct
import wave
from pathlib import Path

import pytest

from aivoice.audio_prep import prepare_reference

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def _tone_pause_tone(path: Path, sr: int = 16000) -> float:
    """0.5 s silence, 1 s tone, 0.8 s pause, 1 s tone, 0.5 s silence. Returns speech+pause seconds."""

    def tone(sec):
        return [int(9000 * math.sin(2 * math.pi * 220 * i / sr)) for i in range(int(sec * sr))]

    silence = lambda sec: [0] * int(sec * sr)
    samples = silence(0.5) + tone(1.0) + silence(0.8) + tone(1.0) + silence(0.5)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return 1.0 + 0.8 + 1.0


def test_a_pause_inside_the_speech_does_not_truncate_the_reference(tmp_path):
    src, dest = tmp_path / "in.wav", tmp_path / "out.wav"
    expected = _tone_pause_tone(src)
    res = prepare_reference(src, dest)
    assert res.duration_s is not None
    # Both tones and the pause between them survive. The filter keeps a 0.3 s margin of
    # edge silence on each side, so: content + 2 * 0.3.
    assert res.duration_s == pytest.approx(expected + 0.6, abs=0.25)
    assert res.duration_s > 2.0, "cut at the first pause"


def test_edge_silence_is_still_trimmed(tmp_path):
    src, dest = tmp_path / "in.wav", tmp_path / "out.wav"
    _tone_pause_tone(src)  # 3.8 s in total, 2.8 s of it speech and pause
    res = prepare_reference(src, dest)
    assert res.duration_s < 3.8 - 0.2  # the 0.5 s edges were shortened
