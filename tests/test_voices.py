"""Voice library create/list/verify."""

from __future__ import annotations

import wave
from pathlib import Path

from aivoice import voices as voices_mod


def _wav(path: Path, frames: int = 16000):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * frames)


def test_create_list_verify(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    ref = tmp_path / "ref.wav"
    _wav(ref)
    v = voices_mod.create_from_reference("Test Voice", ref)
    assert v.engine == "meanvc2"
    assert v.status == "ready"
    listed = voices_mod.list_voices()
    assert any(x.id == v.id for x in listed)
    assert voices_mod.verify_voice(v.id) == []
    voices_mod.rename_voice(v.id, "Renamed")
    assert voices_mod.load_voice(v.id).display_name == "Renamed"
    voices_mod.remove_voice(v.id)
    assert voices_mod.list_voices() == []
