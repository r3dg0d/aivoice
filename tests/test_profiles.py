from pathlib import Path

from aivoice.profiles import create_profile, list_profiles, load_profile


def test_profile_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    ref = tmp_path / "voice.wav"
    # minimal RIFF/WAV header-ish bytes — profile only copies file
    ref.write_bytes(b"RIFF" + b"\x00" * 36 + b"data" + b"\x00" * 16)
    p = create_profile("alice", ref, mode="balanced", notes="test")
    assert p.name == "alice"
    items = list_profiles()
    assert any(i.name == "alice" for i in items)
    loaded = load_profile("alice")
    assert loaded.mode == "balanced"
    assert Path(loaded.reference).is_file()
