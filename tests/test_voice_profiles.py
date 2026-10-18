import wave

import numpy as np
import pytest

from aivoice.tts_voices import (
    VoiceProfileError,
    add_hosted,
    add_local_voice,
    add_local_voice_id,
    export_profile,
    import_profile,
    list_profiles,
    load_profile,
    remove_profile,
    verify_profile,
)


def ref(path, seconds=5.0):
    t = np.arange(int(16000 * seconds)) / 16000
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((0.2 * np.sin(2 * np.pi * 150 * t) * 32767).astype("<i2").tobytes())
    return path


def test_attestation_required(tmp_path):
    with pytest.raises(VoiceProfileError, match="--i-have-rights"):
        add_local_voice("X", ref(tmp_path / "a.wav"), "words", i_have_rights=False)
    with pytest.raises(VoiceProfileError):
        add_hosted("H", "802e3bc2b27e49c2995d23ef70e6ac89", i_have_rights=False)
    assert list_profiles() == []


def test_local_voice_lifecycle(tmp_path):
    p = add_local_voice("My Voice", ref(tmp_path / "a.wav"), "the exact words", i_have_rights=True, character="calm narrator")
    assert p.kind == "local_voice" and p.consent["attested"] and p.character == "calm narrator"
    assert load_profile("my voice").id == p.id
    assert verify_profile(p) in ([], [i for i in verify_profile(p) if "soundfile" in i])
    p2 = add_local_voice("My Voice", ref(tmp_path / "b.wav"), "more words", i_have_rights=True)
    assert p2.id != p.id  # never overwrites
    remove_profile(p.id)
    with pytest.raises(FileNotFoundError):
        load_profile(p.id)


def test_reference_transcript_required(tmp_path):
    with pytest.raises(VoiceProfileError, match="transcript"):
        add_local_voice("X", ref(tmp_path / "a.wav"), "  ", i_have_rights=True)


def test_ids_validated():
    with pytest.raises(VoiceProfileError):
        add_hosted("H", "not a valid id!", i_have_rights=True)
    with pytest.raises(VoiceProfileError):
        add_local_voice_id("L", "bad/id", server_url="http://127.0.0.1:8080", i_have_rights=True)
    p = add_local_voice_id("L", "narrator 01", server_url="http://127.0.0.1:8080", i_have_rights=True)
    assert p.kind == "local_voice_id" and verify_profile(p) == []


def test_register_needs_audio_and_text():
    with pytest.raises(VoiceProfileError, match="--audio and --text"):
        add_local_voice_id("L", "x", server_url="http://127.0.0.1:8080", i_have_rights=True, register=True)
    assert list_profiles() == []


def test_export_import_roundtrip_requires_fresh_attestation(tmp_path):
    p = add_local_voice("Exp", ref(tmp_path / "a.wav"), "hello there", i_have_rights=True)
    z = export_profile(p, tmp_path / "exp")
    assert z.suffix == ".zip"
    with pytest.raises(VoiceProfileError):
        import_profile(z, i_have_rights=False)
    q = import_profile(z, i_have_rights=True, name="Imported")
    assert q.kind == "local_voice" and q.reference_text == "hello there"
    assert q.reference_bytes()[:4] == b"RIFF"
