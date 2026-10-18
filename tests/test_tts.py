import struct
import time
import wave
from pathlib import Path

import numpy as np
import pytest

from aivoice import settings as st
from aivoice.tts import FallbackTTS, RemoteNotAllowed, TTSError, create_tts
from aivoice.tts.fish import FishAPIBackend, FishServerBackend, FishServerClient, build_request
from aivoice.tts.mock import ToneTTS
from aivoice.tts.wavstream import WavStreamDecoder, WavStreamError
from aivoice.tts_voices import add_hosted, add_local_voice, add_local_voice_id
from fakes.fish_server import FakeFishServer, wav_header


def ref_wav(path: Path, seconds=4.0, rate=16000) -> Path:
    t = np.arange(int(rate * seconds)) / rate
    pcm = (0.2 * np.sin(2 * np.pi * 150 * t) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return path


# --- wav streaming ----------------------------------------------------------


def test_wav_stream_decoder_byte_by_byte_with_placeholder_size():
    pcm = (np.arange(1000, dtype="<i2") * 10).tobytes()
    data = wav_header(24000, data_size=0) + pcm
    dec = WavStreamDecoder()
    out = [dec.feed(data[i : i + 7]) for i in range(0, len(data), 7)]
    x = np.concatenate(out)
    assert dec.sample_rate == 24000 and x.size == 1000
    assert np.allclose(x[:3], np.array([0, 10, 20]) / 32768.0)


def test_wav_stream_decoder_extra_chunks_24bit_and_raw():
    body = b"\x01\x00\x00\xff\xff\x7f"
    hdr = (b"RIFF" + struct.pack("<I", 0) + b"WAVE" + b"LIST" + struct.pack("<I", 4) + b"abcd"
           + b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, 8000, 24000, 3, 24) + b"data" + struct.pack("<I", len(body)))
    dec = WavStreamDecoder()
    x = dec.feed(hdr + body + b"trailing-junk")
    assert dec.sample_rate == 8000 and x.size == 2 and x[1] > 0.99
    raw = WavStreamDecoder(raw_pcm_rate=44100)
    assert raw.feed(b"\x00\x40\x00\xc0").tolist() == [0.5, -0.5]
    with pytest.raises(WavStreamError):
        WavStreamDecoder().feed(b"ID3\x04" + b"\x00" * 20)


# --- request building --------------------------------------------------------


def test_build_request_voice_kinds(tmp_path):
    lv = add_local_voice("Me", ref_wav(tmp_path / "r.wav"), "this is my voice", i_have_rights=True)
    req = build_request("hi", voice=lv)
    assert req["references"][0]["text"] == "this is my voice" and req["references"][0]["audio"][:4] == b"RIFF"
    assert req["streaming"] is True and req["use_memory_cache"] == "on"
    hosted = add_hosted("H", "802e3bc2b27e49c2995d23ef70e6ac89", i_have_rights=True)
    with pytest.raises(TTSError, match="only works with tts.backend=fish-api"):
        build_request("hi", voice=hosted)
    api = build_request("hi", voice=hosted, hosted=True, sample_rate=44100, fmt="pcm")
    assert api["reference_id"] == hosted.reference_id and "streaming" not in api and api["chunk_length"] <= 300


# --- local / remote fish-speech server -----------------------------------------


def test_server_streaming_synthesis_and_model_detection():
    with FakeFishServer(ttfa_ms=50, chunk_ms=40, model="drama-3") as srv:
        be = FishServerBackend(srv.url)
        ok, _ = be.health()
        assert ok and be.capabilities().dialect == "drama3" and not be.capabilities().remote
        t0 = time.perf_counter()
        chunks = list(be.synthesize("one two three four five six"))
        assert len(chunks) > 5, "audio must arrive incrementally, not as one blob"
        assert all(c.sample_rate == 44100 for c in chunks)
        total = sum(c.pcm.size for c in chunks)
        assert abs(total - 44100 * 2) < 2000
        assert srv.requests[-1]["text"] == "one two three four five six"
        assert (time.perf_counter() - t0) < 5


def test_server_cancellation_stops_reading():
    import threading

    with FakeFishServer(chunk_ms=20) as srv:
        be = FishServerBackend(srv.url, model="s2-pro")
        cancel = threading.Event()
        n = 0
        for _ in be.synthesize("a " * 60, cancel=cancel):
            n += 1
            if n == 3:
                cancel.set()
        assert n == 3


def test_server_local_voice_id_registration(tmp_path):
    with FakeFishServer() as srv:
        prof = add_local_voice_id("Narrator", "narrator_01", server_url=srv.url, i_have_rights=True,
                                  audio=ref_wav(tmp_path / "n.wav"), text="hello narrator", register=True)
        assert FishServerClient(srv.url).list_references() == ["narrator_01"]
        be = FishServerBackend(srv.url)
        assert list(be.synthesize("hi", voice=prof))
        assert srv.requests[-1]["reference_id"] == "narrator_01" and srv.requests[-1]["references"] == []


def test_remote_server_requires_consent():
    with pytest.raises(RemoteNotAllowed, match="remote allow"):
        FishServerBackend("http://gpu-box.lan:8080")
    st.grant_remote_consent("http://gpu-box.lan:8080")
    be = FishServerBackend("http://gpu-box.lan:8080")
    assert be.capabilities().remote


def test_server_error_is_reported():
    with FakeFishServer(fail_tts=True) as srv:
        with pytest.raises(TTSError, match="HTTP 500"):
            list(FishServerBackend(srv.url, model="s2-pro").synthesize("x"))


# --- hosted API ----------------------------------------------------------------


def test_hosted_api_requires_consent_and_key(monkeypatch):
    with pytest.raises(RemoteNotAllowed):
        FishAPIBackend()
    st.grant_remote_consent("fish-api")
    with pytest.raises(TTSError, match="FISH_API_KEY"):
        FishAPIBackend()


def test_hosted_api_headers_pcm_stream_and_redaction(monkeypatch):
    st.grant_remote_consent("fish-api")
    key = "secret-key-123456"
    with FakeFishServer(require_key=key, rate=44100) as srv:
        be = FishAPIBackend(api_key=key, base_url=srv.url, model="drama-3")
        hosted = add_hosted("H", "802e3bc2b27e49c2995d23ef70e6ac89", i_have_rights=True)
        chunks = list(be.synthesize("hello there", voice=hosted))
        assert chunks and chunks[0].sample_rate == 44100
        h = {k.lower(): v for k, v in srv.headers[-1].items()}  # HTTP header names are case-insensitive
        assert h["model"] == "drama-3" and h["authorization"] == f"Bearer {key}"
        assert srv.requests[-1]["reference_id"] == hosted.reference_id and srv.requests[-1]["format"] == "pcm"
        assert be.health()[0]
        bad = FishAPIBackend(api_key="wrong-key-999999", base_url=srv.url)
        with pytest.raises(TTSError) as ei:
            list(bad.synthesize("x"))
        assert "wrong-key-999999" not in str(ei.value)


# --- fallback / factory --------------------------------------------------------


def test_fallback_only_before_first_audio_and_never_remote():
    fb = FallbackTTS(ToneTTS(fail=True), ToneTTS())
    assert list(fb.synthesize("hi")) and fb.fallbacks == 1

    class Remote(ToneTTS):
        def capabilities(self):
            c = super().capabilities()
            c.remote = True
            return c

    with pytest.raises(TTSError, match="remote fallback"):
        FallbackTTS(ToneTTS(), Remote())


def test_create_tts_auto_falls_back_to_local_when_server_down(monkeypatch):
    import shutil

    if not shutil.which("espeak-ng"):
        pytest.skip("espeak-ng not installed")
    tts = create_tts({**st.load()["tts"], "backend": "auto", "server_url": "http://127.0.0.1:9"})
    assert tts.name == "local:espeak-ng"
    chunks = list(tts.synthesize("[laughs] hello world"))
    assert chunks and chunks[0].pcm.size > 1000


def test_create_tts_fish_api_never_silent(monkeypatch):
    with pytest.raises(RemoteNotAllowed):
        create_tts({**st.load()["tts"], "backend": "fish-api"})
