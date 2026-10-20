"""Engines through the real worker protocol, against API-faithful fake packages (tests/fakes/fakepkgs)."""

import json
import shutil
import wave
from pathlib import Path

import numpy as np
import pytest

from aivoice.engines import (
    EngineUnavailable,
    SpeakerRef,
    SynthesisOptions,
    UnsupportedOption,
    get_engine,
    validate_options,
)
from aivoice.engines.registry import all_status, capability_table, names
from aivoice.engines.worker import WorkerClient

FAKES = Path(__file__).parent / "fakes" / "fakepkgs"


def wav(path: Path, seconds=3.0, sr=16000, lead=0.0):
    t = np.arange(int(sr * seconds)) / sr
    x = (0.3 * np.sin(2 * np.pi * 160 * t) * (1 + np.sin(2 * np.pi * 3 * t)) / 2).astype(np.float32)
    x = np.concatenate([np.zeros(int(lead * sr), np.float32), x])
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((x * 32767).astype("<i2").tobytes())
    return path


@pytest.fixture
def fakes(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(FAKES))
    monkeypatch.setenv("PYTHONPATH", str(FAKES))
    log = tmp_path / "fake.log"
    monkeypatch.setenv("AIVOICE_FAKE_LOG", str(log))
    snap = tmp_path / "VoxCPM2"
    snap.mkdir()
    return log, snap


def calls(log: Path):
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


def test_registry_and_capability_table():
    assert {"voxcpm2", "xvoice", "voicestar", "meanvc2", "fish-server", "fish-api", "espeak"} <= set(names())
    assert get_engine("vox-cpm2").capabilities().name == "voxcpm2"
    table = capability_table(all_status(check=False))
    assert "voxcpm2" in table and "48k" in table


def test_option_validation_is_explicit():
    caps = get_engine("xvoice").capabilities()
    ref = SpeakerRef("x", Path("ref.wav"))
    with pytest.raises(UnsupportedOption, match="style"):
        validate_options(caps, ref, SynthesisOptions(style_prompt="whisper"))
    with pytest.raises(UnsupportedOption, match="separate style reference"):
        validate_options(get_engine("voxcpm2").capabilities(), ref, SynthesisOptions(style_reference=Path("s.wav")))
    cleaned, warnings = validate_options(caps, ref, SynthesisOptions(style_prompt="whisper", allow_unsupported=True))
    assert cleaned.style_prompt is None and "ignored style_prompt" in warnings[0]
    with pytest.raises(UnsupportedOption, match="transcript"):
        validate_options(get_engine("voicestar").capabilities(), ref, SynthesisOptions())
    with pytest.raises(UnsupportedOption, match="cannot clone"):
        validate_options(get_engine("espeak").capabilities(), ref, SynthesisOptions())
    with pytest.raises(UnsupportedOption, match="pitch"):
        validate_options(get_engine("voxcpm2").capabilities(), ref, SynthesisOptions(pitch_shift=2.0))


def test_worker_protocol_tone():
    w = WorkerClient("tone")
    assert w.call("load")["sample_rate"] == 16000
    chunks = list(w.stream(text="a b c d"))
    assert len(chunks) > 3 and chunks[0][1] == 16000
    r = w.synthesize(text="a b")
    assert Path(r["path"]).is_file() and r["total_ms"] >= 0
    w.stop()
    assert not w.running


def test_worker_reports_errors():
    w = WorkerClient("tone")
    with pytest.raises(Exception, match="unknown op"):
        w.call("nope")
    w.stop()


def test_voxcpm2_transcript_free_style_and_ultimate(fakes, tmp_path):
    log, snap = fakes
    eng = get_engine("voxcpm2", model_path=snap)
    assert eng.available() is None
    ref = wav(tmp_path / "ref.wav")
    r = eng.synthesize("Hello there.", SpeakerRef("me", ref), SynthesisOptions(style_prompt="calm, warm", seed=3))
    assert r.sample_rate == 48000 and r.native_sample_rate == 48000 and r.audio.size > 1000
    c = calls(log)
    init = next(x for x in c if x["event"] == "init")
    assert init["denoiser"] is False and init["zip"] is None  # no hidden ModelScope download
    gen = [x for x in c if x["event"] == "generate"][-1]
    assert gen["text"] == "(calm, warm)Hello there." and gen["reference"] == str(ref) and gen["prompt_wav"] is None
    # with a transcript: official "ultimate cloning" (same clip as prompt + reference)
    eng.synthesize("Again.", SpeakerRef("me", ref, transcript="the words"), SynthesisOptions())
    gen = [x for x in calls(log) if x["event"] == "generate"][-1]
    assert gen["prompt_wav"] == str(ref) and gen["prompt_text"] == "the words"
    # streaming
    chunks = list(eng.synthesize_stream("one two three four", SpeakerRef("me", ref), SynthesisOptions()))
    assert len(chunks) > 2 and all(sr == 48000 for _, sr in chunks)
    eng.unload()


def test_voxcpm2_unavailable_messages(monkeypatch, tmp_path):
    eng = get_engine("voxcpm2", model_path=tmp_path)
    why = eng.available()
    assert why and "voxcpm" in why
    with pytest.raises(EngineUnavailable):
        eng.load()


def test_voicestar_gets_silence_aware_duration_plan(fakes, tmp_path):
    log, _ = fakes
    ref = wav(tmp_path / "ref.wav", seconds=2.0, lead=2.0)  # 2 s of leading silence
    eng = get_engine("voicestar")
    r = eng.synthesize("A short line.", SpeakerRef("me", ref, transcript="one two three four five six"), SynthesisOptions())
    gen = [x for x in calls(log) if x["event"] == "generate"][-1]
    assert gen["reference_text"] == "one two three four five six"
    assert gen["target_duration"] == r.params["target_duration_s"]
    assert r.params["duration_method"].startswith("silence-aware")
    assert gen["target_duration"] < 3.5  # not the 4 s reference length VoiceStar's default would use
    eng.unload()


def test_xvoice_cli_arguments_and_code_switch(fakes, tmp_path):
    log, _ = fakes
    snap = tmp_path / "xv"
    snap.mkdir()
    for n in ("model_stage2_70000.safetensors", "srp_model_28000.safetensors", "vocab.txt"):
        (snap / n).write_text("x")
    eng = get_engine("xvoice", model_dir=snap)
    assert eng.available() is None
    ref = wav(tmp_path / "ref.wav")
    r = eng.synthesize("[main|en]hello\n[main|zh]你好", SpeakerRef("me", ref), SynthesisOptions(speed=1.1, target_duration_s=2.0))
    a = [x for x in calls(log) if x["event"] == "xvoice"][-1]
    assert a["srp_ckpt_file"].endswith("srp_model_28000.safetensors") and a["p"].endswith("stage2_70000.safetensors")
    assert a["layered"] and a["auto_detect_lang"] and a["speed"] == "1.1" and a["fix_duration"] == "2.0"
    assert r.sample_rate == 24000 and abs(r.duration_s - 2.0) < 0.01


@pytest.mark.skipif(not shutil.which("espeak-ng"), reason="espeak-ng not installed")
def test_espeak_engine_is_not_a_cloner(tmp_path):
    eng = get_engine("espeak")
    r = eng.synthesize("hello world", None, SynthesisOptions(speed=1.2))
    assert r.audio.size > 1000 and r.params["wpm"] == 210
    with pytest.raises(UnsupportedOption):
        eng.synthesize("x", SpeakerRef("me", wav(tmp_path / "r.wav")), SynthesisOptions())


def test_engine_backend_for_realtime(fakes, tmp_path):
    from aivoice.engines.realtime import EngineTTSBackend

    _, snap = fakes
    eng = get_engine("voxcpm2", model_path=snap)
    be = EngineTTSBackend(eng, SpeakerRef("me", wav(tmp_path / "r.wav")))
    chunks = list(be.synthesize("one two three"))
    assert len(chunks) > 1 and chunks[0].sample_rate == 48000 and be.capabilities().streaming
    be.close()
