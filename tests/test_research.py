import importlib.util
import shutil
import wave
from pathlib import Path

import numpy as np
import pytest

from aivoice.audio.enhancement import (
    EnhancementUnavailable,
    crossfade_concat,
    detect_discontinuities,
    enhance,
)
from aivoice.audio.loudness import integrated_loudness, normalize
from aivoice.research import duration, longform, phonetics
from aivoice.research.flow_matching import FieldSpec, IncompatibleFields, euler_sample, fused_field
from aivoice.research.reference_quality import analyze_reference, effective_bandwidth, prepare_reference

ESPEAK = shutil.which("espeak-ng") is not None
_REAL_MAKE_VAD = __import__("aivoice.vad", fromlist=["make_vad"]).make_vad


@pytest.fixture(autouse=True)
def _energy_vad(monkeypatch):
    """Tone-based fixtures are not speech to Silero; pin the energy VAD so CI (no Silero) and dev match."""
    import aivoice.vad as vad

    monkeypatch.setattr(vad, "make_vad", lambda engine="auto", sample_rate=16000: vad.EnergyVAD(sample_rate=sample_rate))


def tone(seconds, sr=16000, f=180.0, amp=0.3):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


def write(path, x, sr):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    return path


# --- loudness ---------------------------------------------------------------


def test_loudness_known_values():
    # 1 kHz full-scale sine at 48 kHz: BS.1770 gives -3.01 LUFS (K-weighting is ~0 dB at 1 kHz)
    assert integrated_loudness(tone(5, 48000, 1000, 1.0), 48000) == pytest.approx(-3.01, abs=0.05)
    assert integrated_loudness(np.zeros(48000), 48000) == float("-inf")
    y, info = normalize(tone(5, 48000, 1000, 0.01), 48000, target_lufs=-23, max_gain_db=12)
    assert info["gain_db"] == pytest.approx(12.0)  # bounded: never boosts more than max_gain_db
    y, info = normalize(tone(5, 48000, 1000, 0.5), 48000, target_lufs=-6)
    assert np.abs(y).max() <= 10 ** (-1 / 20) + 1e-6  # peak ceiling respected


@pytest.mark.skipif(importlib.util.find_spec("pyloudnorm") is None, reason="pyloudnorm not installed")
def test_loudness_matches_pyloudnorm_deman():
    import pyloudnorm as pyln

    rng = np.random.default_rng(0)
    for sr in (16000, 22050, 44100):
        x = (rng.normal(0, 0.1, sr * 4) * np.sin(np.linspace(0, 20, sr * 4))).astype(np.float32)
        assert integrated_loudness(x, sr) == pytest.approx(pyln.Meter(sr, filter_class="DeMan").integrated_loudness(x.astype(float)), abs=1e-3)


# --- enhancement ------------------------------------------------------------


def test_click_detection_and_repair():
    x = tone(2.0) * 0.5
    assert detect_discontinuities(x, 16000).size == 0
    y = x.copy()
    y[[8000, 20000]] += 0.7
    found = detect_discontinuities(y, 16000)
    assert len(found) == 2
    out, rep = enhance(y, 16000, mode="auto")
    assert "declick" in rep.applied and rep.clicks_after == 0
    clean, rep2 = enhance(x, 16000, mode="auto")
    assert "declick" not in rep2.applied  # nothing to fix: clean audio is not rewritten
    with pytest.raises(EnhancementUnavailable):
        enhance(x, 16000, mode="high-fidelity")
    same, rep3 = enhance(x, 16000, mode="off")
    assert same is x and rep3.applied == []


def test_crossfade_concat_no_clicks_and_gaps():
    a, b = tone(1.0, f=200), tone(1.0, f=330) * -1
    hard = np.concatenate([a, b])
    soft = crossfade_concat([a, b], 16000, gaps_ms=[200])
    assert soft.size == a.size + b.size + 3200
    assert detect_discontinuities(soft, 16000).size <= detect_discontinuities(hard, 16000).size


def test_identity_guard_rejects_harmful_enhancement():
    class Enc:
        def similarity(self, *a):
            return 0.5

    y = tone(1.0)
    y[4000] += 0.9
    out, rep = enhance(y, 16000, mode="auto", encoder=Enc())
    assert rep.rejected and out is not None and np.array_equal(out, y.astype(np.float32))


# --- duration ---------------------------------------------------------------


def test_syllable_counting():
    assert duration.count_syllables("hello world") in (3, 3)
    assert duration.count_syllables("The birch canoe slid on the smooth planks.") >= 8


def test_silence_aware_plan_ignores_padding():
    speechy = tone(3.0) * (1 + np.sin(2 * np.pi * 4 * np.arange(48000) / 16000)).astype(np.float32) / 2
    padded = np.concatenate([np.zeros(32000, np.float32), speechy, np.zeros(32000, np.float32)])
    p1 = duration.plan_duration("one two three four five six", speechy, sr=16000, transcript="a b c d e f g h i j k l")
    p2 = duration.plan_duration("one two three four five six", padded, sr=16000, transcript="a b c d e f g h i j k l")
    assert p1.seconds == pytest.approx(p2.seconds, rel=0.1)
    naive = duration.naive_plan("one two three four five six", padded.size / 16000, "a b c d e f g h i j k l")
    assert naive > p2.seconds * 1.4
    assert duration.voicestar_default("anything at all", 7.0) == pytest.approx(7.0)


def test_plan_without_reference_uses_default_rate():
    p = duration.plan_duration("one two three four", None)
    assert p.method == "default-rate" and p.seconds > 0


# --- phonetics --------------------------------------------------------------


def test_normalize_text():
    t = phonetics.normalize_text("Dr. Smith paid $12.50 for 3 items (15%) via NASA.")
    assert "doctor" in t and "twelve dollars and fifty cents" in t and "three" in t and "fifteen percent" in t
    assert "N A S A" in t
    assert phonetics.normalize_text("STOP NOW") == "STOP NOW"
    assert phonetics.normalize_text("say gif", lexicon={"gif": "jif"}) == "say jif"


def test_language_detection_and_code_switch():
    assert phonetics.detect_language("Bonjour, je suis très content de vous voir") == "fr"
    assert phonetics.detect_language("今日はいい天気ですね") == "ja"
    assert phonetics.detect_language("Привет, как дела?") == "ru"
    runs = phonetics.segment_code_switch("I was going out, but 外面好像快下雨了. Maybe later.")
    assert [r.language for r in runs] == ["en", "zh", "en"]
    assert phonetics.xvoice_tagged("Hi there 你好").splitlines() == ["[main|en]Hi there", "[main|zh]你好"]


@pytest.mark.skipif(not ESPEAK, reason="espeak-ng not installed")
def test_ipa():
    assert "w" in (phonetics.to_ipa("hello world") or "")


# --- long-form --------------------------------------------------------------


def test_split_text_respects_abbreviations_and_limits():
    txt = "Dr. Smith arrived. " + "This is a long sentence, with clauses, " * 12 + "end.\n\nNew paragraph here."
    segs = longform.split_text(txt, max_chars=120)
    assert all(len(s.text) <= 120 for s in segs)
    assert segs[0].text.startswith("Dr. Smith arrived.")
    assert any(s.pause_after_ms == 650 for s in segs) and segs[-1].pause_after_ms == 0


def test_longform_synthesis_stitches(tmp_path):
    from aivoice.engines.base import AudioResult, EngineCapabilities

    class Fake:
        def capabilities(self):
            return EngineCapabilities("fake", "fake", ("tts",), zero_shot=True)

        def synthesize(self, text, speaker, options):
            return AudioResult(tone(0.5 + len(text) / 200) * np.random.default_rng(len(text)).uniform(0.2, 1), 16000, "fake")

    from aivoice.engines.base import SynthesisOptions

    res = longform.synthesize_longform(Fake(), "First sentence here. " * 6 + "\n\nAnd a final one.", None, SynthesisOptions(), max_chars=60)
    assert len(res.segments) >= 3 and res.audio.size > 0
    assert detect_discontinuities(res.audio, 16000).size == 0


# --- reference quality ------------------------------------------------------


def test_reference_analysis_and_preparation(tmp_path):
    sr = 16000
    x = np.concatenate([np.zeros(sr * 2, np.float32), tone(4.0) * 0.05, np.zeros(sr, np.float32)])
    p = write(tmp_path / "r.wav", x, sr)
    rep = analyze_reference(p)
    assert rep.leading_silence_s > 1.5 and any("silence at edges" in i for i in rep.issues)
    prep = prepare_reference(p, tmp_path / "prepared.wav", denoise=False)
    assert "trim" in prep.steps and "loudness" in prep.steps
    assert prep.report_after.leading_silence_s < 0.4
    assert Path(prep.prepared).is_file() and (tmp_path / "prepared.json").is_file()
    assert p.read_bytes() == write(tmp_path / "copy.wav", x, sr).read_bytes()  # original untouched


def test_identity_guard_reverts_steps(tmp_path):
    class Enc:
        name = "fake"

        def similarity(self, *a):
            return 0.1

        def embed(self, x, sr, return_partials=False, trim=True):
            raise ValueError("short")

    x = np.concatenate([np.zeros(32000, np.float32), tone(4.0) * 0.05])
    p = write(tmp_path / "r.wav", x, 16000)
    prep = prepare_reference(p, tmp_path / "out.wav", encoder=Enc(), denoise=True)
    assert prep.steps == [] and {"trim", "loudness", "denoise"} <= set(prep.reverted)


def test_bandwidth_flags_upsampled_audio(tmp_path):
    from aivoice.audio.resample import resample

    rng = np.random.default_rng(0)
    narrow = rng.normal(0, 0.1, 16000 * 3).astype(np.float32)
    up = resample(narrow, 16000, 48000)
    assert effective_bandwidth(up, 48000) < 9000
    rep = analyze_reference(write(tmp_path / "u.wav", up, 48000))
    assert rep.likely_upsampled


# --- flow matching ------------------------------------------------------------


def test_vector_field_fusion_requires_compatible_spaces():
    spec = FieldSpec("mel-100@24k", 100, 93.75, "log-mel/mean-std")

    class F:
        def __init__(self, target, s=spec):
            self.target, self.spec = target, s

        def __call__(self, x, t):
            return self.target - x  # straight path toward target

    a, b = F(np.ones(4)), F(-np.ones(4))
    v = fused_field(a, b, weight=0.75)
    out = euler_sample(v, np.zeros(4), steps=200)
    assert np.allclose(out, 0.5 * (1 - np.exp(-1)) * np.ones(4) / (1 - np.exp(-1)) * (1 - np.exp(-1)), atol=0.05) or out.mean() > 0
    assert out.mean() == pytest.approx(0.5 * (1 - np.exp(-1)), abs=0.01)
    with pytest.raises(IncompatibleFields, match="frame_rate"):
        fused_field(a, F(np.zeros(4), FieldSpec("mel-100@24k", 100, 50.0, "log-mel/mean-std")))


# --- speaker identity ---------------------------------------------------------


def test_yin_f0_on_tone():
    from aivoice.research.speaker_identity import yin_f0

    f0 = yin_f0(tone(1.0, f=150), 16000)
    assert np.median(f0[f0 > 0]) == pytest.approx(150, rel=0.02)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None or importlib.util.find_spec("resemblyzer") is None
                    or importlib.util.find_spec("librosa") is None, reason="GE2E needs torch, librosa, resemblyzer")
@pytest.mark.skipif(not ESPEAK, reason="espeak-ng not installed")
def test_ge2e_separates_synthetic_speakers(monkeypatch):
    import aivoice.vad as vad

    monkeypatch.setattr(vad, "make_vad", _REAL_MAKE_VAD)  # real (Silero) VAD on real synthetic speech
    from aivoice.evaluation import synthetic_data as sd
    from aivoice.research.speaker_identity import get_encoder

    enc = get_encoder(required=True)
    a1, sr = sd.speak(sd.HARVARD[0] + " " + sd.HARVARD[1], "synth-a")
    a2, _ = sd.speak(sd.HARVARD[2] + " " + sd.HARVARD[3], "synth-a")
    b1, _ = sd.speak(sd.HARVARD[2] + " " + sd.HARVARD[3], "synth-b")
    same, diff = enc.similarity(a1, sr, a2, sr), enc.similarity(a2, sr, b1, sr)
    assert same > diff + 0.15
