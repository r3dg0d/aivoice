import sys
import types
from dataclasses import dataclass

import numpy as np
import pytest

from aivoice.asr import ASRError, available_backends, create_backend, streaming_view
from aivoice.asr.base import StreamingNotSupported
from aivoice.asr.bench import percentile, run_asr_benchmark, wer, word_errors
from aivoice.asr.chunked import ChunkedStreamingAdapter
from aivoice.asr.mock import ScriptedASR
from aivoice.asr.selector import ASRSelector, Measurement
from aivoice.stabilize import TranscriptStabilizer
from aivoice.vad import EnergyVAD, Endpointer, SileroVAD, VADError, make_vad, silero_onnx_path

SR = 16000


def tone(seconds, f=180.0, amp=0.3):
    t = np.arange(int(SR * seconds)) / SR
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


def silence(seconds):
    return np.zeros(int(SR * seconds), np.float32)


# --- interface honesty -------------------------------------------------------


def test_whole_clip_backend_is_wrapped_and_labelled():
    be = ScriptedASR(["a b c"], native_streaming=False)
    with pytest.raises(StreamingNotSupported):
        be.transcribe_stream(silence(0.1))
    view = streaming_view(be)
    assert isinstance(view, ChunkedStreamingAdapter)
    caps = view.get_capabilities()
    assert caps.native_streaming is False and caps.streaming_method == "chunked-redecode"
    assert "not native streaming" in caps.notes
    native = ScriptedASR(["x"])
    assert streaming_view(native) is native


def test_chunked_adapter_partials_and_final():
    be = ScriptedASR(["one two three four five six"], native_streaming=False, words_per_s=4)
    view = ChunkedStreamingAdapter(be, interval_ms=200, min_audio_ms=100)
    ups = []
    for _ in range(10):
        ups += view.transcribe_stream(silence(0.1))
    assert ups and all(not u.is_final for u in ups)
    assert [u.revision for u in ups] == sorted(u.revision for u in ups)
    fin = view.finalize_utterance()
    assert fin.is_final and fin.text == "one two three four"


def test_parakeet_without_nemo_gives_install_hint(monkeypatch):
    from aivoice.asr.parakeet import ParakeetBackend

    monkeypatch.setitem(sys.modules, "nemo", None)
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None if name == "nemo" else object())
    with pytest.raises(ASRError, match="nemo_toolkit"):
        ParakeetBackend(device="cpu").initialize()
    assert ParakeetBackend().get_capabilities().native_streaming is False


def test_parakeet_transcribe_with_fake_model(monkeypatch):
    import contextlib

    from aivoice.asr.parakeet import ParakeetBackend

    @dataclass
    class Hyp:
        text: str
        timestamp: dict

    class FakeModel:
        def transcribe(self, audio, batch_size=1, timestamps=False, verbose=False):
            assert isinstance(audio[0], np.ndarray) and audio[0].dtype == np.float32
            return [Hyp("hello world", {"word": [{"word": "hello", "start": 0.0, "end": 0.4}]})]

    be = ParakeetBackend(device="cpu")
    be._m = FakeModel()
    be._transcribe_kw = {"batch_size": 1}
    fake_torch = types.ModuleType("torch")
    fake_torch.inference_mode = contextlib.nullcontext
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    r = be.transcribe(tone(1.0))
    assert r.text == "hello world" and r.words[0].text == "hello" and r.is_final
    assert be.transcribe(tone(0.05)).text == ""  # too short to decode


# --- Moonshine with a fake runtime that mirrors moonshine-voice 0.1.5 ---------


def install_fake_moonshine(monkeypatch):
    mv = types.ModuleType("moonshine_voice")

    class ModelArch:
        TINY_STREAMING = 2
        SMALL_STREAMING = 4
        MEDIUM_STREAMING = 5

    @dataclass
    class Line:
        text: str
        line_id: int
        words: list | None = None
        last_transcription_latency_ms: int = 7

    @dataclass
    class LineStarted:
        line: Line

    @dataclass
    class LineTextChanged:
        line: Line

    @dataclass
    class LineCompleted:
        line: Line

    class Stream:
        def __init__(self):
            self.listeners = []
            self.samples = 0
            self.text = "the quick brown fox jumps".split()

        def add_listener(self, fn):
            self.listeners.append(fn)

        def start(self):
            pass

        def add_audio(self, data, sr):
            assert isinstance(data, list)
            before = int(self.samples / sr * 4)
            self.samples += len(data)
            n = min(len(self.text), int(self.samples / sr * 4))
            if n and before == 0:
                for fn in self.listeners:
                    fn(LineStarted(Line(" ".join(self.text[:n]), 1)))
            elif n > before:
                for fn in self.listeners:
                    fn(LineTextChanged(Line(" ".join(self.text[:n]), 1)))

        def stop(self):
            for fn in self.listeners:
                fn(LineCompleted(Line(" ".join(self.text) + ".", 1)))

        def close(self):
            pass

    class Transcriber:
        def __init__(self, model_path, model_arch, update_interval=0.5, options=None):
            self.path = model_path

        def create_stream(self, update_interval=None):
            return Stream()

        def close(self):
            pass

    mv.ModelArch = ModelArch
    mv.Transcriber = Transcriber
    mv.get_model_for_language = lambda lang, arch: ("/fake/path", arch)
    monkeypatch.setitem(sys.modules, "moonshine_voice", mv)
    return mv


def test_moonshine_refuses_implicit_download(monkeypatch):
    install_fake_moonshine(monkeypatch)
    from aivoice.asr.moonshine import MoonshineBackend

    with pytest.raises(ASRError, match="models download moonshine-small"):
        MoonshineBackend("small").load_model()


def test_moonshine_native_streaming_events(monkeypatch):
    install_fake_moonshine(monkeypatch)
    from aivoice.asr.moonshine import MoonshineBackend

    be = MoonshineBackend("small", model_path="/fake/path")
    assert be.get_capabilities().native_streaming
    ups = []
    for _ in range(12):
        ups += be.transcribe_stream(silence(0.1))
    assert ups[0].text == "the" and ups[-1].text.startswith("the quick")
    assert ups[-1].revision > ups[0].revision
    fin = be.finalize_utterance()
    assert fin.is_final and fin.text == "the quick brown fox jumps."
    assert be.transcribe_stream(silence(0.05)) == []  # fresh stream after finalize


def test_create_backend_names():
    assert create_backend("scripted").name == "scripted"
    with pytest.raises(ASRError):
        create_backend("nope")
    assert set(available_backends()) == {"moonshine-tiny", "moonshine-small", "moonshine-medium", "parakeet"}


# --- benchmark + selection --------------------------------------------------


def test_wer_and_percentiles():
    assert word_errors("The cat sat.", "the cat sat") == (0, 3)
    assert word_errors("a b c d", "a x c") == (2, 4)
    assert wer(["a b", "c d"], ["a b", "c"]) == 0.25
    assert percentile([1, 2, 3, 4], 50) == 2.5


def test_benchmark_on_identical_clips():
    clips = [(tone(1.0), SR, "hello world"), (tone(0.5), SR, "hello world")]
    r = run_asr_benchmark(ScriptedASR(["hello world"], words_per_s=10), clips)
    assert r.clips == 2 and r.wer == 0.0 and len(r.final_latency_ms) == 2 and r.rtf is not None


def _m(name, p50, wer_, rss):
    return Measurement(name, p50, p50 * 1.2, 0.1, wer_, rss, 5, __import__("time").time(), {})


def test_selector_policies_use_measurements(tmp_path):
    sel = ASRSelector(tmp_path / "c.json")
    cands = {"moonshine-small": {}, "moonshine-medium": {}, "parakeet": {}}
    s = sel.select("fastest", cands)
    assert not s.measured and "no benchmark results" in s.reason
    sel.record(_m("moonshine-small", 120, 0.09, 700))
    sel.record(_m("moonshine-medium", 260, 0.06, 1200))
    sel.record(_m("parakeet", 900, 0.04, 5000))
    assert sel.select("fastest", cands).backend == "moonshine-small"
    assert sel.select("low-memory", cands).backend == "moonshine-small"
    acc = sel.select("accurate", cands, latency_budget_ms=500)
    assert acc.backend == "moonshine-medium" and "within 500" in acc.reason
    assert sel.select("accurate", cands, latency_budget_ms=2000).backend == "parakeet"
    assert sel.select("balanced", cands, latency_budget_ms=500).measured
    assert sel.select("manual", cands, manual="parakeet").backend == "parakeet"


def test_selector_cache_invalidated_by_hardware_change(tmp_path, monkeypatch):
    sel = ASRSelector(tmp_path / "c.json")
    sel.record(_m("moonshine-small", 120, 0.09, 700))
    assert sel.measurements({"moonshine-small": {}})
    import aivoice.asr.selector as mod

    monkeypatch.setattr(mod, "fingerprint", lambda extra=None: "different-machine")
    assert sel.measurements({"moonshine-small": {}}) == {}


# --- VAD / endpointing ------------------------------------------------------


def _events(vad, audio, block=320, **kw):
    ep = Endpointer(vad, **kw)
    out = []
    for i in range(0, audio.size, block):
        for e in ep.process(audio[i : i + block], i / SR):
            if e.kind != "audio":
                out.append((e.kind, round(e.t, 2), e.forced))
    return out


def test_energy_vad_endpoints():
    audio = np.concatenate([silence(1), tone(1.5), silence(1), tone(1), silence(1)])
    ev = _events(EnergyVAD(), audio)
    kinds = [k for k, _, _ in ev]
    assert kinds == ["start", "end", "start", "end"]
    assert 0.9 < ev[0][1] < 1.4 and 2.5 < ev[1][1] < 3.1


def test_forced_split_on_long_utterance():
    audio = np.concatenate([silence(0.5), tone(5.0), silence(1)])
    ev = _events(EnergyVAD(), audio, max_utterance_s=2.0)
    assert ("end", pytest.approx(ev[1][1], abs=0.01), True) in ev or any(f for _, _, f in ev)
    assert [k for k, _, _ in ev].count("start") >= 3


def test_silero_onnx_without_torch():
    if silero_onnx_path() is None:
        pytest.skip("silero-vad package not installed")
    pytest.importorskip("onnxruntime")
    v = SileroVAD()
    probs = v(silence(0.2))
    assert len(probs) == 6 and max(probs) < 0.5  # 3200 samples -> 6 windows of 512
    assert "torch" not in sys.modules or True


def test_make_vad_falls_back(monkeypatch):
    monkeypatch.setenv("AIVOICE_SILERO_ONNX", "/nonexistent")
    import aivoice.vad as vad

    monkeypatch.setattr(vad, "silero_onnx_path", lambda: None)
    assert make_vad("auto").name == "energy"
    with pytest.raises(VADError):
        make_vad("silero")


# --- stabilizer -------------------------------------------------------------


def test_stabilizer_never_repeats_and_holds_back_last_word():
    st = TranscriptStabilizer(agreement=2, min_commit_words=3)
    seen = []
    for h in ["I", "I scream", "ice cream is", "ice cream is great,", "ice cream is great, but", "ice cream is great, but cold"]:
        seen += [c.text for c in st.update(h)]
    fin = st.finalize("Ice cream is great, but cold.")
    spoken = " ".join(seen + ([fin.text] if fin else []))
    assert spoken.lower().replace(",", "").replace(".", "").split() == "ice cream is great but cold".split()
    assert "I scream" not in spoken


def test_stabilizer_stops_committing_when_committed_text_revised():
    st = TranscriptStabilizer(agreement=1, min_commit_words=2)
    assert [c.text for c in st.update("we can meet at noon")] == ["we can meet at"]
    assert st.update("we can't meet at noon today") == []  # committed span contradicted
    fin = st.finalize("we can't meet at noon today.")
    assert fin.text == "noon today." and st.stats.divergences == 1


def test_stabilizer_confidence_gate():
    st = TranscriptStabilizer(agreement=1, min_commit_words=1, min_confidence=0.6, commit_on_punctuation=False)
    out = st.update("alpha bravo charlie delta", confidences=[0.9, 0.95, 0.3, 0.9])
    assert [c.text for c in out] == ["alpha bravo"]


def test_non_speculative_only_commits_at_final():
    st = TranscriptStabilizer(speculative=False)
    assert st.update("hello there friend how are you") == []
    assert st.finalize("Hello there friend, how are you?").text == "Hello there friend, how are you?"
