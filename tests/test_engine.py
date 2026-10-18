"""Real-time engine with scripted ASR / tone TTS (real threads, real clocks, no models)."""

import json
import time

import numpy as np
import pytest

from aivoice.asr.mock import ScriptedASR
from aivoice.audio.capture import ArrayCapture, _BlockQueue
from aivoice.audio.playback import ClockDriver, Player, soft_limit
from aivoice.audio.resample import StreamResampler, resample
from aivoice.llm import ScriptedLLM, SpeechIntelligenceEngine
from aivoice.realtime.engine import EngineOptions, RealtimeEngine
from aivoice.tts.mock import ToneTTS
from aivoice.vad import EnergyVAD, Endpointer

SR = 16000


def speechlike(seconds):
    t = np.arange(int(SR * seconds)) / SR
    return (0.3 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)


def clip(*parts):
    out = []
    for p in parts:
        out.append(speechlike(p) if p > 0 else np.zeros(int(SR * -p), np.float32))
    return np.concatenate(out)


def make(audio, scripts, *, tts=None, opts=None, llm=None, realtime=True, words_per_s=4.0):
    drivers = []

    def df(p):
        d = ClockDriver(p, realtime=True)
        drivers.append(d)
        return d

    eng = RealtimeEngine(
        capture=ArrayCapture(audio, SR, realtime=realtime),
        asr=ScriptedASR(scripts, words_per_s=words_per_s),
        endpointer=Endpointer(EnergyVAD(), min_silence_ms=250),
        tts=tts or ToneTTS(ttfa_ms=20),
        driver_factory=df,
        llm=llm,
        options=opts or EngineOptions(),
    )
    return eng, drivers


def run(eng, timeout=30):
    eng.start()
    assert eng.wait_idle(timeout)
    eng.stop()


def spoken(eng):
    return [s for t in eng.timelines.all() for s in t.segments if not s.cancelled]


def test_speculative_pipeline_speaks_each_word_once_in_order():
    eng, drv = make(clip(-0.5, 3.0, -0.8), ["Hello there, this is a test of the voice engine."])
    run(eng)
    segs = spoken(eng)
    text = " ".join(s.text for s in segs)
    assert text == "Hello there, this is a test of the voice engine."
    assert len(segs) >= 2 and segs[-1].final
    utt = eng.timelines.all()[0]
    m = utt.metrics()
    assert m["endpoint_to_first_audio_ms"] < 0  # started speaking before the endpoint
    assert m["tts_request_to_first_audio_ms"] >= 15
    assert all(s.t_done for s in segs)
    out = drv[0].audio()
    assert np.abs(out).max() > 0.1 and eng.player.underruns == 0


def test_non_speculative_waits_for_final():
    eng, _ = make(clip(-0.5, 2.0, -0.8), ["one two three four five six"], opts=EngineOptions(speculative=False))
    run(eng)
    segs = spoken(eng)
    assert len(segs) == 1 and segs[0].final and segs[0].text == "one two three four five six"
    m = eng.timelines.all()[0].metrics()
    assert m["endpoint_to_first_audio_ms"] > 0


def test_two_utterances_get_distinct_ids_and_queue():
    eng, _ = make(clip(-0.4, 1.5, -0.8, 1.5, -0.8), ["first sentence here.", "second one now."])
    run(eng)
    tl = eng.timelines.all()
    assert len(tl) == 2 and tl[0].uid != tl[1].uid
    assert tl[0].final_text == "first sentence here." and tl[1].final_text == "second one now."
    assert not any(s.cancelled for t in tl for s in t.segments)  # queue policy: nothing cut off


def test_barge_in_interrupt_cancels_previous_output():
    slow = ToneTTS(ttfa_ms=10, words_per_s=0.5, rtf=1.0)  # long, slow audio
    eng, _ = make(clip(-0.3, 1.2, -0.5, 1.0, -1.0), ["a b c d e f.", "stop now."], tts=slow,
                  opts=EngineOptions(barge_in="interrupt", speculative=False))
    run(eng, timeout=60)
    first = eng.timelines.all()[0]
    assert any(s.cancelled == "barge-in" for s in first.segments)


def test_backlog_drops_stale_segments():
    stuck = ToneTTS(ttfa_ms=400, words_per_s=1)
    eng, _ = make(clip(-0.3, 3.0, -0.8), ["w1 w2 w3, w4 w5 w6, w7 w8 w9, w10 w11 w12."], tts=stuck,
                  opts=EngineOptions(max_backlog_ms=300, min_commit_words=1))
    run(eng, timeout=60)
    reasons = [s.cancelled for t in eng.timelines.all() for s in t.segments]
    assert "backlog" in reasons


def test_mute_drops_input_and_ptt():
    eng, _ = make(clip(-0.2, 1.5, -0.6), ["muted words."])
    eng.mute(True)
    run(eng)
    assert eng.timelines.all() == []

    eng, _ = make(clip(-0.2, 1.5, -0.6), ["push to talk."], opts=EngineOptions(input_mode="ptt"))
    eng.start()
    eng.set_ptt(True)
    time.sleep(1.2)
    eng.set_ptt(False)
    assert eng.wait_idle(20)
    eng.stop()
    assert eng.timelines.all()[0].final_text == "push to talk."


def test_tts_errors_are_recorded_not_fatal():
    eng, _ = make(clip(-0.3, 1.2, -0.6), ["will fail."], tts=ToneTTS(fail=True))
    run(eng)
    segs = eng.timelines.all()[0].segments
    assert segs and segs[0].error and eng.errors


def test_llm_rewrite_disables_speculation_and_applies():
    import json as _j

    llm = SpeechIntelligenceEngine(
        ScriptedLLM(fn=lambda u: _j.dumps({"transformed": u.upper(), "emotion": "excited", "delivery": "", "confidence": 1})),
        mode="natural", timeout_ms=500)
    tts = ToneTTS(dialect="drama3")
    eng, _ = make(clip(-0.3, 1.5, -0.6), ["make it loud"], tts=tts, llm=llm)
    run(eng)
    segs = spoken(eng)
    assert len(segs) == 1 and segs[0].tts_text == "[excited] MAKE IT LOUD"
    assert tts.requests[-1] == "[excited] MAKE IT LOUD"
    assert segs[0].t_llm_end >= segs[0].t_llm_start


def test_status_file(tmp_path):
    p = tmp_path / "engine.json"
    eng, _ = make(clip(-0.3, 1.0, -0.6), ["status check."], opts=EngineOptions(status_path=p))
    run(eng)
    data = json.loads(p.read_text())
    assert data["state"] == "STOPPED" and data["latency"]["utterances"] == 1
    assert data["asr"]["streaming"] == "native" and data["last"]["final"] == "status check."


def test_asr_switch_applies_between_utterances():
    eng, _ = make(clip(-0.3, 1.0, -1.0, 1.0, -0.8), ["first.", "x"], realtime=True)
    eng.start()
    eng.switch_asr(ScriptedASR(["second from new backend."], name="scripted-2"))
    assert eng.wait_idle(30)
    eng.stop()
    backends = [t.asr_backend for t in eng.timelines.all()]
    assert backends[-1] == "scripted-2"


# --- audio primitives ------------------------------------------------------


def test_player_prebuffer_underrun_cancel():
    p = Player(1000, prebuffer_ms=50)
    p.write("a", np.ones(20, np.float32) * 0.5)
    assert not p.pull(10).any()  # below prebuffer, not ended -> wait
    p.write("a", np.ones(40, np.float32) * 0.5)
    assert p.pull(30).max() > 0.4
    p.pull(30)
    p.pull(10)
    assert p.underruns >= 1  # ran dry mid-utterance
    p.end("a")
    p.pull(10)
    kinds = [e.kind for e in p.events()]
    assert kinds[0] == "first_audio" and "done" in kinds
    p.write("b", np.ones(100, np.float32))
    assert p.cancel("b") == ["b"] and p.pending == 0


def test_limiter_and_gain_ramp():
    y = soft_limit(np.array([0.5, 1.5, -2.0], np.float32))
    assert y[0] == 0.5 and 0.9 < y[1] <= 1.0 and -1.0 <= y[2] < -0.9 and np.abs(y).max() <= 1.0
    p = Player(1000, prebuffer_ms=0)
    p.write("a", np.ones(100, np.float32) * 0.5)
    p.end("a")
    p.duck(True, -20)
    blk = p.pull(50)
    assert blk[0] > blk[-1] > 0  # ramped, not stepped


def test_stream_resampler_is_continuous():
    sr_in, sr_out = 44100, 48000
    t = np.arange(sr_in) / sr_in
    x = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    rs = StreamResampler(sr_in, sr_out)
    y = np.concatenate([rs.process(x[i : i + 1000]) for i in range(0, x.size, 1000)])
    assert abs(y.size - sr_out) <= 2
    assert np.abs(np.diff(y)).max() < 0.07  # no clicks at chunk boundaries
    assert resample(x, 44100, 16000).size == pytest.approx(16000, abs=2)


def test_block_queue_drops_oldest_never_blocks():
    q = _BlockQueue(3)
    for i in range(10):
        q.put(i)
    assert q.dropped == 7 and [q.get(0) for _ in range(3)] == [7, 8, 9]
