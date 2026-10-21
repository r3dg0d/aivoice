"""Live VC layer: DSP modulation, presets, streaming resampler, worker transport, fallback, router, control, CLI."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from aivoice.audio.resample import StreamingResampler
from aivoice.modulation.dsp import ModulationParams, SpectralModulator, _yin_frame, measure_delay
from aivoice.modulation.presets import BUILTIN, get_preset, overrides_from_flags, save_user_preset, user_presets
from aivoice.vc.base import EngineState, EngineStatusVC, StreamOptions, VCCapabilities, VoiceTarget
from aivoice.vc.engines import (
    DspEngine,
    EchoWorkerEngine,
    PassthroughEngine,
    WorkerVCEngine,
    all_vc_status,
    catalog_only,
    get_vc_engine,
)
from aivoice.vc.live import LiveConfig, LivePipeline, read_status, send_control
from aivoice.vc.router import route_vc

SR = 48000
WORKER = Path(__file__).resolve().parent.parent / "src" / "aivoice" / "vc" / "stream_worker.py"


def harmonic(f0=140.0, seconds=1.0, sr=SR, n=25):
    t = np.arange(int(sr * seconds)) / sr
    return (0.5 * sum((0.3 / k) * np.sin(2 * np.pi * f0 * k * t) for k in range(1, n))).astype(np.float32)


def load_worker_module():
    spec = importlib.util.spec_from_file_location("aivoice_stream_worker_test", WORKER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.stdout = sys.__stdout__  # the worker module redirects stdout for its protocol
    return mod


# --- DSP ----------------------------------------------------------------------------------


def test_identity_round_trip_is_transparent():
    x = harmonic()
    y = SpectralModulator(SR).process_all(x)
    assert np.max(np.abs(y[2000:-2000] - x[2000:-2000])) < 1e-5


@pytest.mark.parametrize("st", [-4.0, -12.0, 5.0])
def test_pitch_shift_lands_on_target(st):
    x = harmonic(140.0, 1.5)
    y = SpectralModulator(SR, ModulationParams(pitch_semitones=st)).process_all(x, block=480)
    f, _ = _yin_frame(y[SR // 2 : SR // 2 + 2048].astype(float), SR, 50, 900)
    assert abs(12 * np.log2(f / (140 * 2 ** (st / 12)))) < 0.1  # within 10 cents


def test_formant_warp_moves_envelope_not_pitch():
    # one strong resonance at 1200 Hz on a 120 Hz pulse train
    sr, f0 = SR, 120.0
    n = sr
    pulses = np.zeros(n)
    pulses[:: int(sr / f0)] = 1.0
    from numpy.fft import irfft, rfft

    f = np.fft.rfftfreq(n, 1 / sr)
    H = 1.0 / (1 + ((f - 1200) / 150) ** 2) + 0.05
    x = irfft(rfft(pulses) * H, n).astype(np.float32) * 0.3
    y = SpectralModulator(sr, ModulationParams(formant_ratio=0.85)).process_all(x)

    def peak(sig):
        seg = sig[sr // 4 : sr // 4 + 8192] * np.hanning(8192)
        S = np.abs(np.fft.rfft(seg))
        ff = np.fft.rfftfreq(8192, 1 / sr)
        sm = np.convolve(S, np.ones(41) / 41, "same")  # envelope, not harmonics
        band = (ff > 400) & (ff < 3000)
        return ff[band][np.argmax(sm[band])]

    assert abs(peak(y) / peak(x) - 0.85) < 0.08
    fx, _ = _yin_frame(x[sr // 2 : sr // 2 + 2048].astype(float), sr, 50, 600)
    fy, _ = _yin_frame(y[sr // 2 : sr // 2 + 2048].astype(float), sr, 50, 600)
    assert abs(fy / fx - 1) < 0.02


def test_output_independent_of_block_size_and_delay_constant():
    x = harmonic(seconds=0.8)
    p = ModulationParams(pitch_semitones=-3, formant_ratio=0.92)
    outs = [SpectralModulator(SR, p).process_all(x, block=b) for b in (384, 480, 4801, 7)]
    for o in outs[1:]:
        assert np.max(np.abs(o - outs[0])) == 0.0
    m = SpectralModulator(SR)
    assert measure_delay(m) == m.latency_samples


def test_chunk_boundaries_have_no_dropouts():
    x = harmonic(seconds=1.0)
    m = SpectralModulator(SR, ModulationParams(pitch_semitones=-4))
    y = np.concatenate([m.process(x[i : i + 480]) for i in range(0, x.size, 480)])
    steady = y[m.latency_samples + 2000 : -2000]
    win = 240  # 5 ms windows: a dropped / zero-filled stretch would show as a near-silent window
    rms = np.sqrt(np.mean(steady[: steady.size // win * win].reshape(-1, win) ** 2, axis=1))
    assert rms.min() > 0.3 * np.median(rms)


def test_bypass_keeps_delay_and_nan_input_is_repaired():
    x = harmonic(seconds=0.5)
    m = SpectralModulator(SR, ModulationParams(pitch_semitones=-5))
    m.bypass = True
    y = np.concatenate([m.process(x[i : i + 480]) for i in range(0, x.size, 480)])
    d = m.latency_samples
    assert np.allclose(y[d:], np.clip(x[: y.size - d], -1, 1), atol=1e-4) or np.max(np.abs(y[d:] - x[: y.size - d])) < 0.12
    bad = x.copy()
    bad[100:200] = np.nan
    out = SpectralModulator(SR).process(bad)
    assert np.all(np.isfinite(out))


def test_params_reject_unknown_fields():
    with pytest.raises(ValueError):
        ModulationParams.from_dict({"pitch": 1})


# --- presets --------------------------------------------------------------------------------


def test_masculine_is_not_pitch_only_and_strength_scales():
    m = get_preset("masculine")
    p = m.params(1.0)
    assert p.pitch_semitones < 0 and p.formant_ratio < 1.0
    half = m.params(0.5)
    assert half.pitch_semitones == pytest.approx(p.pitch_semitones / 2)
    assert half.formant_ratio == pytest.approx(p.formant_ratio**0.5)
    assert m.params(0.0).is_identity()
    assert get_preset("male").name == "masculine"


def test_every_builtin_preset_builds_and_runs():
    x = harmonic(seconds=0.3)
    for p in BUILTIN.values():
        y = SpectralModulator(SR, p.params(), window_ms=p.window_ms, hop_ms=p.window_ms / 4).process_all(x)
        assert y.shape == x.shape and np.all(np.isfinite(y)) and np.max(np.abs(y)) <= 1.0


def test_advanced_flags_and_user_presets():
    over = overrides_from_flags(pitch=-2, formants=-10, roughness=0.3)
    assert over == {"pitch_semitones": -2.0, "formant_ratio": pytest.approx(0.9), "roughness": 0.3}
    with pytest.raises(ValueError):
        overrides_from_flags(breathiness=2)
    save_user_preset("my-deep", get_preset("deep").params(0.5), description="test")
    assert "my-deep" in user_presets()
    assert get_preset("my-deep").params().pitch_semitones == pytest.approx(get_preset("deep").params(0.5).pitch_semitones)
    with pytest.raises(ValueError):
        save_user_preset("natural", ModulationParams())


# --- resampler ---------------------------------------------------------------------------------


@pytest.mark.parametrize("a,b", [(48000, 16000), (16000, 48000), (48000, 44100)])
def test_streaming_resampler_is_chunk_invariant(a, b):
    x = np.random.default_rng(0).normal(0, 0.1, a // 2).astype(np.float32)
    full = StreamingResampler(a, b).process(x)
    r = StreamingResampler(a, b)
    rng = np.random.default_rng(1)
    parts, i = [], 0
    while i < x.size:
        n = int(rng.integers(1, 500))
        parts.append(r.process(x[i : i + n]))
        i += n
    assert np.max(np.abs(np.concatenate(parts) - full)) < 1e-6
    assert abs(full.size - x.size * b / a) <= 1


# --- X-VC streaming schedule == upstream run_streaming ----------------------------------------


def _upstream_run_streaming(forward, src, sr, chunk_ms, current_ms, future_ms, smooth_ms):
    """numpy port of X-VC bins/infer_utils.run_streaming (commit 49df8c5) for comparison."""
    source_len = src.size
    history_ms = chunk_ms - current_ms - smooth_ms - future_ms
    overlap_len = smooth_ms * sr // 1000
    fade_in = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, overlap_len)))
    fade_out = 1 - fade_in
    tail = np.zeros(overlap_len)
    current_len = current_ms * sr // 1000
    n_chunks = (source_len + current_len - 1) // current_len
    out = []
    for i in range(n_chunks):
        s = (i * current_ms - history_ms) * sr // 1000
        e = (i * current_ms + current_ms + smooth_ms + future_ms) * sr // 1000
        lp, rp = max(0, -s), max(0, e - source_len)
        w = np.pad(src[s + lp : e - rp], (lp, rp))
        y = forward(w)
        seg = y[history_ms * sr // 1000 : (history_ms + current_ms) * sr // 1000].copy()
        if i > 0:
            seg[:overlap_len] = tail * fade_out + seg[:overlap_len] * fade_in
        t0 = (history_ms + current_ms) * sr // 1000
        tail = y[t0 : t0 + overlap_len]
        out.append(seg)
    return np.concatenate(out)[:source_len]


def test_xvc_incremental_windows_match_upstream():
    mod = load_worker_module()
    sr = 16000

    def forward(w):  # depends on the whole window, like a model with context
        return np.tanh(w * 3) * (1 + 0.1 * np.mean(np.abs(w))) + 0.01 * np.cumsum(w) / (1 + np.arange(w.size))

    src = np.random.default_rng(3).normal(0, 0.2, int(sr * 3.3)).astype(np.float32)
    ref = _upstream_run_streaming(forward, src.astype(np.float64), sr, 2400, 120, 100, 20)
    st = mod.SlidingWindowStreamer(forward, sr, 2400, 120, 100, 20)
    rng = np.random.default_rng(4)
    parts, i = [], 0
    while i < src.size:  # irregular microphone blocks
        n = int(rng.integers(100, 900))
        parts.append(st.push(src[i : i + n]))
        i += n
    parts.append(st.flush())
    got = np.concatenate(parts)[: src.size]
    assert got.size == ref.size
    assert np.max(np.abs(got - ref)) < 1e-5
    assert st.latency_samples == (120 + 100 + 20) * sr // 1000


# --- worker transport, crash, timeout, fallback ------------------------------------------------


def test_echo_worker_round_trip():
    e = EchoWorkerEngine(delay_samples=160)
    try:
        e.load()
        e.prepare_reference(None)
        e.start_stream(StreamOptions())
        x = np.random.default_rng(0).normal(0, 0.1, 16000).astype(np.float32)
        y = np.concatenate([e.process_chunk(x[i : i + 320]) for i in range(0, x.size, 320)])
        assert np.allclose(y[160:], x[:-160])
        assert e.timing_summary()["chunks"] == 50
    finally:
        e.unload()


def _speech(seconds=1.0):
    return harmonic(130.0, seconds)


def test_worker_crash_falls_back_and_audio_continues():
    e = EchoWorkerEngine(crash_after=10)
    p = LivePipeline(LiveConfig(engine="echo", fallback=["dsp"]), e)
    p.start()
    y = p.run_array(_speech(1.0), SR)
    s = p.snapshot()
    assert s["engine"] == "dsp" and s["fallbacks"][0]["from"] == "echo"
    tail = y[int(0.5 * SR) : int(0.9 * SR)]
    assert np.sqrt(np.mean(tail**2)) > 0.01  # the stream kept producing audio


def test_worker_timeout_falls_back():
    e = EchoWorkerEngine(sleep_ms=800)
    e.chunk_timeout_s = 0.3
    p = LivePipeline(LiveConfig(engine="echo", fallback=["dsp"]), e)
    p.start()
    p.run_array(_speech(0.2), SR)
    assert p.active_name == "dsp" and "timed out" in p.stats.fallbacks[0]["reason"]


def test_primary_restart_after_crash():
    e = EchoWorkerEngine(crash_after=3)
    cfg = LiveConfig(engine="echo", fallback=["dsp"], restart_primary_after_s=0.0)
    p = LivePipeline(cfg, e)
    p.start()
    p.run_array(_speech(0.2), SR)
    assert p.active_name == "dsp"
    assert p.maybe_restart_primary(lambda: EchoWorkerEngine())
    assert p.active_name == "echo"
    p.engine.unload()


def test_missing_interpreter_is_missing_runtime(monkeypatch):
    monkeypatch.setenv("AIVOICE_XVC_PYTHON", "/nonexistent/python")
    st = get_vc_engine("xvc").status()
    assert st.state == EngineState.MISSING_RUNTIME and not st.ready


def test_catalog_only_engines_can_never_start(monkeypatch, tmp_path):
    for c in catalog_only():
        st = c.status()
        assert not st.state.runnable
        with pytest.raises(Exception):
            c.load()
    # 0.7: RT-VC / Zero-VC / TVTSyn are paper reproductions; without a trained checkpoint they cannot start
    monkeypatch.setenv("AIVOICE_REPRO_HOME", str(tmp_path))
    states = {s.name: s.state for s in all_vc_status()}
    assert states["dsp"] == EngineState.READY
    for k in ("rtvc", "zerovc-repro", "tvtsyn-repro"):
        assert states[k] == EngineState.MISSING_WEIGHTS
        assert not states[k].runnable


def test_meanvc2_handler_drives_vcrunner(tmp_path):
    pytest.importorskip("torch")
    vendor = tmp_path / "MeanVC2"
    (vendor / "runtime" / "src").mkdir(parents=True)
    (vendor / "runtime" / "src" / "__init__.py").write_text("")
    (vendor / "runtime" / "src" / "speaker.py").write_text("def extract_embedding(m, path, device='cpu'):\n    return path\n")
    (vendor / "runtime" / "run_rt.py").write_text(
        "import numpy as np\n"
        "class _VC:\n    def gtm(self, e):\n        return ('gtm', e)\n"
        "class VCRunner:\n"
        "    def __init__(self, target_wav, device='cpu', model='120ms'):\n"
        "        self.target, self.vc, self.spk_model, self.calls = target_wav, _VC(), None, []\n"
        "    def _init_cache(self):\n        self.buf = np.zeros(0, np.float32); self.calls.append('init')\n"
        "    def process_chunk(self, x):\n"
        "        self.buf = np.concatenate([self.buf, x])\n"
        "        if self.buf.size < 640: return None\n"
        "        out, self.buf = self.buf[:640] * 0.5, self.buf[640:]\n        return out\n")
    mod = load_worker_module()
    h = mod.MeanVC2Handler()
    cwd = os.getcwd()
    try:
        h.load({"vendor_dir": str(vendor), "reference": "a.wav", "device": "cpu", "model": "40ms"})
        h.prepare({"reference": "b.wav"})
        assert h.runner.vc_spk_emb == "b.wav" and h.runner.vc_gtm_kv == ("gtm", "b.wav")
        h.start({})
        assert h.chunk(np.ones(320, np.float32)).size == 0
        assert np.allclose(h.chunk(np.ones(320, np.float32)), 0.5)
    finally:
        os.chdir(cwd)
        for k in [k for k in sys.modules if k in ("run_rt", "src", "src.speaker")]:
            del sys.modules[k]


def test_rt_worker_imports_nothing_heavy_at_module_level():
    path = Path(__file__).resolve().parent.parent / "src" / "aivoice" / "rvc" / "rt_worker.py"
    r = subprocess.run([sys.executable, "-c", f"import importlib.util,sys;s=importlib.util.spec_from_file_location('m','{path}');"
                        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
                        "assert hasattr(m,'RvcRealtime');assert 'torch' not in sys.modules and 'aivoice' not in sys.modules"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


# --- router -----------------------------------------------------------------------------------------


def _st(name, state=EngineState.READY, **caps):
    return EngineStatusVC(name, state, VCCapabilities(name=name, display_name=name, **caps))


def test_router_unmeasured_order_and_filters():
    sts = [_st("xvc", algorithmic_latency_ms=240.0), _st("meanvc2", algorithmic_latency_ms=80.0),
           _st("rvc", EngineState.INSTALLED, requires_model_file=True, requires_reference=False, algorithmic_latency_ms=310.0),
           _st("stylestream", EngineState.EXPERIMENTAL), _st("dsp", task="modulation", requires_reference=False)]
    t = VoiceTarget("v", reference=Path("r.wav"))
    d = route_vc("balanced", target=t, statuses=sts, measured={})
    assert d.engine == "meanvc2" and not d.measured  # xvc not promoted without local measurements
    assert "experimental" in d.rejected["stylestream"] and "rvc" in d.rejected
    d = route_vc("gaming", target=t, statuses=sts, measured={})
    assert "xvc" in d.rejected and d.engine == "meanvc2"
    t2 = VoiceTarget("v", reference=Path("r.wav"), model_file=Path("m.pth"))
    d = route_vc("latency", target=t2, statuses=[s for s in sts if s.name != "meanvc2"], measured={})
    assert d.engine == "rvc"


def test_router_uses_local_measurements():
    sts = [_st("xvc", algorithmic_latency_ms=240.0), _st("meanvc2", algorithmic_latency_ms=80.0)]
    t = VoiceTarget("v", reference=Path("r.wav"))
    m = {"xvc": {"target_similarity": 0.81, "wer": 0.05, "measured_delay_ms": 200, "chunk_p95_ms": 30, "rtf": 0.3},
         "meanvc2": {"target_similarity": 0.62, "wer": 0.09, "measured_delay_ms": 90, "chunk_p95_ms": 10, "rtf": 0.2}}
    d = route_vc("similarity", target=t, statuses=sts, measured=m)
    assert d.engine == "xvc" and d.measured
    assert route_vc("latency", target=t, statuses=sts, measured=m).engine == "meanvc2"
    m["xvc"]["rtf"] = 1.3  # not real time here -> never chosen by score
    assert route_vc("similarity", target=t, statuses=sts, measured=m).engine == "meanvc2"


# --- live control / status (QuickShell IPC contract) --------------------------------------------------


STATUS_KEYS = {"pid", "running", "mode", "engine", "engine_display", "voice", "input", "output", "sample_rate", "chunk_ms",
               "algorithmic_latency_ms", "processing", "rtf", "level_in", "level_out", "bypass", "muted", "fallbacks",
               "errors", "modulation"}


def test_snapshot_schema_and_live_control():
    eng = DspEngine(get_preset("natural").params())
    p = LivePipeline(LiveConfig(mode="modulation", engine="dsp", post_preset="natural"), eng)
    p.start()
    p.run_array(_speech(0.2), SR)
    snap = p.snapshot()
    assert STATUS_KEYS <= set(snap)
    json.dumps(snap)
    p.apply_control({"preset": "masculine", "strength": 1.0, "bypass": True})
    assert eng.mod.params.pitch_semitones == pytest.approx(get_preset("masculine").params(1.0).pitch_semitones)
    assert p.bypass
    ctl = send_control(strength=0.5)
    assert ctl["strength"] == 0.5 and "seq" in ctl
    assert read_status() is None  # nothing running in this test


def test_passthrough_pipeline_is_transparent_after_delay():
    p = LivePipeline(LiveConfig(mode="modulation", engine="passthrough"), PassthroughEngine())
    p.start()
    x = _speech(0.5) * 0.3
    y = p.run_array(x, SR)
    assert np.allclose(y[: x.size], x, atol=1e-4)


# --- CLI ----------------------------------------------------------------------------------------------


def test_cli_modulate_and_vc(tmp_path):
    from aivoice.audio_io import write_wav
    from aivoice.cli import main

    r = CliRunner()
    out = r.invoke(main, ["modulate", "preset", "list"])
    assert out.exit_code == 0 and "masculine-natural" in out.output
    out = r.invoke(main, ["vc", "engines", "--json"])
    assert out.exit_code == 0
    names = {e["name"]: e["state"] for e in json.loads(out.output)}
    assert names["dsp"] == "READY" and names["zerovc-repro"] in ("MISSING_WEIGHTS", "EXPERIMENTAL")
    src = tmp_path / "in.wav"
    write_wav(src, _speech(0.6), SR)
    dst = tmp_path / "out.wav"
    out = r.invoke(main, ["modulate", "masculine", "--strength", "65", "--file", str(src), "-o", str(dst)])
    assert out.exit_code == 0, out.output
    assert dst.is_file() and "f0_median_hz" in out.output
    out = r.invoke(main, ["modulate", "nosuchpreset", "--file", str(src)])
    assert out.exit_code != 0
    out = r.invoke(main, ["engines", "list"])
    assert out.exit_code == 0 and "aivoice reproduce" in out.output and "Live voice conversion" in out.output
    out = r.invoke(main, ["vc", "status", "--json"])
    assert json.loads(out.output) == {"running": False}


def test_benchmark_results_structure(tmp_path):
    from aivoice.vc.bench import Clip, format_summary, measured_delay_ms, run_vc_benchmark

    x = _speech(1.0)
    d = measured_delay_ms(x * (np.arange(x.size) % 9600 < 6000), np.concatenate([np.zeros(4800, np.float32),
                          x * (np.arange(x.size) % 9600 < 6000)]), SR)
    assert d == pytest.approx(100.0, abs=10)
    res = run_vc_benchmark(["dsp", "passthrough", "rtvc"], [Clip("s", x, SR)], [], tmp_path, record_results=False)
    assert set(res["summary"]) == {"dsp", "passthrough"} and "rtvc" in res["skipped"]
    root = Path(res["out_dir"])
    assert (root / "results.json").is_file() and (root / "results.csv").is_file()
    man = json.loads((root / "manifest.json").read_text())
    assert man["loudness_lufs"] == -23.0 and man["files"]
    assert "engine" in format_summary(res)


def test_worker_vc_engine_is_abstract_base():
    assert issubclass(EchoWorkerEngine, WorkerVCEngine)


def test_breathiness_adds_measurable_aspiration_noise():
    x = harmonic(seconds=1.0)
    y0 = SpectralModulator(SR, ModulationParams(formant_ratio=1.0002)).process_all(x)
    y1 = SpectralModulator(SR, ModulationParams(formant_ratio=1.0002, breathiness=1.0)).process_all(x)
    ratio_db = 10 * np.log10(np.sum((y1 - y0) ** 2) / np.sum(y0**2))
    assert -18.0 < ratio_db < -3.0  # ~8 dB below the voice on speech; this LF-heavy test tone gives ~-13.6


def test_roughness_modulates_voiced_amplitude():
    x = harmonic(seconds=1.0)
    y = SpectralModulator(SR, ModulationParams(roughness=1.0)).process_all(x)
    assert np.linalg.norm(y - x) / np.linalg.norm(x) > 0.05
