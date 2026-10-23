"""Shared live streaming path for direct VC and modulation.

    physical mic -> input gain -> [pre: denoise / light pitch-formant normalisation] -> resample ->
    streaming engine (X-VC / RVC / MeanVC2 / DSP ...) -> resample -> [post: modulation] ->
    limiter -> aivoice_sink -> aivoice-virtual-mic

One virtual microphone for every engine: switching engine never changes what Discord/OBS/VRChat
record from. If the engine raises, times out or its worker dies, the stream switches to the next
engine in the fallback chain (default: DSP with the current modulation, else passthrough), so the
microphone output never goes silent. Nothing here sends audio off the machine.

Live state is written to ``$XDG_RUNTIME_DIR/aivoice/vc.json`` once a second (the QuickShell widget
reads it); parameter changes are picked up from ``vc-control.json`` without restarting the engine.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..audio.resample import StreamingResampler
from ..paths import runtime_dir
from .base import BaseEngine, StreamOptions, VoiceTarget

PIPE_SR = 48000  # capture / output rate (PipeWire native); engines run at their own rate


def status_path() -> Path:
    return runtime_dir() / "vc.json"


def control_path() -> Path:
    return runtime_dir() / "vc-control.json"


def pid_path() -> Path:
    return runtime_dir() / "vc.pid"


@dataclass
class LiveConfig:
    mode: str = "voice_conversion"  # voice_conversion | modulation
    engine: str = "dsp"
    voice: str | None = None
    quality: str = "balanced"
    input_device: str | None = None
    output_sink: str | None = "aivoice_sink"
    input_gain_db: float = 0.0
    output_gain_db: float = 0.0
    pre: dict[str, Any] = field(default_factory=dict)  # ModulationParams fields applied before the engine
    post_preset: str | None = None
    post_strength: float = 0.0
    post_overrides: dict[str, Any] = field(default_factory=dict)
    fallback: list[str] = field(default_factory=lambda: ["dsp"])
    block_ms: float = 10.0
    prebuffer_ms: float = 30.0
    stream_options: StreamOptions = field(default_factory=StreamOptions)
    status_interval_s: float = 1.0
    restart_primary_after_s: float = 10.0
    max_restarts: int = 3


@dataclass
class LiveStats:
    started: float = field(default_factory=time.time)
    blocks: int = 0
    in_samples: int = 0
    out_samples: int = 0
    # bounded windows for percentiles (memory stays flat over long sessions); busy_ms is the running total
    proc_ms: deque = field(default_factory=lambda: deque(maxlen=20000))
    engine_ms: deque = field(default_factory=lambda: deque(maxlen=20000))
    busy_ms: float = 0.0
    fallbacks: list[dict[str, Any]] = field(default_factory=list)
    errors: deque = field(default_factory=lambda: deque(maxlen=50))
    nan_blocks: int = 0
    level_in: float = 0.0
    level_out: float = 0.0
    restarts: int = 0

    def window(self, n: int = 500) -> dict[str, Any]:
        def pct(v: list[float]) -> dict[str, float]:
            if not v:
                return {}
            a = np.asarray(list(v)[-n:])
            return {"p50": round(float(np.percentile(a, 50)), 2), "p95": round(float(np.percentile(a, 95)), 2),
                    "p99": round(float(np.percentile(a, 99)), 2), "max": round(float(a.max()), 2)}
        return {"block_ms": pct(self.proc_ms), "engine_ms": pct(self.engine_ms)}


def _pending_fade(n: int) -> np.ndarray:
    return np.sin(0.5 * np.pi * np.linspace(0, 1, n)).astype(np.float32) ** 2


class LivePipeline:
    """Owns the engine chain and the per-block DSP. ``feed()`` is the whole per-block path (used live and offline)."""

    def __init__(self, cfg: LiveConfig, engine: BaseEngine, *, target: VoiceTarget | None = None,
                 fallback_factory: Callable[[str], BaseEngine] | None = None,
                 on_event: Callable[[str, dict[str, Any]], None] | None = None) -> None:
        from ..modulation.dsp import ModulationParams, SpectralModulator

        self.cfg = cfg
        self.target = target
        self.on_event = on_event
        self.fallback_factory = fallback_factory or _default_fallback
        self.primary_name = engine.capabilities().name
        self.engine = engine
        self.active_name = self.primary_name
        self.stats = LiveStats()
        self.bypass = False
        self.muted = False
        self._lock = threading.Lock()
        self.pre = None
        pre = ModulationParams.from_dict({**(cfg.pre or {}), "limiter": False})
        if not pre.is_identity():
            self.pre = SpectralModulator(PIPE_SR, pre)
        self._fifo = np.zeros(0, np.float32)
        self.post = None
        self._set_post(cfg.post_preset, cfg.post_strength, cfg.post_overrides)
        self.in_gain = 10 ** (cfg.input_gain_db / 20)
        self.out_gain = 10 ** (cfg.output_gain_db / 20)
        self._fade_in: np.ndarray | None = None
        self._setup_resamplers()
        self._dead_since: float | None = None
        self._failed_primary = False
        self._stop = threading.Event()

    # ------------------------------------------------------------------ setup

    def _set_post(self, preset: str | None, strength: float, overrides: dict[str, Any] | None) -> None:
        from ..modulation.dsp import SpectralModulator
        from ..modulation.presets import get_preset

        if not preset or (strength <= 0 and not overrides):
            self.post = None
            return
        p = get_preset(preset)
        params = p.params(strength, overrides or {})
        if self.post is None:
            self.post = SpectralModulator(PIPE_SR, params, window_ms=p.window_ms, hop_ms=p.window_ms / 4)
        else:
            self.post.set_params(params)

    def _setup_resamplers(self) -> None:
        sr = self.engine.capabilities().sample_rate or PIPE_SR
        self.engine_sr = sr
        self.to_engine = StreamingResampler(PIPE_SR, sr)
        self.from_engine = StreamingResampler(sr, PIPE_SR)

    def start(self) -> None:
        self.engine.load()
        self.engine.prepare_reference(self.target)
        self.engine.start_stream(self.cfg.stream_options)
        self._emit("engine", {"engine": self.active_name, "state": "running"})

    # ------------------------------------------------------------------ per block

    def feed(self, block: np.ndarray) -> np.ndarray:
        """One capture block (float32 @ 48 kHz) in, the same number of output samples out (zero-padded while priming)."""
        t0 = time.perf_counter()
        x = np.asarray(block, np.float32).reshape(-1)
        n = x.size
        if not np.all(np.isfinite(x)):
            x = np.nan_to_num(x)
            self.stats.nan_blocks += 1
        x = x * self.in_gain
        self.stats.level_in = 0.9 * self.stats.level_in + 0.1 * float(np.sqrt(np.mean(x * x)) if n else 0.0)
        if self.muted:
            y = np.zeros(n, np.float32)
        elif self.bypass:
            y = x
        else:
            if self.pre is not None:
                x = self.pre.process(x)
            y = self._engine_block(x)
            if self.post is not None:
                y = self.post.process(y)
        y = self._fit(y, n) * self.out_gain
        if not np.all(np.isfinite(y)):
            y = np.nan_to_num(y)
            self.stats.nan_blocks += 1
        from ..audio.playback import soft_limit

        y = soft_limit(y.astype(np.float32), knee=0.9)
        self.stats.level_out = 0.9 * self.stats.level_out + 0.1 * float(np.sqrt(np.mean(y * y)) if n else 0.0)
        self.stats.blocks += 1
        self.stats.in_samples += n
        self.stats.out_samples += y.size
        self.stats.proc_ms.append((time.perf_counter() - t0) * 1000)
        self.stats.busy_ms += self.stats.proc_ms[-1]
        return y

    def _fit(self, y: np.ndarray, n: int) -> np.ndarray:
        """Engines emit in their own chunk rhythm; a FIFO turns that into one output block per input block."""
        self._fifo = np.concatenate([self._fifo, y])
        if self._fifo.size >= n:
            out, self._fifo = self._fifo[:n], self._fifo[n:]
            if self._fifo.size > PIPE_SR // 2:  # > 500 ms backlog: the engine emitted a burst; keep latency bounded
                self.stats.errors.append(f"output backlog {self._fifo.size / PIPE_SR * 1000:.0f} ms trimmed")
                self._fifo = self._fifo[-n:]
            return out
        return np.zeros(n, np.float32)  # priming / underrun of engine output

    def _engine_block(self, x: np.ndarray) -> np.ndarray:
        try:
            e_in = self.to_engine.process(x)
            t = time.perf_counter()
            e_out = self.engine.process_chunk(e_in) if e_in.size else np.zeros(0, np.float32)
            self.stats.engine_ms.append((time.perf_counter() - t) * 1000)
            y = self.from_engine.process(e_out)
        except Exception as e:  # noqa: BLE001 - any engine failure falls back, the mic never goes silent
            self._fail_over(e)
            return self._engine_block_safe(x)
        if self._fade_in is not None and y.size:
            f = self._fade_in[: y.size]
            y = y.copy()
            y[: f.size] *= f
            self._fade_in = self._fade_in[f.size :] if self._fade_in.size > f.size else None
        return y

    def _engine_block_safe(self, x: np.ndarray) -> np.ndarray:
        try:
            return self.from_engine.process(self.engine.process_chunk(self.to_engine.process(x)))
        except Exception as e:  # noqa: BLE001
            self._fail_over(e)
            return x  # last resort inside this block: dry signal

    def _fail_over(self, err: BaseException) -> None:
        failed = self.active_name
        msg = f"{failed}: {type(err).__name__}: {err}"
        self.stats.errors.append(msg[-400:])
        try:
            self.engine.unload()
        except Exception:  # noqa: BLE001
            pass
        chain = [n for n in self.cfg.fallback if n != failed] + ["passthrough"]
        for name in chain:
            try:
                eng = self.fallback_factory(name)
                eng.load()
                eng.prepare_reference(self.target)
                eng.start_stream(self.cfg.stream_options)
            except Exception as e2:  # noqa: BLE001
                self.stats.errors.append(f"fallback {name} failed: {e2}"[-400:])
                continue
            self.engine = eng
            self.active_name = name
            self._setup_resamplers()
            self._fifo = np.zeros(0, np.float32)
            self._fade_in = _pending_fade(int(0.02 * PIPE_SR))
            ev = {"t": time.time(), "from": failed, "to": name, "reason": msg[-200:]}
            self.stats.fallbacks.append(ev)
            if failed == self.primary_name:
                self._failed_primary = True
                self._dead_since = time.monotonic()
            self._emit("fallback", ev)
            return
        raise RuntimeError(f"no fallback engine could start after {msg}")

    def maybe_restart_primary(self, factory: Callable[[], BaseEngine]) -> bool:
        """Called from the supervisor thread: bring the selected engine back after a crash."""
        if not self._failed_primary or self._dead_since is None:
            return False
        if self.stats.restarts >= self.cfg.max_restarts:
            return False
        if time.monotonic() - self._dead_since < self.cfg.restart_primary_after_s:
            return False
        self.stats.restarts += 1
        try:
            eng = factory()
            eng.load()
            eng.prepare_reference(self.target)
            eng.start_stream(self.cfg.stream_options)
        except Exception as e:  # noqa: BLE001
            self.stats.errors.append(f"restart {self.primary_name} failed: {e}"[-400:])
            self._dead_since = time.monotonic()
            return False
        with self._lock:
            old = self.engine
            self.engine, self.active_name = eng, self.primary_name
            self._setup_resamplers()
            self._fifo = np.zeros(0, np.float32)
            self._fade_in = _pending_fade(int(0.02 * PIPE_SR))
            self._failed_primary = False
        try:
            old.unload()
        except Exception:  # noqa: BLE001
            pass
        self._emit("engine", {"engine": self.primary_name, "state": "restarted"})
        return True

    def apply_control(self, ctl: dict[str, Any]) -> None:
        """Live parameter change (widget sliders): modulation preset/strength/overrides, bypass, mute, gains."""
        if "bypass" in ctl:
            self.bypass = bool(ctl["bypass"])
        if "mute" in ctl:
            self.muted = bool(ctl["mute"])
        if "input_gain_db" in ctl:
            self.in_gain = 10 ** (float(ctl["input_gain_db"]) / 20)
        if "output_gain_db" in ctl:
            self.out_gain = 10 ** (float(ctl["output_gain_db"]) / 20)
        if any(k in ctl for k in ("preset", "strength", "overrides")):
            from .engines import DspEngine
            from ..modulation.presets import get_preset

            preset = ctl.get("preset", self.cfg.post_preset)
            strength = float(ctl.get("strength", self.cfg.post_strength))
            overrides = ctl.get("overrides", self.cfg.post_overrides) or {}
            self.cfg.post_preset, self.cfg.post_strength, self.cfg.post_overrides = preset, strength, overrides
            if self.cfg.mode == "modulation" and isinstance(self.engine, DspEngine) and preset:
                self.engine.set_params(get_preset(preset).params(strength, overrides))
            else:
                self._set_post(preset, strength, overrides)

    def _emit(self, kind: str, data: dict[str, Any]) -> None:
        if self.on_event:
            try:
                self.on_event(kind, data)
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------ offline

    def run_array(self, x: np.ndarray, sr: int) -> np.ndarray:
        """Stream a clip through exactly the live per-block path (no device clock). Returns 48 kHz audio."""
        from ..audio.resample import resample

        a = resample(np.asarray(x, np.float32), sr, PIPE_SR) if sr != PIPE_SR else np.asarray(x, np.float32)
        b = int(PIPE_SR * self.cfg.block_ms / 1000)
        a = np.concatenate([a, np.zeros(PIPE_SR, np.float32)])  # 1 s tail to drain engine latency
        return np.concatenate([self.feed(a[i : i + b]) for i in range(0, a.size, b)])

    # ------------------------------------------------------------------ status

    def snapshot(self) -> dict[str, Any]:
        caps = self.engine.capabilities()
        up = time.time() - self.stats.started
        audio_s = self.stats.in_samples / PIPE_SR
        busy_s = self.stats.busy_ms / 1000
        out = {
            "pid": os.getpid(), "running": True, "mode": self.cfg.mode, "engine": self.active_name, "selected_engine": self.primary_name,
            "engine_display": caps.display_name, "voice": self.cfg.voice, "input": self.cfg.input_device,
            "output": self.cfg.output_sink, "quality": self.cfg.quality, "sample_rate": PIPE_SR, "engine_sample_rate": self.engine_sr,
            "block_ms": self.cfg.block_ms, "chunk_ms": caps.chunk_ms, "algorithmic_latency_ms": caps.algorithmic_latency_ms,
            "pipeline_extra_latency_ms": round(self._pipeline_extra_ms(), 2),
            "processing": self.stats.window(), "rtf": round(busy_s / audio_s, 4) if audio_s else None,
            "uptime_s": round(up, 1), "audio_s": round(audio_s, 1), "bypass": self.bypass, "muted": self.muted,
            "level_in": round(min(1.0, self.stats.level_in * 6), 3), "level_out": round(min(1.0, self.stats.level_out * 6), 3),
            "fallbacks": self.stats.fallbacks[-5:], "errors": list(self.stats.errors)[-5:], "nan_blocks": self.stats.nan_blocks,
            "restarts": self.stats.restarts,
            "engine_info": {k: v for k, v in (getattr(self.engine, "load_info", {}) or {}).items()
                            if k in ("rvc_version", "f0", "model_sr", "chunk_samples", "lookahead_frames")},
            "modulation": {"preset": self.cfg.post_preset, "strength": self.cfg.post_strength, "overrides": self.cfg.post_overrides},
            "t": time.time(),
        }
        return out

    def _pipeline_extra_ms(self) -> float:
        ms = self.to_engine.delay_out * 1000 / max(1, self.engine_sr) + self.from_engine.delay_out * 1000 / PIPE_SR
        if self.pre is not None:
            ms += self.pre.latency_ms
        if self.post is not None:
            ms += self.post.latency_ms
        return ms


def _default_fallback(name: str) -> BaseEngine:
    from .engines import get_vc_engine

    return get_vc_engine(name)


# =============================================================================
# device runner
# =============================================================================


class LiveRunner:
    """Capture thread -> processing thread -> player -> sink, plus status / control / supervisor."""

    def __init__(self, pipeline: LivePipeline, *, capture=None, driver_kind: str = "auto",
                 primary_factory: Callable[[], BaseEngine] | None = None) -> None:
        from ..audio.playback import Player

        self.p = pipeline
        self.capture = capture
        self.driver_kind = driver_kind
        self.primary_factory = primary_factory
        self.player = Player(PIPE_SR, prebuffer_ms=int(pipeline.cfg.prebuffer_ms))
        self.player.rebuffer_on_underrun = True
        self.prebuffer_ms = float(pipeline.cfg.prebuffer_ms)
        self.calibrated_ms = 0.0  # measured real-chunk burst: the buffer never adapts below it
        self.driver = None
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self.gpu: dict[str, Any] = {}
        self.worker_stats: dict[str, Any] = {}

    def start(self) -> None:
        from ..audio.capture import make_capture
        from ..audio.playback import make_driver

        cfg = self.p.cfg
        self.p.start()
        self._calibrate_prebuffer()
        if self.capture is None:
            self.capture = make_capture(device=cfg.input_device, sample_rate=PIPE_SR, block_ms=int(cfg.block_ms))
        self.capture.start()
        self.driver = make_driver(self.player, kind=self.driver_kind, sink=cfg.output_sink)
        self.driver.start()
        for fn, name in ((self._process_loop, "aivoice-vc-proc"), (self._status_loop, "aivoice-vc-status")):
            t = threading.Thread(target=fn, name=name, daemon=True)
            t.start()
            self._threads.append(t)

    MAX_PREBUFFER_MS = 200.0

    def _calibrate_prebuffer(self) -> None:
        """Engines that emit in chunks (MeanVC2 160 ms, RVC 150-350 ms) stall the per-block loop for their whole
        compute time once per chunk. The output buffer must cover that burst or every chunk underruns
        (measured with MeanVC2: 26 underruns in 25 s at a fixed 30 ms). Time real chunks on the loaded
        engine, size the prebuffer to the burst, then reset the stream."""
        caps = self.p.engine.capabilities()
        chunk_ms = caps.chunk_ms or 0
        if chunk_ms <= self.p.cfg.block_ms or caps.name in ("dsp", "passthrough"):
            return
        n = int((self.p.engine_sr or PIPE_SR) * chunk_ms / 1000)
        times = []
        # noise above any noise gate (RVC skips inference on silence, which would measure nothing)
        probe = (np.random.default_rng(0).standard_normal(n) * 0.05).astype(np.float32)
        try:
            for _ in range(5):  # the first calls pay one-time CUDA/cuDNN setup: keep the last three
                t = time.perf_counter()
                self.p.engine.process_chunk(probe)
                times.append((time.perf_counter() - t) * 1000)
            worst = max(times[2:])
            self.p.engine.start_stream(self.p.cfg.stream_options)
        except Exception as e:  # noqa: BLE001
            self.p.stats.errors.append(f"prebuffer calibration: {e}"[-300:])
            return
        self.calibrated_ms = worst * 1.25 + 10
        self._set_prebuffer(self.calibrated_ms)

    def _set_prebuffer(self, ms: float) -> None:
        # never below the calibrated burst: in silence a gated engine (RVC) computes nothing, and shrinking to
        # that would underrun the first words after a pause
        ms = min(self.MAX_PREBUFFER_MS, max(float(self.p.cfg.prebuffer_ms), self.calibrated_ms, ms))
        if abs(ms - self.prebuffer_ms) >= 2:
            self.prebuffer_ms = ms
            self.player.set_prebuffer_ms(ms)

    def _adapt_prebuffer(self) -> None:
        """Track the engine's real bursts: grow quickly when p99 compute nears the buffer, shrink slowly."""
        w = self.p.stats.window(300).get("engine_ms") or {}
        if w.get("p99") is None:
            return
        # cover the recent worst burst too: one rare 55 ms spike over a 57 ms buffer is an audible click
        want = max(w["p99"] * 1.25 + 10, w["max"] * 1.05 + 5)
        if want > self.prebuffer_ms:
            self._set_prebuffer(want)
        elif want < self.prebuffer_ms - 10:
            self._set_prebuffer(max(want, self.prebuffer_ms - 5))

    def _process_loop(self) -> None:
        while not self._stop.is_set():
            item = self.capture.read(timeout=0.1)
            if item is None:
                if getattr(self.capture, "eof", False):
                    self.player.end("stream")
                    self._stop.set()
                continue
            _, block = item
            with self.p._lock:
                y = self.p.feed(block)
            self.player.write("stream", y)
            self.player.trim_quiet("stream", self.prebuffer_ms + 30.0)

    def _status_loop(self) -> None:
        last_gpu = 0.0
        while not self._stop.is_set():
            try:
                ctl = _read_json(control_path())
                if ctl and ctl.get("seq") != getattr(self, "_ctl_seq", None):
                    self._ctl_seq = ctl.get("seq")
                    with self.p._lock:
                        self.p.apply_control(ctl)
                if self.primary_factory is not None:
                    self.p.maybe_restart_primary(self.primary_factory)
                self._adapt_prebuffer()
                now = time.monotonic()
                if now - last_gpu > 5.0:  # nvidia-smi / worker stats every 5 s, not every block
                    last_gpu = now
                    self.gpu = gpu_usage()
                    st = getattr(self.p.engine, "stats", None)
                    self.worker_stats = st() if callable(st) else {}
                snap = self._full_snapshot()
                _write_json(status_path(), snap)
            except Exception as e:  # noqa: BLE001
                self.p.stats.errors.append(f"status: {e}")
            self._stop.wait(self.p.cfg.status_interval_s)

    def _full_snapshot(self) -> dict[str, Any]:
        snap = self.p.snapshot()
        snap.update({"underruns": self.player.underruns, "trimmed_ms": round(getattr(self.player, "trimmed", 0) * 1000 / PIPE_SR, 1), "dropped_blocks": getattr(self.capture, "dropped", 0),
                     "capture_overflows": getattr(self.capture, "overflows", 0),
                     "output_buffered_ms": round(self.player.buffered_ms(), 1), "gpu": self.gpu,
                     "worker": self.worker_stats, "prebuffer_ms": round(self.prebuffer_ms, 1),
                     "end_to_end_estimate_ms": round((snap.get("algorithmic_latency_ms") or 0) + snap.get("pipeline_extra_latency_ms", 0)
                                                     + self.prebuffer_ms + self.p.cfg.block_ms, 1)})
        return snap

    def wait(self, duration_s: float | None = None) -> None:
        end = time.monotonic() + duration_s if duration_s else None
        while not self._stop.is_set():
            if end and time.monotonic() >= end:
                break
            self._stop.wait(0.2)

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=2)
        for x in (self.capture, self.driver):
            try:
                if x is not None:
                    x.stop()
            except Exception:  # noqa: BLE001
                pass
        try:
            self.p.engine.unload()
        except Exception:  # noqa: BLE001
            pass
        snap = self._full_snapshot()
        snap["running"] = False
        _write_json(status_path(), snap)


def _read_json(p: Path) -> dict[str, Any] | None:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_json(p: Path, data: dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, p)


def send_control(**values: Any) -> dict[str, Any]:
    cur = _read_json(control_path()) or {}
    cur.update({k: v for k, v in values.items() if v is not None})
    cur["seq"] = time.time_ns()
    _write_json(control_path(), cur)
    return cur


def read_status() -> dict[str, Any] | None:
    s = _read_json(status_path())
    if not s:
        return None
    pid = s.get("pid")
    alive = False
    if pid:
        try:
            os.kill(int(pid), 0)
            alive = True
        except (ProcessLookupError, ValueError):
            alive = False
        except PermissionError:
            alive = True
    s["running"] = bool(s.get("running")) and alive
    return s


def gpu_usage() -> dict[str, Any]:
    """GPU utilisation / memory from nvidia-smi (cheap; called every few seconds)."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return {}
    try:
        r = subprocess.run([exe, "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3)
        name, util, used, total = [v.strip() for v in r.stdout.splitlines()[0].split(",")]
        return {"name": name, "util_pct": float(util), "mem_used_mib": float(used), "mem_total_mib": float(total)}
    except Exception:  # noqa: BLE001
        return {}


__all__ = ["LiveConfig", "LivePipeline", "LiveRunner", "PIPE_SR", "control_path", "gpu_usage", "pid_path", "read_status",
           "send_control", "status_path"]
