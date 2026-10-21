"""Low-latency playback with a per-utterance jitter buffer, cancellation, ducking and a limiter.

Synthesized audio arrives in bursts. The player holds a new utterance until
``prebuffer_ms`` of it is queued (or its end has arrived), then plays it
continuously; running dry *inside* an utterance is counted as an underrun.
Gain changes (ducking, volume) ramp over one block so they never click, and a
soft limiter keeps peaks under full scale instead of hard clipping.

The device side is a driver that calls :meth:`Player.pull` from the audio
callback: :class:`SoundDeviceDriver` for real output (e.g. the aivoice virtual
mic sink), :class:`ClockDriver` for tests and benchmarks.
"""

from __future__ import annotations

import collections
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

END = object()


@dataclass
class PlaybackEvent:
    kind: str  # first_audio | done | cancelled | underrun
    uid: str
    t: float


@dataclass
class _Utt:
    uid: str
    chunks: collections.deque = field(default_factory=collections.deque)
    queued: int = 0
    ended: bool = False
    started: bool = False
    played: int = 0


def soft_limit(x: np.ndarray, knee: float = 0.9) -> np.ndarray:
    a = np.abs(x)
    over = a > knee
    if not over.any():
        return x
    y = x.copy()
    headroom = 1.0 - knee
    y[over] = np.sign(x[over]) * (knee + headroom * np.tanh((a[over] - knee) / headroom))
    return y


class Player:
    def __init__(
        self,
        sample_rate: int,
        *,
        prebuffer_ms: int = 60,
        gain_db: float = 0.0,
        on_event: Callable[[PlaybackEvent], None] | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.prebuffer = int(sample_rate * prebuffer_ms / 1000)
        self._lock = threading.Lock()
        self._utts: collections.deque[_Utt] = collections.deque()
        self._by_id: dict[str, _Utt] = {}
        self._gain = 10 ** (gain_db / 20)
        self._target_gain = self._gain
        self._duck = 1.0
        self._target_duck = 1.0
        self.underruns = 0
        self.frames_out = 0
        self.peak = 0.0
        self.on_event = on_event
        self._events: collections.deque[PlaybackEvent] = collections.deque(maxlen=1024)

    # --- producer side ---------------------------------------------------

    def write(self, uid: str, pcm: np.ndarray) -> None:
        pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if pcm.size == 0:
            return
        with self._lock:
            u = self._by_id.get(uid)
            if u is None:
                u = _Utt(uid)
                self._by_id[uid] = u
                self._utts.append(u)
            if u.ended:
                return
            u.chunks.append(pcm)
            u.queued += pcm.size

    def end(self, uid: str) -> None:
        with self._lock:
            u = self._by_id.get(uid)
            if u is None:
                u = _Utt(uid)
                self._by_id[uid] = u
                self._utts.append(u)
            u.ended = True

    def cancel(self, uid: str | None = None) -> list[str]:
        """Drop one utterance (or everything when uid is None). Returns cancelled ids."""
        with self._lock:
            targets = [u for u in self._utts if uid is None or u.uid == uid]
            for u in targets:
                self._utts.remove(u)
                self._by_id.pop(u.uid, None)
        now = time.monotonic()
        for u in targets:
            self._emit(PlaybackEvent("cancelled", u.uid, now))
        return [u.uid for u in targets]

    flush = cancel

    def trim_quiet(self, uid: str, target_ms: float, *, quiet_rms: float = 0.01) -> int:
        """Bound latency on a continuous stream: drop queued *quiet* chunks while more than *target_ms* is buffered.

        Two independent clocks (capture and playback) drift, and every underrun adds latency for good;
        dropping silence brings it back without cutting into speech. Returns samples dropped.
        """
        dropped = 0
        limit = int(self.sample_rate * target_ms / 1000)
        with self._lock:
            u = self._by_id.get(uid)
            if u is None or not u.started:
                return 0
            keep: collections.deque = collections.deque()
            while u.chunks and u.queued > limit:
                c = u.chunks.popleft()
                if c.size and float(np.sqrt(np.mean(c * c))) < quiet_rms and u.queued - c.size >= limit // 2:
                    u.queued -= c.size
                    dropped += c.size
                else:
                    keep.append(c)
                    break
            while keep:
                u.chunks.appendleft(keep.pop())
        self.trimmed = getattr(self, "trimmed", 0) + dropped
        return dropped

    def set_gain_db(self, db: float) -> None:
        self._target_gain = 10 ** (db / 20)

    def duck(self, on: bool, db: float = -12.0) -> None:
        self._target_duck = 10 ** (db / 20) if on else 1.0

    @property
    def is_playing(self) -> bool:
        with self._lock:
            return any(u.started for u in self._utts)

    @property
    def pending(self) -> int:
        with self._lock:
            return len(self._utts)

    def buffered_ms(self) -> float:
        with self._lock:
            return sum(u.queued for u in self._utts) * 1000 / self.sample_rate

    def events(self) -> list[PlaybackEvent]:
        out = []
        while self._events:
            out.append(self._events.popleft())
        return out

    def _emit(self, ev: PlaybackEvent) -> None:
        self._events.append(ev)
        if self.on_event:
            try:
                self.on_event(ev)
            except Exception:  # noqa: BLE001
                pass

    # --- consumer side (audio thread) -----------------------------------

    def pull(self, frames: int) -> np.ndarray:
        out = np.zeros(frames, dtype=np.float32)
        pos = 0
        now = time.monotonic()
        with self._lock:
            while pos < frames and self._utts:
                u = self._utts[0]
                if not u.started:
                    if u.queued < self.prebuffer and not u.ended:
                        break  # still buffering this utterance
                    u.started = True
                    if u.queued:
                        self._emit(PlaybackEvent("first_audio", u.uid, now))
                if not u.chunks:
                    if u.ended:
                        self._utts.popleft()
                        self._by_id.pop(u.uid, None)
                        self._emit(PlaybackEvent("done", u.uid, now))
                        continue
                    self.underruns += 1
                    self._emit(PlaybackEvent("underrun", u.uid, now))
                    break
                chunk = u.chunks[0]
                n = min(frames - pos, chunk.size)
                out[pos : pos + n] = chunk[:n]
                pos += n
                u.queued -= n
                u.played += n
                if n == chunk.size:
                    u.chunks.popleft()
                else:
                    u.chunks[0] = chunk[n:]
        # Ramp gain/duck over this block to avoid zipper noise.
        g0 = self._gain * self._duck
        self._gain, self._duck = self._target_gain, self._target_duck
        g1 = self._gain * self._duck
        if pos:
            ramp = np.linspace(g0, g1, frames, dtype=np.float32) if g0 != g1 else g1
            out = soft_limit(out * ramp)
            self.peak = max(self.peak * 0.95, float(np.abs(out).max()))
        self.frames_out += frames
        return out


class ClockDriver:
    """Pulls from a player on a simulated (or real-time) device clock; collects the output."""

    def __init__(self, player: Player, *, block_ms: int = 10, realtime: bool = True, keep: bool = True) -> None:
        self.player = player
        self.block = int(player.sample_rate * block_ms / 1000)
        self.realtime = realtime
        self.keep = keep
        self.output: list[np.ndarray] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="aivoice-clock", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        t0 = time.monotonic()
        n = 0
        while not self._stop.is_set():
            block = self.player.pull(self.block)
            if self.keep:
                self.output.append(block)
            n += 1
            if self.realtime:
                delay = t0 + n * self.block / self.player.sample_rate - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
            else:
                time.sleep(0)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def audio(self) -> np.ndarray:
        return np.concatenate(self.output) if self.output else np.zeros(0, np.float32)


class SoundDeviceDriver:
    """Real output through PortAudio (PipeWire/Pulse); point ``device`` at the aivoice sink."""

    def __init__(self, player: Player, *, device: str | int | None = None, block_ms: int = 10) -> None:
        self.player = player
        self.device = device
        self.block = int(player.sample_rate * block_ms / 1000)
        self._stream = None

    def start(self) -> None:
        import sounddevice as sd  # type: ignore

        def cb(outdata, frames, t, status):
            if status and status.output_underflow:
                self.player.underruns += 1
            outdata[:, 0] = self.player.pull(frames)

        self._stream = sd.OutputStream(
            device=self.device,
            channels=1,
            samplerate=self.player.sample_rate,
            blocksize=self.block,
            dtype="float32",
            latency="low",
            callback=cb,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None


class PacatDriver:
    """Feeds a named PipeWire/Pulse sink through ``pacat`` (pipewire-pulse or PulseAudio).

    Targets the sink by name, which PortAudio cannot do reliably. Blocks are
    written on our own real-time clock, so the pipe never holds more than a few
    blocks of audio (no hidden 64 KiB pipe-buffer latency).
    """

    def __init__(self, player: Player, *, sink: str | None = None, block_ms: int = 10, latency_ms: int = 30) -> None:
        self.player = player
        self.sink = sink
        self.block = int(player.sample_rate * block_ms / 1000)
        self.latency_ms = latency_ms
        self._proc = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        import shutil
        import subprocess

        exe = shutil.which("pacat")
        if not exe:
            raise RuntimeError("pacat not found (install pulseaudio-utils / pipewire-pulse)")
        argv = [
            exe,
            "--playback",
            "--raw",
            "--format=float32le",
            f"--rate={self.player.sample_rate}",
            "--channels=1",
            f"--latency-msec={self.latency_ms}",
            "--client-name=aivoice",
            "--stream-name=aivoice-output",
        ]
        if self.sink:
            argv.append(f"--device={self.sink}")
        self._proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self._thread = threading.Thread(target=self._run, name="aivoice-pacat", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        t0 = time.monotonic()
        n = 0
        while not self._stop.is_set() and self._proc and self._proc.poll() is None:
            block = self.player.pull(self.block)
            try:
                self._proc.stdin.write(block.astype("<f4").tobytes())  # type: ignore[union-attr]
                self._proc.stdin.flush()  # type: ignore[union-attr]
            except (BrokenPipeError, OSError, ValueError):
                break
            n += 1
            delay = t0 + n * self.block / self.player.sample_rate - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            elif delay < -0.2:  # fell behind (suspend, overload): resync the clock
                t0, n = time.monotonic(), 0

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        if self._proc:
            try:
                self._proc.stdin.close()  # type: ignore[union-attr]
            except OSError:
                pass
            try:
                self._proc.terminate()
                self._proc.wait(timeout=2)
            except Exception:  # noqa: BLE001
                pass
            self._proc = None


def make_driver(player: Player, *, kind: str = "auto", sink: str | None = None, device=None):
    """pacat when a named sink is the target and pacat exists; else PortAudio."""
    import shutil

    if kind == "pacat" or (kind == "auto" and sink and shutil.which("pacat")):
        return PacatDriver(player, sink=sink)
    if kind == "clock":
        return ClockDriver(player, keep=False)
    return SoundDeviceDriver(player, device=device)
