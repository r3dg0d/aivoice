"""Microphone / file capture into a bounded, non-blocking block queue.

The audio callback does one thing: copy the block and ``put_nowait`` it. If the
consumer falls behind, the oldest block is dropped (and counted) rather than
ever blocking the audio thread. Downstream work - VAD, ASR, everything - runs on
other threads and can never stall capture.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from pathlib import Path

import numpy as np

from .resample import resample


class CaptureError(RuntimeError):
    pass


class _BlockQueue:
    def __init__(self, max_blocks: int) -> None:
        self.q: queue.Queue = queue.Queue(maxsize=max_blocks)
        self.dropped = 0

    def put(self, block) -> None:
        try:
            self.q.put_nowait(block)
        except queue.Full:
            try:
                self.q.get_nowait()
            except queue.Empty:
                pass
            self.dropped += 1
            try:
                self.q.put_nowait(block)
            except queue.Full:
                self.dropped += 1

    def get(self, timeout: float | None):
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None


class MicCapture:
    """sounddevice InputStream at the ASR rate (PipeWire resamples on its RT thread)."""

    def __init__(
        self,
        *,
        device: str | int | None = None,
        sample_rate: int = 16000,
        block_ms: int = 20,
        max_queue_ms: int = 4000,
    ) -> None:
        self.device = device
        self.sample_rate = sample_rate
        self.block = int(sample_rate * block_ms / 1000)
        self._q = _BlockQueue(max(4, max_queue_ms // block_ms))
        self._stream = None
        self.overflows = 0
        self.eof = False

    @property
    def dropped(self) -> int:
        return self._q.dropped

    def start(self) -> None:
        try:
            from .portaudio import import_sounddevice

            sd = import_sounddevice()
        except (ImportError, OSError) as e:
            raise CaptureError(f"sounddevice/PortAudio unavailable: {e}. pip install 'aivoice[realtime]'") from e

        def cb(indata, frames, t, status):  # audio thread: keep it trivial
            if status and status.input_overflow:
                self.overflows += 1
            self._q.put((time.monotonic(), indata[:, 0].copy()))

        try:
            self._stream = sd.InputStream(
                device=self.device,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=self.block,
                dtype="float32",
                latency="low",
                callback=cb,
            )
            self._stream.start()
        except Exception as e:  # noqa: BLE001
            raise CaptureError(f"could not open input device {self.device!r}: {e}") from e

    def read(self, timeout: float | None = 0.1):
        """(monotonic_ts, float32 block) or None."""
        return self._q.get(timeout)

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None


class ArrayCapture:
    """Feeds a clip block by block, optionally paced like a live microphone."""

    def __init__(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        target_rate: int = 16000,
        block_ms: int = 20,
        realtime: bool = True,
        tail_silence_ms: int = 800,
    ) -> None:
        a = np.asarray(audio, dtype=np.float32)
        if a.ndim > 1:
            a = a.mean(axis=1)
        if sample_rate != target_rate:
            a = resample(a, sample_rate, target_rate)
        a = np.concatenate([a, np.zeros(int(target_rate * tail_silence_ms / 1000), np.float32)])
        self.audio = a
        self.sample_rate = target_rate
        self.block = int(target_rate * block_ms / 1000)
        self.realtime = realtime
        self._pos = 0
        self._t0: float | None = None
        self.dropped = 0
        self.overflows = 0
        self.eof = False
        self._stop = threading.Event()

    @classmethod
    def from_file(cls, path: Path, **kw) -> "ArrayCapture":
        from ..audio_io import read_wav

        pcm, sr = read_wav(Path(path))
        return cls(pcm, sr, **kw)

    def start(self) -> None:
        self._t0 = time.monotonic()

    def read(self, timeout: float | None = 0.1):
        if self._stop.is_set() or self._pos >= self.audio.size:
            self.eof = True
            if timeout:
                time.sleep(min(timeout, 0.01))
            return None
        if self._t0 is None:
            self.start()
        if self.realtime:
            due = self._t0 + (self._pos + self.block) / self.sample_rate  # block is "captured" when it ends
            delay = due - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        block = self.audio[self._pos : self._pos + self.block]
        self._pos += self.block
        if block.size < self.block:
            block = np.pad(block, (0, self.block - block.size))
        return (time.monotonic(), block)

    def position_s(self) -> float:
        return self._pos / self.sample_rate

    def stop(self) -> None:
        self._stop.set()


class ParecCapture:
    """Reads a named PipeWire/Pulse source through ``parec`` (no PortAudio needed)."""

    def __init__(
        self,
        *,
        source: str | None = None,
        sample_rate: int = 16000,
        block_ms: int = 20,
        max_queue_ms: int = 4000,
    ) -> None:
        self.source = source
        self.sample_rate = sample_rate
        self.block = int(sample_rate * block_ms / 1000)
        self._q = _BlockQueue(max(4, max_queue_ms // block_ms))
        self._proc = None
        self._thread: threading.Thread | None = None
        self.overflows = 0
        self.eof = False

    @property
    def dropped(self) -> int:
        return self._q.dropped

    def start(self) -> None:
        import shutil
        import subprocess

        exe = shutil.which("parec")
        if not exe:
            raise CaptureError("parec not found (install pulseaudio-utils / pipewire-pulse)")
        argv = [
            exe, "--raw", "--format=float32le", f"--rate={self.sample_rate}", "--channels=1",
            f"--latency-msec={int(os.environ.get('AIVOICE_PAREC_LATENCY_MS', '10'))}", "--client-name=aivoice", "--stream-name=aivoice-input",
        ]
        if self.source:
            argv.append(f"--device={self.source}")
        self._proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self._thread = threading.Thread(target=self._run, name="aivoice-parec", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        nbytes = self.block * 4
        out = self._proc.stdout  # type: ignore[union-attr]
        while True:
            raw = out.read(nbytes)
            if not raw or len(raw) < nbytes:
                self.eof = True
                return
            self._q.put((time.monotonic(), np.frombuffer(raw, dtype="<f4").copy()))

    def read(self, timeout: float | None = 0.1):
        return self._q.get(timeout)

    def stop(self) -> None:
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except Exception:  # noqa: BLE001
                pass
            self._proc = None


def make_capture(*, kind: str = "auto", device=None, sample_rate: int = 16000, block_ms: int = 20):
    import shutil

    # parec whenever PipeWire/Pulse tools exist and no PortAudio index was asked for (device None = default
    # source): it needs no PortAudio, which NixOS python wheels cannot find on their own.
    if kind == "parec" or (kind == "auto" and shutil.which("parec") and not (isinstance(device, int) or (isinstance(device, str) and device.isdigit()))):
        return ParecCapture(source=device, sample_rate=sample_rate, block_ms=block_ms)
    return MicCapture(device=int(device) if isinstance(device, str) and device.isdigit() else device,
                      sample_rate=sample_rate, block_ms=block_ms)
