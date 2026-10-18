"""Voice activity detection and utterance endpointing.

* :class:`SileroVAD` runs the ONNX model that ships inside the ``silero-vad``
  pip package directly with onnxruntime (no torch import, no download). It
  follows upstream's 16 kHz framing: 512-sample windows with the previous 64
  samples prepended as context, recurrent state carried between calls.
* :class:`EnergyVAD` is a dependency-free fallback: frame energy against an
  adaptive noise floor. Good enough for a close mic, poor in noise.
* :class:`Endpointer` turns per-frame probabilities into utterance start/end
  events with hysteresis, pre-roll padding and a hangover, and force-splits
  very long utterances so a monologue still reaches TTS.
"""

from __future__ import annotations

import collections
import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class VADError(RuntimeError):
    pass


def silero_onnx_path() -> Path | None:
    env = os.environ.get("AIVOICE_SILERO_ONNX")
    if env and Path(env).is_file():
        return Path(env)
    spec = importlib.util.find_spec("silero_vad")  # locate only; importing pulls torch
    if spec and spec.submodule_search_locations:
        for loc in spec.submodule_search_locations:
            p = Path(loc) / "data" / "silero_vad.onnx"
            if p.is_file():
                return p
    return None


class SileroVAD:
    name = "silero"
    window = 512
    context = 64

    def __init__(self, path: Path | None = None, *, sample_rate: int = 16000, threads: int = 1) -> None:
        if sample_rate != 16000:
            raise VADError("aivoice runs Silero at 16 kHz only")
        p = path or silero_onnx_path()
        if p is None:
            raise VADError("silero_vad.onnx not found: pip install silero-vad (only the data file is used)")
        try:
            import onnxruntime as ort  # type: ignore
        except ImportError as e:
            raise VADError("onnxruntime is not installed: pip install onnxruntime") from e
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(p), sess_options=so, providers=["CPUExecutionProvider"])
        self.sample_rate = sample_rate
        self._sr = np.array(sample_rate, dtype=np.int64)
        self._pending = np.zeros(0, np.float32)
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._ctx = np.zeros((1, self.context), dtype=np.float32)
        self._pending = np.zeros(0, np.float32)

    def __call__(self, block: np.ndarray) -> list[float]:
        """Probabilities for each complete 32 ms window in *block* (+ carry-over)."""
        x = np.concatenate([self._pending, np.asarray(block, np.float32)])
        probs: list[float] = []
        n = (x.size // self.window) * self.window
        for i in range(0, n, self.window):
            w = x[i : i + self.window][None, :]
            inp = np.concatenate([self._ctx, w], axis=1)
            out, self._state = self.sess.run(None, {"input": inp, "state": self._state, "sr": self._sr})
            self._ctx = inp[:, -self.context :]
            probs.append(float(out[0][0]))
        self._pending = x[n:]
        return probs

    frame_ms = 32.0


class EnergyVAD:
    name = "energy"
    frame_ms = 20.0

    def __init__(self, *, sample_rate: int = 16000, margin_db: float = 12.0, floor_db: float = -60.0) -> None:
        self.sample_rate = sample_rate
        self.frame = int(sample_rate * self.frame_ms / 1000)
        self.margin_db = margin_db
        self.min_floor = floor_db
        self.reset()

    def reset(self) -> None:
        self.noise_db = -50.0
        self._pending = np.zeros(0, np.float32)

    def __call__(self, block: np.ndarray) -> list[float]:
        x = np.concatenate([self._pending, np.asarray(block, np.float32)])
        probs = []
        n = (x.size // self.frame) * self.frame
        for i in range(0, n, self.frame):
            f = x[i : i + self.frame]
            db = 10 * np.log10(float(np.mean(f * f)) + 1e-12)
            # Noise floor follows quiet frames quickly and loud frames very slowly.
            rate = 0.05 if db < self.noise_db else 0.002
            self.noise_db = max(self.min_floor, self.noise_db + rate * (db - self.noise_db))
            snr = db - self.noise_db
            probs.append(float(1 / (1 + np.exp(-(snr - self.margin_db) / 2.5))))
        self._pending = x[n:]
        return probs


def make_vad(engine: str = "auto", *, sample_rate: int = 16000):
    if engine in ("auto", "silero"):
        try:
            return SileroVAD(sample_rate=sample_rate)
        except VADError:
            if engine == "silero":
                raise
    return EnergyVAD(sample_rate=sample_rate)


@dataclass
class VADEvent:
    kind: str  # start | audio | end
    audio: np.ndarray
    t: float  # monotonic timestamp of the block that produced the event
    forced: bool = False


class Endpointer:
    """Hysteresis endpointing on top of a frame-level VAD."""

    def __init__(
        self,
        vad,
        *,
        threshold: float = 0.5,
        min_silence_ms: int = 300,
        speech_pad_ms: int = 120,
        min_speech_ms: int = 120,
        max_utterance_s: float = 15.0,
        sample_rate: int = 16000,
    ) -> None:
        self.vad = vad
        self.on = threshold
        self.off = max(0.05, threshold - 0.15)
        self.min_silence_ms = min_silence_ms
        self.min_speech_ms = min_speech_ms
        self.max_utterance_s = max_utterance_s
        self.sr = sample_rate
        self._preroll: collections.deque[np.ndarray] = collections.deque()
        self._preroll_max = int(sample_rate * speech_pad_ms / 1000)
        self.in_speech = False
        self._silence_ms = 0.0
        self._speech_ms = 0.0
        self._utt_samples = 0
        self.last_prob = 0.0

    def reset(self) -> None:
        self.vad.reset()
        self._preroll.clear()
        self.in_speech = False
        self._silence_ms = self._speech_ms = 0.0
        self._utt_samples = 0

    def _preroll_audio(self) -> np.ndarray:
        return np.concatenate(list(self._preroll)) if self._preroll else np.zeros(0, np.float32)

    def process(self, block: np.ndarray, t: float) -> list[VADEvent]:
        probs = self.vad(block)
        block_ms = block.size * 1000 / self.sr
        if probs:
            self.last_prob = max(probs)
        p = self.last_prob
        events: list[VADEvent] = []
        if not self.in_speech:
            self._preroll.append(block)
            while sum(b.size for b in self._preroll) > self._preroll_max + block.size:
                self._preroll.popleft()
            if p >= self.on:
                self._speech_ms += block_ms
                if self._speech_ms >= self.min_speech_ms:
                    self.in_speech = True
                    self._silence_ms = 0.0
                    audio = self._preroll_audio()
                    self._preroll.clear()
                    self._utt_samples = audio.size
                    events.append(VADEvent("start", audio, t))
            else:
                self._speech_ms = 0.0
            return events
        # in speech
        self._utt_samples += block.size
        events.append(VADEvent("audio", block, t))
        if p < self.off:
            self._silence_ms += block_ms
        else:
            self._silence_ms = 0.0
        if self._silence_ms >= self.min_silence_ms:
            events.append(VADEvent("end", np.zeros(0, np.float32), t))
            self._end()
        elif self._utt_samples / self.sr >= self.max_utterance_s:
            # Force a split and continue straight into a new utterance.
            events.append(VADEvent("end", np.zeros(0, np.float32), t, forced=True))
            self._end()
            self.in_speech = True
            events.append(VADEvent("start", np.zeros(0, np.float32), t, forced=True))
        return events

    def _end(self) -> None:
        self.in_speech = False
        self._silence_ms = 0.0
        self._speech_ms = 0.0
        self._utt_samples = 0
