"""Sample-rate conversion.

Live streams avoid this entirely where they can: the capture stream is opened
at the ASR rate and the playback stream at the TTS rate, so PipeWire's own
resampler does the work on its real-time thread. This module covers files and
the cases where a device refuses the rate.
"""

from __future__ import annotations

from functools import lru_cache
from math import gcd

import numpy as np


@lru_cache(maxsize=16)
def _lowpass(up: int, down: int, taps_per_phase: int = 16) -> np.ndarray:
    cutoff = 0.5 / max(up, down)
    n = taps_per_phase * max(up, down) * 2 + 1
    t = np.arange(n) - (n - 1) / 2
    h = 2 * cutoff * np.sinc(2 * cutoff * t) * np.kaiser(n, 8.0)
    return (h * up / h.sum()).astype(np.float32)


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if sr_in == sr_out or x.size == 0:
        return x
    g = gcd(int(sr_in), int(sr_out))
    up, down = sr_out // g, sr_in // g
    try:
        from scipy.signal import resample_poly  # type: ignore

        return resample_poly(x, up, down).astype(np.float32)
    except ImportError:
        pass
    if up > 64 or down > 64:
        # Awkward ratio: linear interpolation (fine for ASR input, not for music).
        n_out = int(round(x.size * sr_out / sr_in))
        return np.interp(np.linspace(0, x.size - 1, n_out), np.arange(x.size), x).astype(np.float32)
    h = _lowpass(up, down)
    y = np.zeros(x.size * up, dtype=np.float32)
    y[::up] = x
    y = np.convolve(y, h, mode="same")
    return y[::down].astype(np.float32)


def int16_to_f32(raw: bytes) -> np.ndarray:
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def f32_to_int16(x: np.ndarray) -> bytes:
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


class StreamResampler:
    """Chunk-continuous linear-interpolation resampler for streamed TTS output.

    Keeps the fractional read position and the previous chunk's last sample, so
    chunk boundaries do not click. Linear interpolation is adequate for speech
    between common rates (24/44.1/48 kHz); it is only used when the output
    device runs at a different rate than the TTS.
    """

    def __init__(self, sr_in: int, sr_out: int) -> None:
        self.sr_in = sr_in
        self.sr_out = sr_out
        self.step = sr_in / sr_out
        self._pos = 0.0  # position in input samples, relative to _prev
        self._prev = np.zeros(1, np.float32)

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if self.sr_in == self.sr_out or x.size == 0:
            return x
        buf = np.concatenate([self._prev, x])  # buf[0] is the last sample of the previous chunk
        last = buf.size - 1
        n = int(np.floor((last - self._pos) / self.step)) + 1
        if n <= 0:
            self._pos -= x.size
            self._prev = buf[-1:]
            return np.zeros(0, np.float32)
        idx = self._pos + self.step * np.arange(n)
        out = np.interp(idx, np.arange(buf.size), buf).astype(np.float32)
        self._pos = idx[-1] + self.step - last
        self._prev = buf[-1:]
        return out


class StreamingResampler:
    """Chunk-continuous polyphase windowed-sinc resampler (Kaiser, beta 8) for live engine I/O.

    Unlike :class:`StreamResampler` (linear interpolation, for TTS playback) this keeps proper
    anti-aliasing for 48 kHz <-> 16 kHz around neural VC engines. Exact rational ratio; state is
    the input history needed by the filter and the fractional output phase, so chunk boundaries are
    seamless (output of any chunking equals the output of one call, to float precision).
    Group delay: ``taps_per_phase / 2`` input samples (``delay_out`` output samples).
    """

    def __init__(self, sr_in: int, sr_out: int, *, taps_per_phase: int = 24) -> None:
        g = gcd(int(sr_in), int(sr_out))
        self.up, self.down = int(sr_out) // g, int(sr_in) // g
        self.sr_in, self.sr_out = int(sr_in), int(sr_out)
        self.identity = self.up == self.down
        self.k = taps_per_phase
        if not self.identity:
            n = self.k * self.up
            cutoff = 0.5 / max(self.up, self.down) * 0.94
            t = np.arange(n) - (n - 1) / 2
            h = 2 * cutoff * np.sinc(2 * cutoff * t) * np.kaiser(n, 8.0) * self.up
            # phase p, tap j -> h[p + j*up]; reversed taps so a dot with history (oldest first) works
            self.bank = np.stack([h[p::self.up][: self.k][::-1] for p in range(self.up)]).astype(np.float64)
        self.reset()

    def reset(self) -> None:
        self.hist = np.zeros(self.k - 1, np.float64)
        self.t = 0  # next output index in units of 1/up input samples, relative to hist start + (k-1)

    @property
    def delay_out(self) -> float:
        return 0.0 if self.identity else (self.k * self.up - 1) / 2 / self.down

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, np.float32)
        if self.identity or x.size == 0:
            return x.copy()
        buf = np.concatenate([self.hist, x.astype(np.float64)])
        avail = buf.size - (self.k - 1)  # input samples that can end a filter window
        # output m uses input position pos = t + m*down (in 1/up units): sample index i = pos // up, phase p = pos % up
        last = avail * self.up  # exclusive bound in 1/up units
        if self.t >= last:
            self.hist = buf[-(self.k - 1):]
            self.t -= x.size * self.up
            return np.zeros(0, np.float32)
        m = np.arange((last - 1 - self.t) // self.down + 1)
        pos = self.t + m * self.down
        i, p = pos // self.up, pos % self.up
        idx = i[:, None] + np.arange(self.k)[None, :]
        y = np.einsum("mk,mk->m", buf[idx], self.bank[p])
        self.t = int(pos[-1] + self.down - x.size * self.up)
        self.hist = buf[-(self.k - 1):]
        return y.astype(np.float32)
