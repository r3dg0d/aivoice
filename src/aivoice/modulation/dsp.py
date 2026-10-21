"""Streaming spectral voice modulation: independent pitch and formant control, plus voice-quality stages.

One short-time Fourier transform carries every stage, so adding a stage adds no latency:

* **Pitch** - phase-vocoder pitch shifting with identity phase locking (Laroche & Dolson 1999),
  done on the *excitation* (spectrum divided by its envelope), so moving the harmonics does not
  move the formants.
* **Formants / vocal-tract length** - the spectral envelope (iterative cepstral "true envelope",
  Röbel & Rodet 2005) is warped in frequency independently of pitch: ``env_out(f) = env(f / ratio)``.
* **Intonation** - an optional pitch-*range* scale around a running median F0 (per-frame F0
  from a YIN-style cumulative-mean-normalised difference, de Cheveigné & Kawahara 2002).
* **Voice quality** - body/resonance (low peaking band), brightness (high shelf), spectral tilt,
  aspiration noise shaped by the envelope (breathiness), and slow amplitude modulation on
  voiced frames (roughness).
* **Cleanup / dynamics** - optional spectral-floor noise suppression, de-esser, feed-forward
  compressor and a soft limiter (time domain, sample-wise, no lookahead).

Algorithmic delay is ``window - 1`` samples (32 ms with the default 32 ms window), constant for any
capture block size; measure it with :func:`measure_delay`. Bypass keeps that delay so A/B switching does not jump in time.

Everything is numpy: it runs on the CPU and is the always-available fallback for the live
pipeline. It is a real baseline, not a neural model: it changes acoustic attributes of the
same voice, it does not clone anyone.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from typing import Any

import numpy as np

TWO_PI = 2.0 * np.pi


@dataclass
class ModulationParams:
    """Attribute deltas. Zero / 1.0 everywhere is identity (only the STFT round trip remains)."""

    pitch_semitones: float = 0.0
    formant_ratio: float = 1.0  # < 1 lowers formants (longer apparent vocal tract)
    intonation: float = 1.0  # pitch-range scale around the running median; 1 = unchanged, 0 = monotone
    body_db: float = 0.0  # peaking band around 180 Hz (chest resonance / "voice body")
    resonance_db: float = 0.0  # peaking band around 700 Hz (oral resonance / warmth)
    brightness_db: float = 0.0  # high shelf from ~3.5 kHz
    tilt_db_per_oct: float = 0.0  # spectral tilt above 500 Hz (negative = darker)
    breathiness: float = 0.0  # 0..1 aspiration noise added under the envelope
    roughness: float = 0.0  # 0..1 depth of 25-60 Hz amplitude modulation on voiced frames
    robotic: float = 0.0  # 0..1 monotone + phase flattening
    band_low_hz: float = 0.0  # band-limit (telephone / radio); 0 = off
    band_high_hz: float = 0.0
    saturation: float = 0.0  # 0..1 soft-clip drive (radio character)
    denoise: float = 0.0  # 0..1 spectral-floor suppression strength
    deess: float = 0.0  # 0..1 de-esser strength
    compress: float = 0.0  # 0..1 compressor amount
    output_gain_db: float = 0.0
    limiter: bool = True
    mix: float = 1.0  # wet/dry (dry is delay-matched)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModulationParams":
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"unknown modulation parameter(s): {', '.join(sorted(unknown))}")
        return cls(**d)

    def is_identity(self) -> bool:
        return self == ModulationParams(limiter=self.limiter)

    @property
    def pitch_ratio(self) -> float:
        return float(2.0 ** (self.pitch_semitones / 12.0))


@dataclass
class DspStats:
    frames: int = 0
    voiced_frames: int = 0
    f0_center_hz: float | None = None
    last_f0_hz: float = 0.0
    gain_reduction_db: float = 0.0
    nan_repairs: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


def _lifter_envelope(logmag: np.ndarray, n_fft: int, order: int, iters: int) -> np.ndarray:
    """Iterative cepstral 'true envelope' of a log-magnitude spectrum (natural log)."""
    target = logmag
    env = None
    for _ in range(max(1, iters)):
        c = np.fft.irfft(target, n=n_fft)
        c[order:n_fft - order + 1] = 0.0
        env = np.fft.rfft(c, n=n_fft).real
        target = np.maximum(logmag, env)
    return env  # type: ignore[return-value]


def _yin_frame(x: np.ndarray, sr: int, fmin: float, fmax: float, thresh: float = 0.15) -> tuple[float, float]:
    """(f0 Hz or 0, aperiodicity 0..1) for one frame; difference function computed with FFTs."""
    n = x.size
    tmax = min(int(sr / fmin), n // 2)
    tmin = max(2, int(sr / fmax))
    if tmax <= tmin + 2:
        return 0.0, 1.0
    x = x - x.mean()
    energy = float(np.dot(x, x))
    if energy < 1e-7 * n:
        return 0.0, 1.0
    w = n - tmax
    size = 1 << int(np.ceil(np.log2(n + w)))
    fx = np.fft.rfft(x, size)
    fw = np.fft.rfft(x[:w][::-1], size)
    corr = np.fft.irfft(fx * fw, size)[w - 1 : w - 1 + tmax + 1]
    sq = np.cumsum(np.concatenate([[0.0], x * x]))
    e0 = sq[w]
    et = sq[np.arange(tmax + 1) + w] - sq[np.arange(tmax + 1)]
    d = e0 + et - 2.0 * corr
    d[0] = 0.0
    cm = np.cumsum(d[1:])
    cmnd = np.ones_like(d)
    cmnd[1:] = d[1:] * np.arange(1, d.size) / np.maximum(cm, 1e-12)
    cand = np.where(cmnd[tmin:tmax] < thresh)[0]
    if cand.size == 0:
        t = int(np.argmin(cmnd[tmin:tmax])) + tmin
        return 0.0, float(min(1.0, cmnd[t]))
    t = int(cand[0]) + tmin
    while t + 1 < tmax and cmnd[t + 1] < cmnd[t]:
        t += 1
    a, b, c = cmnd[t - 1], cmnd[t], cmnd[t + 1] if t + 1 < cmnd.size else cmnd[t]
    den = a - 2 * b + c
    tf = t + (0.5 * (a - c) / den if abs(den) > 1e-12 else 0.0)
    return float(sr / tf), float(min(1.0, max(0.0, cmnd[t])))


class SpectralModulator:
    """Streaming modulation of mono float32 audio. ``process()`` accepts blocks of any size."""

    def __init__(self, sample_rate: int = 48000, params: ModulationParams | None = None, *,
                 window_ms: float = 32.0, hop_ms: float = 8.0, phase_locking: bool | None = None,
                 envelope_iters: int = 3, seed: int = 0) -> None:
        self.sr = int(sample_rate)
        self.n = int(round(self.sr * window_ms / 1000))
        self.n += self.n % 2
        self.hop = int(round(self.sr * hop_ms / 1000))
        if self.n % self.hop:
            self.hop = self.n // max(1, round(self.n / self.hop))
        self.bins = self.n // 2 + 1
        self.win = np.hanning(self.n + 1)[:-1].astype(np.float64)  # periodic Hann
        # Hann analysis * Hann synthesis at this overlap sums to a constant; normalise by it.
        ola = np.zeros(self.n + self.hop * (self.n // self.hop))
        for s in range(0, ola.size - self.n + 1, self.hop):
            ola[s : s + self.n] += self.win**2
        self.ola_gain = float(ola[self.n : self.n + self.hop].mean()) or 1.0
        self.omega = TWO_PI * np.arange(self.bins) / self.n  # rad/sample per bin
        self.freqs = np.arange(self.bins) * self.sr / self.n
        # cepstral order: below the quefrency of a 450 Hz voice so harmonics are not resolved
        self.cep_order = max(8, int(self.sr / 450.0 * 0.8))
        self.env_iters = envelope_iters
        # Experiment R: identity phase locking lowers aperiodicity at 32/48 ms windows but raises it at 16 ms.
        self.phase_locking = (window_ms >= 24.0) if phase_locking is None else phase_locking
        self.params = params or ModulationParams()
        self.stats = DspStats()
        self._rng = np.random.default_rng(seed)
        self._shape_cache: tuple[Any, np.ndarray] | None = None
        self.bypass = False
        self.reset()

    # ------------------------------------------------------------------ state

    def reset(self) -> None:
        self._inbuf = np.zeros(self.n, np.float64)
        # hop-1 samples of prefill: with any block size the queue can never run dry mid-stream
        # (a dry queue would insert silence inside the signal), at a constant delay of window-1 samples.
        self._out_q = np.zeros(self.hop - 1, np.float64)
        self._pending = np.zeros(0, np.float64)
        self._outacc = np.zeros(self.n, np.float64)
        self._prev_phase = np.zeros(self.bins)
        self._out_phase = np.zeros(self.bins)
        self._noise_floor = None
        self._ratio_smooth = None
        self._log_f0_center = None
        self._f0_hist: list[float] = []
        self._f0_hist_len = max(20, int(1.2 * self.sr / self.hop))
        self._voiced_smooth = 0.0
        self._am_phase = 0.0
        self._am_state = 0.0
        self._comp_env = 0.0
        self._dry_delay = np.zeros(self.latency_samples, np.float64)
        self.stats = DspStats()

    @property
    def latency_samples(self) -> int:
        return self.n - 1

    @property
    def latency_ms(self) -> float:
        return 1000.0 * self.latency_samples / self.sr

    def set_params(self, params: ModulationParams) -> None:
        """Change parameters between blocks; per-frame ratio smoothing avoids jumps."""
        self.params = params

    # ------------------------------------------------------------------ public

    def process(self, block: np.ndarray) -> np.ndarray:
        x = np.asarray(block, dtype=np.float64).reshape(-1)
        if x.size == 0:
            return np.zeros(0, np.float32)
        if not np.all(np.isfinite(x)):
            self.stats.nan_repairs += 1
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        dry = np.concatenate([self._dry_delay, x])
        self._dry_delay = dry[x.size :] if self._dry_delay.size else self._dry_delay
        dry = dry[: x.size]
        buf = np.concatenate([self._pending, x])
        out_parts = []
        pos = 0
        while buf.size - pos >= self.hop:
            hop_in = buf[pos : pos + self.hop]
            pos += self.hop
            self._inbuf = np.concatenate([self._inbuf[self.hop :], hop_in])
            out_parts.append(self._frame(self._inbuf))
        self._pending = buf[pos:]
        wet = np.concatenate(out_parts) if out_parts else np.zeros(0)
        # Keep output length == input length: emit what is ready, hold the remainder as latency.
        self._out_q = np.concatenate([self._out_q, wet])
        need = x.size
        y, self._out_q = self._out_q[:need], self._out_q[need:]  # never short: see the prefill in reset()
        p = self.params
        if self.bypass:
            y = dry
        elif p.mix < 1.0:
            y = p.mix * y + (1.0 - p.mix) * dry
        y = self._post(y)
        return y.astype(np.float32)

    def process_all(self, x: np.ndarray, *, block: int | None = None, align: bool = True) -> np.ndarray:
        """Offline helper: stream *x* through in blocks and (optionally) remove the algorithmic delay."""
        self.reset()
        b = block or self.hop
        x = np.asarray(x, np.float64)
        tail = np.zeros(self.latency_samples + b, np.float64)
        xx = np.concatenate([x, tail])
        out = np.concatenate([self.process(xx[i : i + b]) for i in range(0, xx.size, b)])
        if align:
            out = out[self.latency_samples : self.latency_samples + x.size]
        return out.astype(np.float32)

    # ------------------------------------------------------------------ core

    def _shape_curve(self) -> np.ndarray:
        """Static log-gain curve (natural log units) for EQ / tilt / band limits; cached per params."""
        p = self.params
        key = (p.body_db, p.resonance_db, p.brightness_db, p.tilt_db_per_oct, p.band_low_hz, p.band_high_hz)
        if self._shape_cache is not None and self._shape_cache[0] == key:
            return self._shape_cache[1]
        f = np.maximum(self.freqs, 1.0)
        db = np.zeros(self.bins)

        def peak(fc: float, oct_bw: float, gain: float) -> np.ndarray:
            d = np.log2(f / fc) / (oct_bw / 2)
            return gain * np.exp(-0.5 * d * d * 2.0)

        if p.body_db:
            db += peak(180.0, 1.4, p.body_db)
        if p.resonance_db:
            db += peak(700.0, 1.2, p.resonance_db)
        if p.brightness_db:
            db += p.brightness_db / (1.0 + (3500.0 / f) ** 4)
        if p.tilt_db_per_oct:
            db += p.tilt_db_per_oct * np.maximum(0.0, np.log2(f / 500.0))
        if p.band_low_hz > 0:
            db += 10 * np.log10(1.0 / (1.0 + (p.band_low_hz / f) ** 8))
        if p.band_high_hz > 0:
            db += 10 * np.log10(1.0 / (1.0 + (f / p.band_high_hz) ** 8))
        curve = db * (np.log(10.0) / 20.0)
        self._shape_cache = (key, curve)
        return curve

    def _frame(self, frame: np.ndarray) -> np.ndarray:
        p = self.params
        st = self.stats
        st.frames += 1
        X = np.fft.rfft(frame * self.win)
        mag = np.abs(X)
        ph = np.angle(X)

        # true instantaneous frequency per bin (rad/sample)
        dphi = ph - self._prev_phase - self.omega * self.hop
        dphi -= TWO_PI * np.round(dphi / TWO_PI)
        inst = self.omega + dphi / self.hop
        self._prev_phase = ph

        # running F0 for intonation / roughness gating
        f0, aper = _yin_frame(frame[-min(frame.size, int(self.sr * 0.032)) :], self.sr, 65.0, 600.0)
        voiced = f0 > 0 and aper < 0.25
        self._voiced_smooth = 0.8 * self._voiced_smooth + 0.2 * (1.0 if voiced else 0.0)
        if voiced:
            st.voiced_frames += 1
            st.last_f0_hz = f0
            # centre = running median of the last ~1.2 s of voiced frames: unbiased from the first frame on
            self._f0_hist.append(math.log(f0))
            if len(self._f0_hist) > self._f0_hist_len:
                self._f0_hist.pop(0)
            self._log_f0_center = float(np.median(self._f0_hist))
            st.f0_center_hz = math.exp(self._log_f0_center)

        ratio = p.pitch_ratio
        intonation = p.intonation * (1.0 - p.robotic)
        if voiced and self._log_f0_center is not None and abs(intonation - 1.0) > 1e-6:
            ratio *= math.exp((intonation - 1.0) * (math.log(f0) - self._log_f0_center))
        if self._ratio_smooth is None:
            self._ratio_smooth = ratio
        self._ratio_smooth = 0.7 * self._ratio_smooth + 0.3 * ratio
        a = float(np.clip(self._ratio_smooth, 0.25, 4.0))

        eps = 1e-9
        logmag = np.log(mag + eps)
        need_env = abs(a - 1.0) > 1e-4 or abs(p.formant_ratio - 1.0) > 1e-4 or p.breathiness > 0
        order = self._cep_order()
        if need_env:
            env = _lifter_envelope(logmag, self.n, order, self.env_iters)
        else:
            env = None

        if p.denoise > 0:
            pw = mag * mag
            if self._noise_floor is None:
                self._noise_floor = pw.copy()
            else:  # slow-rise / fast-fall floor tracker (minimum-statistics style)
                self._noise_floor = np.where(pw < self._noise_floor, 0.7 * self._noise_floor + 0.3 * pw,
                                             self._noise_floor * 1.002 + 1e-12)
            gain = np.maximum(0.0, 1.0 - p.denoise * 1.5 * self._noise_floor / (pw + 1e-12))
            gain = np.maximum(gain, 0.1)
            logmag = logmag + np.log(np.sqrt(gain) + eps)
            if env is not None:
                env = _lifter_envelope(logmag, self.n, order, self.env_iters)

        breath = None
        if env is None:
            out_log = logmag.copy()
            out_inst = inst
            src_idx = np.arange(self.bins)
        else:
            exc = logmag - env
            j = np.arange(self.bins)
            k = j / a  # inverse map: output bin j takes the excitation at input bin j / a
            valid = k <= self.bins - 1
            kc = np.clip(k, 0, self.bins - 1)
            exc_s = np.interp(kc, j, exc)
            exc_s[~valid] = np.log(eps)
            src_idx = np.clip(np.round(kc).astype(int), 0, self.bins - 1)
            out_inst = a * inst[src_idx]
            envw = np.interp(np.clip(j / max(p.formant_ratio, 1e-3), 0, self.bins - 1), j, env)
            envw[j / max(p.formant_ratio, 1e-3) > self.bins - 1] = env[-1]
            out_log = exc_s + envw
            if p.breathiness > 0:
                # aspiration: random-phase noise under the (warped) envelope, weighted to >1.5 kHz.
                # Incoherent frames lose energy in overlap-add relative to the voice; the gain is calibrated by
                # measurement so breathiness 1.0 puts the noise ~8 dB below the voice (0.36 -> ~17 dB below).
                hf = 1.0 / (1.0 + (1500.0 / np.maximum(self.freqs, 1.0)) ** 2)
                breath = np.exp(envw) * hf * (2.7 * p.breathiness)

        out_log = out_log + self._shape_curve()

        if p.deess > 0:
            band = (self.freqs > 4500) & (self.freqs < 10000)
            if band.any():
                e_band = float(np.sum(np.exp(2 * out_log[band])))
                e_all = float(np.sum(np.exp(2 * out_log))) + 1e-12
                frac = e_band / e_all
                if frac > 0.35:
                    cut = min(12.0, p.deess * 20.0 * (frac - 0.35) / 0.65 * 2)
                    out_log[band] -= cut * np.log(10) / 20

        # phase synthesis
        omag = np.exp(out_log)
        if abs(a - 1.0) < 1e-4 and env is None:
            phase = ph  # identity: keep analysis phase (perfect reconstruction)
            self._out_phase = ph
        else:
            prop = self._out_phase + self.hop * out_inst
            if self.phase_locking:
                phase = self._locked_phase(omag, prop, ph, src_idx)
            else:
                phase = prop
            self._out_phase = phase
        if p.robotic > 0:
            phase = (1.0 - p.robotic) * phase  # flattening toward zero phase: buzzy, monotone character
        Y = omag * np.exp(1j * phase)
        if breath is not None:
            rnd = self._rng.uniform(0.0, TWO_PI, self.bins)
            Y = Y + breath * np.exp(self._shape_curve()) * np.exp(1j * rnd) * (1.0 if voiced else 0.3)
        y = np.fft.irfft(Y, n=self.n) * self.win / self.ola_gain
        self._outacc += y
        out = self._outacc[: self.hop].copy()
        self._outacc = np.concatenate([self._outacc[self.hop :], np.zeros(self.hop)])
        if p.roughness > 0:
            out = self._roughen(out, p.roughness)
        return out

    def _cep_order(self) -> int:
        """Lifter order tied to the voice's pitch period: as detailed as possible without resolving harmonics."""
        f0 = self.stats.f0_center_hz or 0.0
        if f0 <= 0:
            return self.cep_order
        return int(np.clip(0.75 * self.sr / f0, self.sr / 600.0, self.sr / 90.0))

    def _locked_phase(self, omag: np.ndarray, prop: np.ndarray, ph_in: np.ndarray, src_idx: np.ndarray) -> np.ndarray:
        """Identity phase locking (Laroche & Dolson 1999): propagate phase at spectral peaks and keep
        each peak region's input phase relations. (Matching peaks across frames was tried and measured
        worse on aperiodicity, see docs/voice-modulation.md, so peaks continue the per-bin phase.)"""
        m = omag
        if m.size < 5:
            return prop
        pk = np.where((m[1:-1] > m[:-2]) & (m[1:-1] >= m[2:]) & (m[1:-1] > 1e-6 * m.max()))[0] + 1
        if pk.size == 0:
            return prop
        bounds = np.concatenate([[0], (pk[:-1] + pk[1:]) // 2 + 1, [m.size]])
        owner = np.repeat(pk, np.diff(bounds))
        return prop[owner] + (ph_in[src_idx] - ph_in[src_idx[owner]])

    def _roughen(self, out: np.ndarray, depth: float) -> np.ndarray:
        n = out.size
        # band-limited random modulator (~25-60 Hz): one-pole smoothed random walk on a 40 Hz carrier
        t = self._am_phase + TWO_PI * 40.0 * np.arange(n) / self.sr
        self._am_phase = float((t[-1] + TWO_PI * 40.0 / self.sr) % TWO_PI) if n else self._am_phase
        jitter = self._rng.normal(0.0, 1.0)
        self._am_state = 0.9 * self._am_state + 0.1 * jitter
        mod = np.sin(t + self._am_state) * (0.5 + 0.5 * abs(self._am_state))
        g = 1.0 + depth * 0.45 * self._voiced_smooth * mod
        return out * g

    def _post(self, y: np.ndarray) -> np.ndarray:
        p = self.params
        if p.saturation > 0:
            drive = 1.0 + 6.0 * p.saturation
            y = np.tanh(y * drive) / np.tanh(drive)
        if p.compress > 0:
            # feed-forward RMS compressor, 5 ms attack / 80 ms release, threshold -26 dBFS
            thr = 10 ** (-26 / 20)
            ratio = 1.0 + 3.0 * p.compress
            att = math.exp(-1.0 / (0.005 * self.sr))
            rel = math.exp(-1.0 / (0.080 * self.sr))
            env = self._comp_env
            out = np.empty_like(y)
            # block-wise envelope (per 1 ms) keeps this cheap in numpy
            step = max(1, self.sr // 1000)
            makeup = (thr ** (1 - 1 / ratio)) ** -0.5
            for s in range(0, y.size, step):
                seg = y[s : s + step]
                lvl = math.sqrt(float(np.mean(seg * seg)) + 1e-12)
                coef = att if lvl > env else rel
                coef = coef**step
                env = coef * env + (1 - coef) * lvl
                g = 1.0 if env <= thr else (env / thr) ** (1 / ratio - 1)
                out[s : s + step] = seg * g * makeup
            self._comp_env = env
            self.stats.gain_reduction_db = float(-20 * math.log10(g * makeup)) if env > thr else 0.0
            y = out
        if p.output_gain_db:
            y = y * 10 ** (p.output_gain_db / 20)
        if p.limiter:
            from ..audio.playback import soft_limit

            y = soft_limit(y.astype(np.float32), knee=0.89).astype(np.float64)
        return y


def measure_delay(mod: SpectralModulator, *, n: int | None = None) -> int:
    """Measured end-to-end algorithmic delay (samples) of *mod* with its current parameters."""
    n = n or mod.sr // 2
    rng = np.random.default_rng(1)
    x = rng.normal(0, 0.1, n)
    saved = mod.params
    mod.set_params(ModulationParams(limiter=False))
    mod.reset()
    y = np.concatenate([mod.process(x[i : i + mod.hop]) for i in range(0, n, mod.hop)])
    mod.set_params(saved)
    mod.reset()
    c = np.correlate(y[: n - 1], x[: n // 2], "valid")
    return int(np.argmax(c))


__all__ = ["DspStats", "ModulationParams", "SpectralModulator", "measure_delay"]
