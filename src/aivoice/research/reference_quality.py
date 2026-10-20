"""Reference audio analysis and conservative preparation for zero-shot cloning.

Pipeline: decode/validate -> silence analysis -> VAD -> (optional) denoise ->
clipping/distortion check -> quality report -> speaker-consistency check ->
engine-specific copy.

Every modifying step is guarded: when a speaker encoder is available, a step
whose output lowers speaker similarity to the original below ``min_identity``
is reverted. The original file is never modified; the prepared copy and a JSON
report are written next to it on request.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..audio.loudness import integrated_loudness, normalize, peak_dbfs
from ..audio.resample import resample
from .duration import silence_edges, speech_mask, speech_seconds


@dataclass
class ReferenceReport:
    path: str
    sha256: str
    sample_rate: int
    duration_s: float
    speech_s: float
    leading_silence_s: float
    trailing_silence_s: float
    peak_dbfs: float
    lufs: float
    clipped_fraction: float
    dc_offset: float
    snr_db: float | None
    effective_bandwidth_hz: float
    likely_upsampled: bool
    speaker_consistency: float | None  # min partial-vs-mean similarity (GE2E), None without encoder
    issues: list[str] = field(default_factory=list)
    score: float = 0.0  # 0..1 heuristic, for ranking clips, not a perceptual measure

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def effective_bandwidth(x: np.ndarray, sr: int, energy: float = 0.995) -> float:
    """Frequency below which `energy` of the spectrum lies (detects band-limited / upsampled audio)."""
    if x.size < 1024:
        return sr / 2
    n = 2048
    frames = np.lib.stride_tricks.sliding_window_view(x, n)[:: n // 2] * np.hanning(n)
    p = (np.abs(np.fft.rfft(frames, axis=1)) ** 2).mean(axis=0)
    c = np.cumsum(p) / max(p.sum(), 1e-20)
    k = int(np.searchsorted(c, energy))
    return float(k * sr / n)


def snr_estimate(x16: np.ndarray) -> float | None:
    mask, frame = speech_mask(x16)
    if mask.size == 0 or mask.all() or not mask.any():
        return None
    f = int(frame)
    n = min(mask.size, x16.size // f)
    seg = x16[: n * f].reshape(n, f)
    e = (seg**2).mean(axis=1)
    sp, ns = e[mask[:n]], e[~mask[:n]]
    if sp.size == 0 or ns.size == 0:
        return None
    noise = max(float(np.median(ns)), 1e-12)
    return float(10 * np.log10(max(float(np.median(sp)), 1e-12) / noise))


def analyze_reference(path: str | Path, *, encoder: Any | None = None) -> ReferenceReport:
    from ..audio_io import read_wav

    path = Path(path)
    x, sr = read_wav(path)
    if x.size == 0:
        raise ValueError(f"{path}: empty audio")
    x16 = resample(x, sr, 16000) if sr != 16000 else x
    lead, trail = silence_edges(x, sr)
    sp = speech_seconds(x, sr)
    clip = float(np.mean(np.abs(x) >= 0.999))
    snr = snr_estimate(x16)
    bw = effective_bandwidth(x, sr)
    upsampled = sr >= 32000 and bw < 0.45 * (sr / 2)
    consistency = None
    if encoder is not None and sp >= 3.0:
        try:
            e, partials = encoder.embed(x, sr, return_partials=True)
            sims = partials @ e
            consistency = float(np.min(sims))
        except ValueError:
            pass
    issues = []
    if sp < 3:
        issues.append(f"only {sp:.1f} s of speech; 5-15 s is the usual sweet spot")
    elif sp > 30:
        issues.append(f"{sp:.0f} s of speech; most engines use 5-30 s (aivoice can pick the best window)")
    if lead > 0.5 or trail > 0.5:
        issues.append(f"silence at edges (lead {lead:.2f} s, trail {trail:.2f} s): trimming recommended")
    if clip > 1e-4:
        issues.append(f"clipping on {clip:.3%} of samples")
    if snr is not None and snr < 20:
        issues.append(f"noisy: estimated SNR {snr:.0f} dB")
    if upsampled:
        issues.append(f"content stops near {bw:.0f} Hz although the file is {sr} Hz: likely upsampled from a lower rate")
    if abs(float(np.mean(x))) > 0.01:
        issues.append("DC offset")
    if consistency is not None and consistency < 0.6:
        issues.append(f"parts of the clip sound like a different speaker (min partial similarity {consistency:.2f})")
    lufs = integrated_loudness(x, sr)
    score = 1.0
    score -= 0.3 * (sp < 3) + 0.1 * (sp > 30)
    score -= min(0.3, clip * 300)
    if snr is not None:
        score -= max(0.0, min(0.3, (25 - snr) / 50))
    score -= 0.1 * upsampled + (0.3 if consistency is not None and consistency < 0.6 else 0)
    return ReferenceReport(
        path=str(path), sha256=_sha256(path), sample_rate=sr, duration_s=x.size / sr, speech_s=sp,
        leading_silence_s=lead, trailing_silence_s=trail, peak_dbfs=peak_dbfs(x), lufs=lufs,
        clipped_fraction=clip, dc_offset=float(np.mean(x)), snr_db=snr, effective_bandwidth_hz=bw,
        likely_upsampled=upsampled, speaker_consistency=consistency, issues=issues, score=max(0.0, round(score, 3)),
    )


def best_window(x: np.ndarray, sr: int, seconds: float) -> tuple[int, int]:
    """Sample range of the `seconds`-long window with the most speech (ties: earliest)."""
    n = int(seconds * sr)
    if x.size <= n:
        return 0, x.size
    x16 = resample(x, sr, 16000) if sr != 16000 else x
    mask, frame = speech_mask(x16)
    if mask.size == 0:
        return 0, n
    per = frame / 16000 * sr
    w = max(1, int(n / per))
    dens = np.convolve(mask.astype(float), np.ones(w), mode="valid")
    i = int(np.argmax(dens))
    a = int(i * per)
    return a, min(x.size, a + n)


def spectral_gate(x: np.ndarray, sr: int, *, reduction_db: float = 12.0) -> np.ndarray:
    """Mild stationary-noise reduction (noise profile from non-speech frames). Off by default."""
    n, hop = 1024, 256
    if x.size < n * 4:
        return x
    x16 = resample(x, sr, 16000) if sr != 16000 else x
    mask, frame = speech_mask(x16)
    win = np.hanning(n)
    pad = np.pad(x, (n, n))
    frames = np.lib.stride_tricks.sliding_window_view(pad, n)[::hop] * win
    S = np.fft.rfft(frames, axis=1)
    mag = np.abs(S)
    t = (np.arange(len(frames)) * hop - n) / sr
    if mask.size:
        fidx = np.clip((t * 16000 / frame).astype(int), 0, mask.size - 1)
        noise_frames = mag[~mask[fidx]]
    else:
        noise_frames = mag[: max(1, len(mag) // 10)]
    if noise_frames.shape[0] < 3:
        return x
    noise = np.percentile(noise_frames, 75, axis=0)
    floor = 10 ** (-reduction_db / 20)
    gain = np.clip(1 - noise / np.maximum(mag, 1e-12), floor, 1.0)
    gain = np.apply_along_axis(lambda g: np.convolve(g, np.ones(3) / 3, mode="same"), 1, gain)
    y_frames = np.fft.irfft(S * gain, n=n, axis=1) * win
    out = np.zeros(pad.size)
    norm = np.zeros(pad.size)
    for i, f in enumerate(y_frames):
        out[i * hop : i * hop + n] += f
        norm[i * hop : i * hop + n] += win**2
    out = out / np.maximum(norm, 1e-8)
    return out[n : n + x.size].astype(np.float32)


@dataclass
class PreparedReference:
    original: str
    prepared: str
    steps: list[str]
    reverted: dict[str, str]
    report_before: ReferenceReport
    report_after: ReferenceReport
    identity_similarity: float | None


def prepare_reference(
    path: str | Path,
    out_path: str | Path,
    *,
    max_seconds: float | None = 20.0,
    target_lufs: float = -20.0,
    denoise: bool | str = "auto",
    noisy_below_snr_db: float = 25.0,
    trim_pad_ms: int = 150,
    encoder: Any | None = None,
    min_identity: float = 0.90,
) -> PreparedReference:
    from ..audio_io import read_wav, write_wav

    path = Path(path)
    x, sr = read_wav(path)
    before = analyze_reference(path, encoder=encoder)
    steps: list[str] = []
    reverted: dict[str, str] = {}
    y = x.astype(np.float32)

    def guard(name: str, cand: np.ndarray) -> None:
        nonlocal y
        if encoder is not None:
            try:
                sim = encoder.similarity(x, sr, cand, sr)
            except ValueError:
                sim = None
            if sim is not None and sim < min_identity:
                reverted[name] = f"speaker similarity to original {sim:.3f} < {min_identity}"
                return
        y = cand
        steps.append(name)

    if abs(float(np.mean(y))) > 1e-3:
        guard("dc", (y - np.mean(y)).astype(np.float32))
    lead, trail = before.leading_silence_s, before.trailing_silence_s
    if lead > trim_pad_ms / 1000 or trail > trim_pad_ms / 1000:
        a = int(max(0.0, lead - trim_pad_ms / 1000) * sr)
        b = y.size - int(max(0.0, trail - trim_pad_ms / 1000) * sr)
        if b - a > sr:
            guard("trim", y[a:b])
    if max_seconds and y.size / sr > max_seconds:
        a, b = best_window(y, sr, max_seconds)
        guard(f"window{max_seconds:g}s", y[a:b])
    # "auto": only when the reference is measurably noisy (experiment C: denoising helped noisy
    # references and did not change clean ones); always identity-guarded.
    want_dn = denoise is True or (denoise == "auto" and before.snr_db is not None and before.snr_db < noisy_below_snr_db)
    if want_dn:
        guard("denoise", spectral_gate(y, sr))
    norm, info = normalize(y, sr, target_lufs=target_lufs)
    if abs(info["gain_db"]) >= 0.5:
        guard("loudness", norm)
    out_path = Path(out_path)
    write_wav(out_path, y, sr)
    after = analyze_reference(out_path, encoder=encoder)
    ident = None
    if encoder is not None and steps:
        try:
            ident = encoder.similarity(x, sr, y, sr)
        except ValueError:
            pass
    rep = PreparedReference(str(path), str(out_path), steps, reverted, before, after, ident)
    out_path.with_suffix(".json").write_text(json.dumps(asdict(rep), indent=2, default=str) + "\n", encoding="utf-8")
    return rep
