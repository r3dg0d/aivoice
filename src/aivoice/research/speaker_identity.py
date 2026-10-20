"""Speaker analysis: a verification embedding plus interpretable voice descriptors.

Speaker-*verification* embeddings (here GE2E, Wan et al. 2018, as trained in
Resemblyzer) measure "how alike do two clips sound" for evaluation. They are
not the conditioning inputs of any synthesis engine in aivoice: VoxCPM2,
X-Voice, VoiceStar and MeanVC2 each condition on their own internal
representations, so these vectors are never fed to a generator.

GE2E is loaded from the ``pretrained.pt`` that ships inside the Resemblyzer
wheel (Apache-2.0), reproducing its preprocessing (16 kHz, 25/10 ms 40-band mel
power spectrogram, 1.6 s partials at 1.3 per second, -30 dBFS volume
normalisation). Resemblyzer's own webrtcvad-based silence trimming is replaced
by aivoice's VAD trimming.

Cosine similarity between two embeddings is evidence of similarity under one
encoder, never proof that two voices are "the same".
"""

from __future__ import annotations

import importlib.util
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..audio.resample import resample

GE2E_NAME = "GE2E (Resemblyzer pretrained.pt, LSTM 3x256)"


class EncoderUnavailable(RuntimeError):
    pass


def _resemblyzer_weights() -> Path | None:
    spec = importlib.util.find_spec("resemblyzer")  # locate only: importing it pulls webrtcvad
    if spec and spec.submodule_search_locations:
        for loc in spec.submodule_search_locations:
            p = Path(loc) / "pretrained.pt"
            if p.is_file():
                return p
    return None


def trim_silence(x: np.ndarray, sr: int, *, pad_ms: int = 60) -> np.ndarray:
    """Cut leading/trailing non-speech using aivoice's VAD (Silero when available)."""
    from ..vad import make_vad

    x16 = resample(x, sr, 16000) if sr != 16000 else np.asarray(x, np.float32)
    vad = make_vad("auto")
    probs = np.asarray(vad(x16), dtype=np.float32)
    if probs.size == 0:
        return x
    frame = x16.size / probs.size  # samples per probability
    speech = np.where(probs >= 0.5)[0]
    if speech.size == 0:
        return x
    pad = int(pad_ms * 16)
    a = max(0, int(speech[0] * frame) - pad)
    b = min(x16.size, int((speech[-1] + 1) * frame) + pad)
    scale = sr / 16000
    return np.asarray(x[int(a * scale) : int(b * scale)], np.float32)


class GE2EEncoder:
    name = GE2E_NAME
    sr = 16000

    def __init__(self, device: str = "cpu", weights: Path | None = None) -> None:
        p = weights or _resemblyzer_weights()
        if p is None:
            raise EncoderUnavailable("speaker encoder weights not found: pip install resemblyzer (only its pretrained.pt is used)")
        try:
            import torch  # type: ignore
            from torch import nn  # type: ignore
        except ImportError as e:
            raise EncoderUnavailable("speaker encoder needs torch: pip install torch") from e
        if importlib.util.find_spec("librosa") is None:
            raise EncoderUnavailable("speaker encoder needs librosa for its mel frontend: pip install librosa")

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(40, 256, 3, batch_first=True)
                self.linear = nn.Linear(256, 256)
                self.relu = nn.ReLU()

            def forward(self, mels):
                _, (hidden, _) = self.lstm(mels)
                e = self.relu(self.linear(hidden[-1]))
                return e / torch.norm(e, dim=1, keepdim=True)

        # Resemblyzer's checkpoint stores {"model_state": state_dict}; tensors only, so weights_only is safe.
        ckpt = torch.load(p, map_location="cpu", weights_only=True)
        self.net = _Net()
        self.net.load_state_dict(ckpt["model_state"], strict=False)
        self.net.eval().to(device)
        self.device = device
        self.torch = torch
        self.weights_path = p

    def _mel(self, wav: np.ndarray) -> np.ndarray:
        import librosa  # type: ignore

        return librosa.feature.melspectrogram(y=wav, sr=16000, n_fft=400, hop_length=160, n_mels=40).astype(np.float32).T

    def preprocess(self, x: np.ndarray, sr: int, *, trim: bool = True) -> np.ndarray:
        w = resample(np.asarray(x, np.float32), sr, 16000) if sr != 16000 else np.asarray(x, np.float32)
        if trim:
            w = trim_silence(w, 16000)
        rms = float(np.sqrt(np.mean((w * 32767) ** 2))) if w.size else 0.0
        if rms > 0:
            change = -30 - 20 * math.log10(rms / 32767)
            if change > 0:  # increase_only, as in Resemblyzer
                w = w * (10 ** (change / 20))
        return w.astype(np.float32)

    def embed(self, x: np.ndarray, sr: int, *, trim: bool = True, return_partials: bool = False):
        w = self.preprocess(x, sr, trim=trim)
        if w.size < 1600:
            raise ValueError("clip too short for a speaker embedding (< 0.1 s of speech)")
        frame_step = int(round((16000 / 1.3) / 160))
        n_frames = int(np.ceil((w.size + 1) / 160))
        slices = []
        steps = max(1, n_frames - 160 + frame_step + 1)
        for i in range(0, steps, frame_step):
            slices.append((i, i + 160))
        last = slices[-1]
        coverage = (w.size - last[0] * 160) / (160 * 160)
        if coverage < 0.75 and len(slices) > 1:
            slices = slices[:-1]
        need = slices[-1][1] * 160
        if need >= w.size:
            w = np.pad(w, (0, need - w.size))
        mel = self._mel(w)
        batch = np.array([mel[a:b] for a, b in slices])
        with self.torch.no_grad():
            partials = self.net(self.torch.from_numpy(batch).to(self.device)).cpu().numpy()
        e = partials.mean(axis=0)
        e = e / np.linalg.norm(e)
        return (e, partials) if return_partials else e

    def similarity(self, a: np.ndarray, sr_a: int, b: np.ndarray, sr_b: int) -> float:
        return cosine(self.embed(a, sr_a), self.embed(b, sr_b))


_ENCODER: GE2EEncoder | None = None


def get_encoder(required: bool = False) -> GE2EEncoder | None:
    global _ENCODER
    if _ENCODER is None:
        try:
            _ENCODER = GE2EEncoder()
        except EncoderUnavailable:
            if required:
                raise
            return None
    return _ENCODER


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / d) if d else 0.0


# --- interpretable descriptors (non-neural) ---------------------------------------


def yin_f0(x: np.ndarray, sr: int, *, fmin: float = 60, fmax: float = 500, frame_ms: float = 40, hop_ms: float = 10,
           threshold: float = 0.15) -> np.ndarray:
    """Per-frame F0 in Hz (0 = unvoiced), YIN (de Cheveigné & Kawahara 2002) in numpy."""
    x = np.asarray(x, np.float64)
    n = int(sr * frame_ms / 1000)
    hop = int(sr * hop_ms / 1000)
    tmin, tmax = int(sr / fmax), min(int(sr / fmin), n - 1)
    out = []
    for s in range(0, max(0, x.size - n), hop):
        f = x[s : s + n]
        if np.sqrt(np.mean(f * f)) < 1e-3:
            out.append(0.0)
            continue
        d = np.array([np.sum((f[: n - tau] - f[tau:n]) ** 2) for tau in range(tmax + 1)])
        cmnd = np.ones_like(d)
        cs = np.cumsum(d[1:])
        cmnd[1:] = d[1:] * np.arange(1, d.size) / np.maximum(cs, 1e-12)
        tau = None
        for t in range(tmin, tmax):
            if cmnd[t] < threshold:
                while t + 1 < tmax and cmnd[t + 1] < cmnd[t]:
                    t += 1
                tau = t
                break
        if tau is None:
            out.append(0.0)
            continue
        if 0 < tau < d.size - 1:  # parabolic refinement
            a, b, c = cmnd[tau - 1], cmnd[tau], cmnd[tau + 1]
            denom = a - 2 * b + c
            tau = tau + 0.5 * (a - c) / denom if denom else tau
        out.append(sr / tau)
    return np.array(out)


def mfcc_stats(x: np.ndarray, sr: int, n_mfcc: int = 20) -> np.ndarray:
    """Mean+std of MFCCs: a weak, non-neural timbre summary (labelled as such wherever reported)."""
    w = resample(np.asarray(x, np.float32), sr, 16000) if sr != 16000 else np.asarray(x, np.float32)
    n_fft, hop, n_mels = 512, 160, 40
    if w.size < n_fft:
        w = np.pad(w, (0, n_fft - w.size))
    frames = np.lib.stride_tricks.sliding_window_view(w, n_fft)[::hop] * np.hanning(n_fft)
    spec = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    mel = _mel_fb(16000, n_fft, n_mels) @ spec.T
    logmel = np.log(mel + 1e-10)
    k = np.arange(n_mels)
    dct = np.cos(np.pi / n_mels * (k[None, :] + 0.5) * np.arange(n_mfcc)[:, None])
    c = dct @ logmel
    energy = logmel.mean(axis=0)
    voiced = c[:, energy > np.percentile(energy, 30)] if c.shape[1] > 4 else c
    return np.concatenate([voiced.mean(axis=1), voiced.std(axis=1)])


def _mel_fb(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    def hz2mel(f):
        return 2595 * np.log10(1 + f / 700)

    def mel2hz(m):
        return 700 * (10 ** (m / 2595) - 1)

    pts = mel2hz(np.linspace(hz2mel(0), hz2mel(sr / 2), n_mels + 2))
    bins = np.floor((n_fft + 1) * pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        a, b, c = bins[m - 1], bins[m], bins[m + 1]
        if b > a:
            fb[m - 1, a:b] = (np.arange(a, b) - a) / (b - a)
        if c > b:
            fb[m - 1, b:c] = (c - np.arange(b, c)) / (c - b)
    return fb


@dataclass
class SpeakerAnalysis:
    duration_s: float
    speech_s: float
    f0_median_hz: float | None
    f0_range_semitones: float | None  # 10th-90th percentile spread
    spectral_centroid_hz: float
    speaking_rate_syl_per_s: float | None
    embedding_model: str | None
    embedding: list[float] | None = field(default=None, repr=False)
    notes: list[str] = field(default_factory=list)

    def to_dict(self, *, include_embedding: bool = False) -> dict[str, Any]:
        d = asdict(self)
        if not include_embedding:
            d.pop("embedding", None)
        return d


def analyze_speaker(x: np.ndarray, sr: int, *, transcript: str | None = None, encoder: GE2EEncoder | None = None) -> SpeakerAnalysis:
    from .duration import speech_seconds, speaking_rate

    x = np.asarray(x, np.float32)
    w16 = resample(x, sr, 16000) if sr != 16000 else x
    f0 = yin_f0(w16, 16000)
    voiced = f0[f0 > 0]
    med = float(np.median(voiced)) if voiced.size else None
    rng = float(12 * np.log2(np.percentile(voiced, 90) / np.percentile(voiced, 10))) if voiced.size > 5 else None
    spec = np.abs(np.fft.rfft(w16 * np.hanning(w16.size))) if w16.size else np.zeros(1)
    freqs = np.fft.rfftfreq(w16.size, 1 / 16000) if w16.size else np.zeros(1)
    centroid = float(np.sum(freqs * spec) / max(np.sum(spec), 1e-12))
    rate = speaking_rate(x, sr, transcript=transcript)
    notes = ["descriptors are interpretable summaries, not identity proof"]
    emb = None
    model = None
    if encoder is not None:
        try:
            emb = encoder.embed(x, sr).tolist()
            model = encoder.name
        except ValueError as e:
            notes.append(f"no embedding: {e}")
    return SpeakerAnalysis(
        duration_s=x.size / sr,
        speech_s=speech_seconds(x, sr),
        f0_median_hz=med,
        f0_range_semitones=rng,
        spectral_centroid_hz=centroid,
        speaking_rate_syl_per_s=rate.syl_per_s if rate else None,
        embedding_model=model,
        embedding=emb,
        notes=notes,
    )
