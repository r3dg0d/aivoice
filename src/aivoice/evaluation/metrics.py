"""Objective measurements for synthesized speech.

* speaker similarity: cosine of GE2E embeddings (one encoder; a second, weak
  non-neural MFCC-statistics similarity is reported alongside and labelled);
* intelligibility: WER/CER from an installed ASR backend (Moonshine / Parakeet);
  ASR error is a proxy, not a perfect intelligibility measure;
* acoustic: duration, loudness, peak, clipping, silence ratio, discontinuities,
  effective bandwidth (exposes resampled-not-synthesized high rates);
* naturalness: no validated predictor (e.g. NISQA) is bundled, so it is reported
  as not measured. Nothing here is a MOS.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import numpy as np

from ..audio.enhancement import detect_discontinuities
from ..audio.loudness import integrated_loudness, peak_dbfs
from ..research.reference_quality import effective_bandwidth


def acoustic_metrics(x: np.ndarray, sr: int) -> dict[str, Any]:
    from ..research.duration import speech_seconds

    x = np.asarray(x, np.float32)
    dur = x.size / sr if sr else 0.0
    sp = speech_seconds(x, sr) if x.size else 0.0
    return {
        "duration_s": round(dur, 3),
        "speech_s": round(sp, 3),
        "silence_ratio": round(1 - sp / dur, 3) if dur else None,
        "lufs": round(integrated_loudness(x, sr), 2) if x.size else None,
        "peak_dbfs": round(peak_dbfs(x), 2) if x.size else None,
        "clipped_fraction": float(np.mean(np.abs(x) >= 0.999)) if x.size else 0.0,
        "discontinuities": int(detect_discontinuities(x, sr).size),
        "effective_bandwidth_hz": round(effective_bandwidth(x, sr), 1) if x.size else None,
        "sample_rate": sr,
    }


def speaker_similarity(a: np.ndarray, sr_a: int, b: np.ndarray, sr_b: int, *, encoder: Any | None = None) -> dict[str, Any]:
    from ..research.speaker_identity import cosine, get_encoder, mfcc_stats

    enc = encoder if encoder is not None else get_encoder()
    out: dict[str, Any] = {}
    if enc is not None:
        try:
            out["ge2e_cosine"] = round(enc.similarity(a, sr_a, b, sr_b), 4)
            out["ge2e_model"] = enc.name
        except ValueError as e:
            out["ge2e_error"] = str(e)
    else:
        out["ge2e_cosine"] = None
        out["ge2e_model"] = "unavailable (pip install torch librosa resemblyzer)"
    ma, mb = mfcc_stats(a, sr_a), mfcc_stats(b, sr_b)
    out["mfcc_stats_cosine"] = round(cosine(ma - ma.mean(), mb - mb.mean()), 4)
    out["note"] = "cosine similarity under one encoder; evidence, not proof of identity"
    return out


def word_error_rate(ref: str, hyp: str) -> float | None:
    from ..asr.bench import word_errors

    e, n = word_errors(ref, hyp)
    return e / n if n else None


def char_error_rate(ref: str, hyp: str) -> float | None:
    import re

    r = re.sub(r"[^\w]", "", ref.lower())
    h = re.sub(r"[^\w]", "", hyp.lower())
    if not r:
        return None
    d = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hc in enumerate(h, 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (rc != hc))
            prev = cur
    return d[len(h)] / len(r)


def intelligibility(x: np.ndarray, sr: int, reference_text: str, *, asr: Any | None = None) -> dict[str, Any]:
    if asr is None:
        return {"wer": None, "cer": None, "asr": "unavailable (install moonshine-voice or NeMo)"}
    hyp = asr.transcribe(x, sr).text
    return {"wer": word_error_rate(reference_text, hyp), "cer": char_error_rate(reference_text, hyp),
            "asr": asr.get_capabilities().name, "hypothesis": hyp,
            "note": "ASR error is a proxy for intelligibility, not a perfect measure"}


def available_asr() -> Any | None:
    """A loaded ASR backend for WER if one is installed and registered locally, else None."""
    from ..asr import available_backends, create_backend

    for name in ("moonshine-small", "parakeet"):
        if available_backends().get(name) is None:
            try:
                be = create_backend(name)
                be.load_model()
                return be
            except Exception:  # noqa: BLE001
                continue
    return None


def write_labeled_wav(path: str | Path, x: np.ndarray, sr: int, *, comment: str, software: str = "aivoice") -> Path:
    """16-bit PCM WAV with a RIFF LIST/INFO chunk marking the audio as synthetic (ICMT, ISFT)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(np.asarray(x, np.float32), -1, 1) * 32767).astype("<i2").tobytes()

    def sub(cid: bytes, text: str) -> bytes:
        b = text.encode("utf-8") + b"\x00"
        if len(b) % 2:
            b += b"\x00"
        return cid + struct.pack("<I", len(b)) + b

    info = b"INFO" + sub(b"ICMT", comment) + sub(b"ISFT", software)
    fmt = b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16)
    lst = b"LIST" + struct.pack("<I", len(info)) + info
    data = b"data" + struct.pack("<I", len(pcm)) + pcm
    body = b"WAVE" + fmt + lst + data
    path.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)
    return path


def read_wav_comment(path: str | Path) -> str | None:
    raw = Path(path).read_bytes()
    i = raw.find(b"ICMT")
    if i < 0:
        return None
    n = struct.unpack("<I", raw[i + 4 : i + 8])[0]
    return raw[i + 8 : i + 8 + n].rstrip(b"\x00").decode("utf-8", "replace")
