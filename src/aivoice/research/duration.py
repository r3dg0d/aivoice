"""Silence-aware speaking-rate estimation and duration planning.

Idea from Cross-Lingual F5-TTS 2 (arXiv:2609.15184): a speaking-rate predictor
should not be fooled by leading/trailing silence in the prompt. That paper
achieves it with silence-aware *training augmentation* of a learned syllable-rate
predictor; aivoice does not reproduce that model. It adapts the idea at
inference time instead: the reference's rate is measured over **speech only**
(VAD), so silence padding around a reference no longer stretches the plan.

Two estimators of syllables in the reference:

* transcript: syllables of the transcript (IPA vowel nuclei via espeak-ng when
  available, else orthographic vowel groups);
* acoustic (transcript-free): syllable nuclei counted as prominent peaks of the
  band-limited intensity envelope inside speech regions (in the spirit of de Jong
  & Wempe 2009).

The plan feeds engines with native duration control (VoiceStar target_duration,
X-Voice fix_duration). It is evaluated in docs/research/experiments.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..audio.resample import resample

IPA_VOWELS = set("aeiouyæøœɐɑɒɔəɘɛɜɞɤɨɪʉʊʌʏɯɵɶ")
LATIN_VOWELS = set("aeiouyàáâãäåæèéêëìíîïòóôõöøùúûüýÿœ")


def speech_mask(x16: np.ndarray) -> tuple[np.ndarray, float]:
    """Per-frame speech probabilities >= 0.5 and the frame length in samples (16 kHz input)."""
    from ..vad import make_vad

    vad = make_vad("auto")
    probs = np.asarray(vad(x16), dtype=np.float32)
    if probs.size == 0:
        return np.zeros(0, bool), 512.0
    return probs >= 0.5, x16.size / probs.size


def speech_seconds(x: np.ndarray, sr: int) -> float:
    x16 = resample(np.asarray(x, np.float32), sr, 16000) if sr != 16000 else np.asarray(x, np.float32)
    mask, frame = speech_mask(x16)
    if mask.size == 0:
        return 0.0
    # Close gaps shorter than 150 ms (stop closures are part of speech, not pauses)
    gap = max(1, int(0.15 * 16000 / frame))
    m = mask.copy()
    idx = np.where(m)[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= gap:
            m[a:b] = True
    return float(m.sum() * frame / 16000)


def silence_edges(x: np.ndarray, sr: int) -> tuple[float, float]:
    """(leading, trailing) non-speech seconds."""
    x16 = resample(np.asarray(x, np.float32), sr, 16000) if sr != 16000 else np.asarray(x, np.float32)
    mask, frame = speech_mask(x16)
    idx = np.where(mask)[0]
    total = x16.size / 16000
    if idx.size == 0:
        return total, 0.0
    lead = idx[0] * frame / 16000
    trail = total - (idx[-1] + 1) * frame / 16000
    return float(lead), float(max(0.0, trail))


def _espeak_ipa(text: str, lang: str) -> str | None:
    exe = shutil.which("espeak-ng")
    if not exe:
        return None
    try:
        r = subprocess.run([exe, "-q", "--ipa", "-v", lang, text], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout if r.returncode == 0 and r.stdout.strip() else None


def count_syllables(text: str, language: str = "en") -> int:
    """Syllable count: IPA vowel nuclei (espeak-ng) when possible, else orthographic vowel groups."""
    ipa = _espeak_ipa(text, language)
    if ipa:
        n = 0
        prev = False
        for ch in unicodedata.normalize("NFD", ipa):
            v = ch in IPA_VOWELS
            if v and not prev:
                n += 1
            prev = v if ch not in "ːˑ" else prev
        if n:
            return n
    words = re.findall(r"[^\W\d_]+", text.lower())
    n = 0
    for w in words:
        groups = re.findall(r"[aeiouyàáâãäåæèéêëìíîïòóôõöøùúûüýÿœ]+", w)
        k = len(groups)
        if w.endswith("e") and k > 1 and not w.endswith(("le", "ee")):
            k -= 1
        n += max(1, k)
    # Non-Latin scripts: count characters of CJK as syllables
    n += len(re.findall(r"[぀-ヿ一-鿿가-힯]", text))
    return n


def acoustic_syllables(x: np.ndarray, sr: int) -> int:
    """Transcript-free syllable-nucleus count from intensity peaks within speech."""
    x16 = resample(np.asarray(x, np.float32), sr, 16000) if sr != 16000 else np.asarray(x, np.float32)
    if x16.size < 1600:
        return 0
    # Emphasise the vowel band (300-3000 Hz) with a simple FFT filter
    spec = np.fft.rfft(x16)
    f = np.fft.rfftfreq(x16.size, 1 / 16000)
    spec[(f < 300) | (f > 3000)] = 0
    band = np.fft.irfft(spec, n=x16.size)
    hop = 160
    env = np.sqrt(np.convolve(band**2, np.ones(400) / 400, mode="same")[::hop])
    env_db = 20 * np.log10(env + 1e-8)
    env_db = np.convolve(env_db, np.ones(5) / 5, mode="same")
    mask, frame = speech_mask(x16)
    if mask.size:
        sp = np.repeat(mask, int(np.ceil(frame / hop)))[: env_db.size]
        sp = np.pad(sp, (0, max(0, env_db.size - sp.size)))
    else:
        sp = np.ones(env_db.size, bool)
    thr = np.percentile(env_db[sp], 30) if sp.any() else env_db.max() - 20
    peaks = 0
    last = -10
    for i in range(1, env_db.size - 1):
        if not sp[i] or env_db[i] < thr:
            continue
        if env_db[i] >= env_db[i - 1] and env_db[i] > env_db[i + 1]:
            # require a dip of >= 2 dB since the previous peak (separates nuclei)
            lo = env_db[max(0, last, i - 25) : i].min() if i - last > 1 and i > 0 else env_db[i]
            if i - last >= 8 and env_db[i] - lo >= 2.0:
                peaks += 1
                last = i
    return peaks


@dataclass
class RateEstimate:
    syl_per_s: float
    syllables: int
    speech_s: float
    method: str  # transcript | acoustic


def speaking_rate(x: np.ndarray, sr: int, *, transcript: str | None = None, language: str = "en") -> RateEstimate | None:
    sp = speech_seconds(x, sr)
    if sp < 0.3:
        return None
    if transcript and transcript.strip():
        syl = count_syllables(transcript, language)
        method = "transcript"
    else:
        syl = acoustic_syllables(x, sr)
        method = "acoustic"
    if syl == 0:
        return None
    return RateEstimate(syl / sp, syl, sp, method)


@dataclass
class DurationPlan:
    seconds: float
    method: str
    syllables: int
    rate_syl_per_s: float
    pause_s: float


DEFAULT_RATE = 4.0  # syllables / s, conversational English (used only when nothing is measurable)


def punctuation_pause_s(text: str) -> float:
    return 0.35 * len(re.findall(r"[.!?;]", text.strip()[:-1])) + 0.15 * len(re.findall(r",", text))


def plan_duration(
    text: str,
    reference: str | Path | np.ndarray | None,
    *,
    sr: int | None = None,
    transcript: str | None = None,
    language: str = "en",
    speed: float = 1.0,
) -> DurationPlan:
    """Target duration for *text* spoken at the reference's speech-only rate."""
    rate = None
    if reference is not None:
        if isinstance(reference, (str, Path)):
            from ..audio_io import read_wav

            x, sr = read_wav(Path(reference))
        else:
            x = np.asarray(reference, np.float32)
        rate = speaking_rate(x, int(sr or 16000), transcript=transcript, language=language)
    syl = max(1, count_syllables(text, language))
    r = rate.syl_per_s if rate else DEFAULT_RATE
    pause = punctuation_pause_s(text)
    secs = syl / (r * max(0.25, speed)) + pause
    method = f"silence-aware/{rate.method}" if rate else "default-rate"
    return DurationPlan(round(secs, 3), method, syl, r, pause)


def naive_plan(text: str, ref_duration_s: float, ref_transcript: str, language: str = "en") -> float:
    """Baseline: rate over the whole reference, silences included (what silence-awareness fixes)."""
    return count_syllables(text, language) / max(1e-6, count_syllables(ref_transcript, language) / ref_duration_s)


def voicestar_default(text: str, ref_duration_s: float) -> float:
    """voicestar 0.1.0 utils.estimate_duration, reproduced: len(text) * (ref_dur / len(text)) == ref_dur."""
    n = max(len(text), 1)
    return n * (ref_duration_s / n)
