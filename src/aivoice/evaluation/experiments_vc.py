"""Experiments I-R: live voice conversion and identity-preserving modulation (docs/research/experiments).

Measured here (CPU, rights-free espeak-ng speakers):

* L - identity retention of every modulation preset across strengths
* M - pitch-only masculinisation vs adding formants, resonance/body, then intonation + brightness
* N - formant contribution on its own (ratio sweep)
* O - long-session streaming stability (memory, latency drift, NaNs, backlog) - see run_long_session
* R - chunk smoothing: phase locking, analysis window, block-size invariance

Not run (no neural VC weights / no GPU here; protocols kept so they run unchanged elsewhere):
I, J, K, P, Q.

Proxies, stated plainly:
* identity = GE2E cosine between the modulated utterance and a *different* clean utterance of
  the same speaker (evidence of similarity, not identity proof);
* masculinity shift = projection of the GE2E embedding change onto a male-female direction
  estimated from the synthetic corpus (mean of male voices minus mean of female voices). This is
  a weak, encoder-specific proxy, learned from four formant-synthesis voices, not a perception
  measure;
* acoustic deltas (F0 in semitones, apparent vocal-tract length, spectral centroid) are measured
  directly.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from . import synthetic_data as sd
from .experiments import ExperimentResult, _ci, not_run

SR = 48000
MALE = ("synth-a", "synth-c")
FEMALE = ("synth-b", "synth-d")


class _Corpus:
    def __init__(self, n_utts: int = 5) -> None:
        from ..audio.resample import resample
        from ..research.speaker_identity import get_encoder

        self.enc = get_encoder(required=True)
        self.utts: dict[str, list[np.ndarray]] = {}
        self.anchor: dict[str, np.ndarray] = {}
        for spk in sd.SPEAKERS:
            self.utts[spk] = []
            for t in sd.HARVARD[:n_utts]:
                x, sr = sd.speak(t, spk)
                self.utts[spk].append(resample(x, sr, SR))
            a, sr = sd.speak(" ".join(sd.HARVARD[8:12]), spk)
            self.anchor[spk] = self.enc.embed(a, sr)
        m = np.mean([self.anchor[s] for s in MALE], axis=0)
        f = np.mean([self.anchor[s] for s in FEMALE], axis=0)
        d = m - f
        self.gender_dir = d / np.linalg.norm(d)

    def embed(self, x: np.ndarray) -> np.ndarray:
        return self.enc.embed(x, SR)


def _modulate(x: np.ndarray, params, *, window_ms: float = 32.0, phase_locking: bool | None = None, block: int = 480) -> np.ndarray:
    from ..modulation.dsp import SpectralModulator

    return SpectralModulator(SR, params, window_ms=window_ms, hop_ms=window_ms / 4, phase_locking=phase_locking).process_all(x, block=block)


def _measure(c: _Corpus, spk: str, x: np.ndarray, y: np.ndarray) -> dict[str, float | None]:
    from ..audio.enhancement import detect_discontinuities
    from .voice_metrics import f0_stats, formants, spectral_shape

    ex, ey = c.embed(x), c.embed(y)
    fx, fy = f0_stats(x, SR), f0_stats(y, SR)
    vx, vy = formants(x, SR), formants(y, SR)
    sx, sy = spectral_shape(x, SR), spectral_shape(y, SR)
    f0d = 12 * np.log2(fy["f0_median_hz"] / fx["f0_median_hz"]) if fx["f0_median_hz"] and fy["f0_median_hz"] else None
    vtl = (vy["apparent_vtl_cm"] - vx["apparent_vtl_cm"]) if vx.get("apparent_vtl_cm") and vy.get("apparent_vtl_cm") else None
    return {
        "identity": float(np.dot(ey, c.anchor[spk])),
        "identity_unmodified": float(np.dot(ex, c.anchor[spk])),
        "masc_shift": float(np.dot(ey - ex, c.gender_dir)),
        "f0_delta_st": f0d,
        "vtl_delta_cm": vtl,
        "centroid_delta_hz": (sy["spectral_centroid_hz"] - sx["spectral_centroid_hz"]) if sy["spectral_centroid_hz"] and sx["spectral_centroid_hz"] else None,
        "clicks": float(detect_discontinuities(y, SR).size),
        "voiced_fraction_ratio": (fy["voiced_fraction"] / fx["voiced_fraction"]) if fx["voiced_fraction"] else None,
    }


def _agg(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for k in rows[0]:
        v = [r[k] for r in rows if r.get(k) is not None]
        if v:
            out[k] = _ci(v)
    return out


# --- M: pitch-only vs multidimensional ---------------------------------------------


def exp_masculine(c: _Corpus | None = None, strength: float = 0.65) -> ExperimentResult:
    from ..modulation.dsp import ModulationParams
    from ..modulation.presets import MASCULINE_CANDIDATES, MASCULINE_DELTA

    t0 = time.perf_counter()
    c = c or _Corpus()
    s = strength
    D = {**MASCULINE_DELTA, **MASCULINE_CANDIDATES}
    arms = {
        "baseline (no change)": ModulationParams(),
        "pitch-only": ModulationParams(pitch_semitones=s * D["pitch_semitones"]),
        "pitch-only, matched (-1.5x pitch)": ModulationParams(pitch_semitones=1.5 * s * D["pitch_semitones"]),
        "pitch+formants": ModulationParams(pitch_semitones=s * D["pitch_semitones"], formant_ratio=D["formant_ratio"] ** s),
        "pitch+formants+body": ModulationParams(pitch_semitones=s * D["pitch_semitones"], formant_ratio=D["formant_ratio"] ** s,
                                                body_db=s * D["body_db"]),
        "full (+intonation, brightness)": ModulationParams(pitch_semitones=s * D["pitch_semitones"], formant_ratio=D["formant_ratio"] ** s,
                                                           body_db=s * D["body_db"], intonation=D["intonation"] ** s,
                                                           brightness_db=s * D["brightness_db"]),
    }
    per: dict[str, list[dict[str, Any]]] = {a: [] for a in arms}
    for spk in sd.SPEAKERS:
        for x in c.utts[spk]:
            for a, p in arms.items():
                per[a].append({"speaker": spk, **_measure(c, spk, x, _modulate(x, p))})
    metrics: dict[str, Any] = {"strength": s, "arms": {}, "steps": {}}
    for a, rows in per.items():
        metrics["arms"][a] = {"all": _agg([{k: v for k, v in r.items() if k != "speaker"} for r in rows]),
                              "female_voices": _agg([{k: v for k, v in r.items() if k != "speaker"} for r in rows if r["speaker"] in FEMALE])}
    order = ["pitch-only", "pitch+formants", "pitch+formants+body", "full (+intonation, brightness)"]
    # Pitch-only's marginal trade-off (1x -> 1.5x semitones) is the bar every added stage must beat:
    # masculinity gained per unit of identity lost.
    p1, p15 = per["pitch-only"], per["pitch-only, matched (-1.5x pitch)"]
    pm = float(np.mean([b["masc_shift"] - a["masc_shift"] for a, b in zip(p1, p15)]))
    pi = float(np.mean([a["identity"] - b["identity"] for a, b in zip(p1, p15)]))
    pitch_eff = pm / pi if pi > 1e-6 else float("inf")
    metrics["pitch_only_marginal_efficiency"] = round(pitch_eff, 4)
    decisions = []
    for prev, cur in zip(order, order[1:]):
        dm = [b["masc_shift"] - a["masc_shift"] for a, b in zip(per[prev], per[cur])]
        di = [b["identity"] - a["identity"] for a, b in zip(per[prev], per[cur])]
        step = {"masc_shift_gain": _ci(dm), "identity_change": _ci(di)}
        gain, cost = step["masc_shift_gain"]["mean"], -step["identity_change"]["mean"]
        eff = gain / cost if cost > 1e-6 else float("inf")
        step["efficiency"] = round(eff, 4) if np.isfinite(eff) else "no identity cost"
        keep = step["masc_shift_gain"]["ci95"][0] > 0 and (cost <= 0.005 or eff >= pitch_eff)
        step["decision"] = ("RETAIN" if keep else
                            "REJECT (masculinity gain not significant, or less efficient than lowering pitch further)")
        metrics["steps"][f"{prev} -> {cur}"] = step
        decisions.append(f"{cur}: {step['decision'].split(' ')[0]}")
    # efficiency vs matched pitch-only: same masculinity budget, who keeps more identity?
    full, matched = per["full (+intonation, brightness)"], per["pitch-only, matched (-1.5x pitch)"]
    metrics["full_vs_pitch_only_matched"] = {
        "masc_shift_diff": _ci([b["masc_shift"] - a["masc_shift"] for a, b in zip(matched, full)]),
        "identity_diff": _ci([b["identity"] - a["identity"] for a, b in zip(matched, full)]),
    }
    return ExperimentResult(
        id="M", title="Masculinisation: pitch-only vs multidimensional",
        change="Add formant lowering, then body resonance, then narrower intonation + darker brightness, on top of a pitch drop.",
        rationale="Perceived masculinity depends on F0 and on formant spacing / apparent vocal-tract length, plus voice "
                  "quality and intonation (e.g. Peterson & Barney 1952; Hillenbrand et al. 1995; Fitch & Giedd 1999; "
                  "Pisanski et al. 2016). Lowering pitch alone is the classic 'pitched-down' sound.",
        implemented="modulation/dsp.py SpectralModulator (excitation pitch shift + envelope warp); modulation/presets.py MASCULINE_DELTA",
        baseline="pitch-only shift with the same semitones (and a 1.5x larger pitch-only shift as a matched-strength check)",
        status="measured", metrics=metrics,
        decision="; ".join(decisions),
        regression="each stage runs in the same STFT: no added latency; CPU cost unchanged within noise",
        limitations=["GE2E masculinity direction learned from 4 espeak-ng formant voices: a weak proxy, not perception",
                     "formant synthesis is not natural speech; real voices respond differently, so the defaults should be "
                     "re-checked on the user's own voice (`aivoice modulate calibrate`)",
                     "no listening test was possible here"],
        data=f"4 synthetic speakers (2 'male', 2 'female' espeak variants) x {len(c.utts['synth-a'])} Harvard sentences; "
             "identity anchor = 4 different sentences of the same speaker",
        seconds=time.perf_counter() - t0,
    )


# --- N: formant contribution ---------------------------------------------------------


def exp_formants(c: _Corpus | None = None) -> ExperimentResult:
    from ..modulation.dsp import ModulationParams

    t0 = time.perf_counter()
    c = c or _Corpus()
    ratios = [1.0, 0.95, 0.90, 0.85, 0.80]
    per = {r: [] for r in ratios}
    for spk in sd.SPEAKERS:
        for x in c.utts[spk][:3]:
            for r in ratios:
                per[r].append(_measure(c, spk, x, _modulate(x, ModulationParams(formant_ratio=r))))
    metrics = {f"ratio {r}": _agg(rows) for r, rows in per.items()}
    # expected VTL change for the measured baseline: VTL scales by 1/ratio
    return ExperimentResult(
        id="N", title="Formant contribution (pitch unchanged)",
        change="Warp only the spectral envelope (formants) by 0.95-0.80.",
        rationale="Formant spacing encodes vocal-tract length, a major cue to body size and sex independent of F0.",
        implemented="SpectralModulator formant_ratio (cepstral true-envelope warp of the excitation-normalised spectrum)",
        baseline="ratio 1.0 (STFT round trip only)", status="measured", metrics=metrics,
        decision="formant lowering raises apparent VTL monotonically; identity cost grows with the shift - the masculine "
                 "preset uses 0.90 at full strength (0.934 at the default 0.65)",
        limitations=["LPC formant tracking on formant-synthesised speech; ~5-10 % tracking error is normal"],
        data="4 synthetic speakers x 3 sentences", seconds=time.perf_counter() - t0,
    )


# --- L: identity retention per preset ------------------------------------------------


def exp_identity(c: _Corpus | None = None) -> ExperimentResult:
    from ..modulation.presets import BUILTIN

    t0 = time.perf_counter()
    c = c or _Corpus()
    names = ["natural", "masculine-subtle", "masculine-natural", "masculine-strong", "feminine-subtle", "feminine-natural",
             "deep", "warm", "bright", "dark", "breathy", "rough", "broadcast", "anonymous"]
    metrics: dict[str, Any] = {}
    for n in names:
        rows = []
        for spk in sd.SPEAKERS:
            for x in c.utts[spk][:2]:
                y = _modulate(x, BUILTIN[n].params())
                m = _measure(c, spk, x, y)
                rows.append({"identity": m["identity"], "identity_drop": m["identity_unmodified"] - m["identity"],
                             "f0_delta_st": m["f0_delta_st"], "vtl_delta_cm": m["vtl_delta_cm"], "clicks": m["clicks"]})
        metrics[n] = _agg(rows)
    flagged = [n for n in names if metrics[n]["identity_drop"]["mean"] > 0.10]
    return ExperimentResult(
        id="L", title="Modulation identity retention",
        change="Each built-in preset at its default strength.",
        rationale="Modulation must change attributes while the voice stays recognisably the same speaker.",
        implemented="modulation/presets.py; identity guard in `aivoice modulate --file --identity`",
        baseline="the unmodified utterance's similarity to the same anchor", status="measured", metrics=metrics,
        decision=("presets with mean GE2E drop > 0.10 (drift risk): " + (", ".join(flagged) if flagged else "none")
                  + ". `--identity FLOOR` backs strength off automatically in file mode."),
        limitations=["GE2E on synthetic speech; thresholds are not perceptual"],
        data="4 synthetic speakers x 2 sentences", seconds=time.perf_counter() - t0,
    )


# --- R: chunk smoothing ---------------------------------------------------------------


def exp_smoothing(c: _Corpus | None = None) -> ExperimentResult:
    from ..modulation.dsp import ModulationParams, _yin_frame

    t0 = time.perf_counter()
    c = c or _Corpus()

    def aper(x):
        n, h = int(0.04 * SR), int(0.01 * SR)
        return np.array([_yin_frame(x[s : s + n].astype(float), SR, 60, 600)[1] for s in range(0, x.size - n, h)])

    p = ModulationParams(pitch_semitones=-4.0, formant_ratio=0.93)
    rows: dict[str, list[float]] = {}
    inv: list[float] = []
    for spk in sd.SPEAKERS:
        for x in c.utts[spk][:2]:
            base = aper(x)
            voiced = base < 0.25
            for win in (16, 32, 48):
                for lock in (True, False):
                    y = _modulate(x, p, window_ms=win, phase_locking=lock)
                    a = aper(y)
                    k = f"window {win} ms, phase locking {'on' if lock else 'off'}"
                    rows.setdefault(k + ": aperiodicity on voiced frames", []).append(float(a[voiced[: a.size]].mean()))
            y1 = _modulate(x, p, block=480)
            y2 = _modulate(x, p, block=4801)
            inv.append(float(np.max(np.abs(y1 - y2))))
            rows.setdefault("original: aperiodicity on voiced frames", []).append(float(base[voiced].mean()))
    metrics: dict[str, Any] = {k: _ci(v) for k, v in rows.items()}
    metrics["block_size_invariance_max_abs_diff"] = float(max(inv))
    on32 = metrics["window 32 ms, phase locking on: aperiodicity on voiced frames"]["mean"]
    off32 = metrics["window 32 ms, phase locking off: aperiodicity on voiced frames"]["mean"]
    return ExperimentResult(
        id="R", title="Chunk smoothing and phase continuity (DSP path)",
        change="Identity phase locking on/off; 16/32/48 ms analysis windows; capture block size.",
        rationale="Phase-vocoder phasiness and frame-boundary discontinuities are the audible failure of streaming "
                  "pitch shifting (Laroche & Dolson 1999).",
        implemented="SpectralModulator(phase_locking, window_ms); output independent of capture block size by design",
        baseline="phase locking off", status="measured", metrics=metrics,
        decision=(f"phase locking {'RETAIN' if on32 <= off32 else 'REJECT'} at 32 ms ({on32:.3f} vs {off32:.3f}); "
                  "default window 32 ms (32 ms delay); 16 ms offered as the low-latency preset (16 ms delay), where phase "
                  "locking is switched off automatically because it measured worse there; block size does not change "
                  "the output (max diff above). Cross-frame peak matching was also tried and measured worse; not kept."),
        limitations=["YIN aperiodicity is an objective periodicity proxy, not a listening test",
                     "X-VC / RVC / MeanVC2 crossfades could not be compared here (no weights)"],
        data="4 synthetic speakers x 2 sentences, -4 st / 0.93 formants", seconds=time.perf_counter() - t0,
    )


VC_BLOCKED = [
    ("I", "X-VC vs MeanVC2 quality", "Same sources/targets through both (aivoice benchmark vc --engines xvc,meanvc2): target "
     "GE2E similarity, source leakage, WER/CER, F0 jumps, clicks; loudness-matched listening set.",
     "X-VC and MeanVC2 weights are on Hugging Face, which is unreachable here; no GPU"),
    ("J", "X-VC vs MeanVC2 latency", "Per-chunk p50/p95/p99, RTF, algorithmic vs measured delay at each quality preset on the RTX 4090.",
     "no weights / GPU here"),
    ("K", "Source-timbre leakage", "GE2E(output, source) vs GE2E(source, target) baseline per engine; lower is better.",
     "needs a neural VC engine running"),
    ("P", "RVC vs modern zero-shot VC", "Per-voice RVC .pth vs X-VC / MeanVC2 on the same voice's reference.",
     "needs an RVC venv with rvc-python and the base models, and the zero-shot weights"),
    ("Q", "Neural vs DSP modulation", "StyleStream (curated styles) vs DSP presets on identity retention and attribute change.",
     "StyleStream checkpoints are on Hugging Face (unreachable here); its public release cannot embed arbitrary targets"),
]

VC_EXPERIMENTS = {"M": exp_masculine, "N": exp_formants, "L": exp_identity, "R": exp_smoothing}


def run(ids: list[str] | None, out_dir) -> list[ExperimentResult]:
    import json
    from dataclasses import asdict

    out_dir.mkdir(parents=True, exist_ok=True)
    want = ids or list(VC_EXPERIMENTS) + [b[0] for b in VC_BLOCKED]
    corpus = _Corpus() if any(i in VC_EXPERIMENTS for i in want) else None
    results = []
    for i in want:
        if i in VC_EXPERIMENTS:
            r = VC_EXPERIMENTS[i](corpus)
        elif i == "O":
            continue  # long session is run separately (minutes of audio): `aivoice research long-session`
        else:
            r = not_run(*next(b for b in VC_BLOCKED if b[0] == i))
        (out_dir / f"{r.id}.json").write_text(json.dumps(asdict(r), indent=2, default=str) + "\n")
        (out_dir / f"{r.id}.md").write_text(r.to_markdown())
        results.append(r)
    return results


__all__ = ["VC_BLOCKED", "VC_EXPERIMENTS", "exp_formants", "exp_identity", "exp_masculine", "exp_smoothing", "run"]
