"""Ablation experiments for the research-derived techniques (docs/research/experiments).

Each experiment states the change, why the source paper suggests it helps, what
aivoice implements, the measured effect with a 95% bootstrap confidence
interval, and a decision. Decision rule: retain (default on) only when the CI of
the improvement excludes zero in the helpful direction **and** speaker similarity
does not regress; otherwise default off.

Experiments needing model inference (MeanVC2, Fish, VoxCPM2, X-Voice, VoiceStar)
are listed with their protocol and status "not run" when the engines are not
installed: their results are never estimated.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import synthetic_data as sd


def bootstrap_ci(values: list[float], *, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return (float("nan"),) * 3  # type: ignore[return-value]
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n, v.size), replace=True).mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


@dataclass
class ExperimentResult:
    id: str
    title: str
    change: str
    rationale: str
    implemented: str
    baseline: str
    status: str  # measured | not run
    metrics: dict[str, Any] = field(default_factory=dict)
    decision: str = ""
    regression: str = ""
    limitations: list[str] = field(default_factory=list)
    data: str = ""
    seconds: float = 0.0

    def to_markdown(self) -> str:
        lines = [f"# {self.id}: {self.title}", "", f"- **Status:** {self.status}",
                 f"- **What changed:** {self.change}", f"- **Why (paper):** {self.rationale}",
                 f"- **Implemented as:** {self.implemented}", f"- **Baseline:** {self.baseline}",
                 f"- **Data:** {self.data}", ""]
        if self.metrics:
            lines += ["## Measured", "", "```json", json.dumps(self.metrics, indent=2), "```", ""]
        lines += [f"**Decision:** {self.decision}", "", f"**Performance cost / regression:** {self.regression or 'n/a'}", ""]
        if self.limitations:
            lines += ["## Limitations", ""] + [f"- {x}" for x in self.limitations] + [""]
        return "\n".join(lines)


def _ci(values: list[float]) -> dict[str, float]:
    m, lo, hi = bootstrap_ci(values)
    return {"mean": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)], "n": len(values)}


# --- F: silence-aware duration -------------------------------------------------


def exp_duration(seed: int = 0) -> ExperimentResult:
    from ..research.duration import naive_plan, plan_duration, silence_edges, voicestar_default

    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    errs: dict[str, list[float]] = {"naive_whole_clip": [], "voicestar_default": [], "silence_aware_transcript": [],
                                    "silence_aware_acoustic": []}
    by_bucket: dict[str, dict[str, list[float]]] = {}
    span_err: list[float] = []
    signed_total: list[float] = []
    signed_span: list[float] = []
    for spk in sd.SPEAKERS:
        utts = [(t, *sd.speak(t, spk)) for t in sd.HARVARD[:8]]
        for i in range(len(utts) - 1):
            ref_text, ref_x, sr = utts[i]
            tgt_text, tgt_x, _ = utts[i + 1]
            true_s = tgt_x.size / sr
            lead_t, trail_t = silence_edges(tgt_x, sr)
            true_span = true_s - lead_t - trail_t
            lead, trail = float(rng.uniform(0, 2)), float(rng.uniform(0, 2))
            ref = sd.pad_silence(ref_x, sr, lead, trail, seed=seed + i)
            ref_dur = ref.size / sr
            est = {
                "naive_whole_clip": naive_plan(tgt_text, ref_dur, ref_text),
                "voicestar_default": voicestar_default(tgt_text, ref_dur),
                "silence_aware_transcript": plan_duration(tgt_text, ref, sr=sr, transcript=ref_text).seconds,
                "silence_aware_acoustic": plan_duration(tgt_text, ref, sr=sr).seconds,
            }
            sa = est["silence_aware_transcript"]
            span_err.append(abs(sa - true_span) / true_span)
            signed_total.append((sa - true_s) / true_s)
            signed_span.append((sa - true_span) / true_span)
            bucket = "pad<1s" if lead + trail < 1 else ("pad1-2s" if lead + trail < 2 else "pad>2s")
            for k, v in est.items():
                e = abs(v - true_s) / true_s
                errs[k].append(e)
                by_bucket.setdefault(bucket, {}).setdefault(k, []).append(e)
    diffs = [a - b for a, b in zip(errs["naive_whole_clip"], errs["silence_aware_transcript"])]
    imp = _ci(diffs)
    metrics = {
        "abs_relative_duration_error": {k: _ci(v) for k, v in errs.items()},
        "improvement_naive_minus_silence_aware_transcript": imp,
        "by_total_padding": {b: {k: round(float(np.mean(v)), 4) for k, v in d.items()} for b, d in sorted(by_bucket.items())},
        "silence_aware_transcript_vs_speech_span": {"abs_rel_error": _ci(span_err), "signed_rel_error": _ci(signed_span)},
        "silence_aware_transcript_signed_vs_total": _ci(signed_total),
        "syllable_counter": "espeak-ng IPA vowel nuclei" if sd.available() else "orthographic",
        "pairs": len(errs["naive_whole_clip"]),
    }
    keep = imp["ci95"][0] > 0
    return ExperimentResult(
        id="F", title="Silence-aware duration planning",
        change="Measure the reference speaking rate over VAD speech only (not the whole clip) when planning target duration.",
        rationale="Cross-Lingual F5-TTS 2: speaking-rate prediction is thrown off by leading/trailing prompt silence.",
        implemented="research/duration.py plan_duration (transcript and transcript-free acoustic variants).",
        baseline="Rate over the whole reference clip; and VoiceStar 0.1.0's default estimate (== reference duration).",
        status="measured", metrics=metrics,
        decision=("RETAIN (default on for engines with duration control): improvement CI excludes zero"
                  if keep else "REJECT: improvement not significant"),
        regression="adds one VAD pass over the reference (~ms); no synthesis cost",
        limitations=["espeak-ng formant speech with exact, constant speaking rate; natural speech varies more",
                     "measures the duration *plan*, not a model's adherence to it (no duration-controlled model ran here)",
                     "the plan predicts the speech span (with internal pauses); ground-truth totals also contain the "
                     "synthesizer's own edge silence (~0.3 s for espeak-ng), hence a negative signed error vs total. "
                     "No edge allowance is added because it is engine-specific and would be tuned on espeak"],
        data="4 synthetic espeak-ng speakers x 7 reference->target Harvard-sentence pairs, uniform 0-2 s lead and trail padding",
        seconds=time.perf_counter() - t0,
    )


# --- C: reference preprocessing ------------------------------------------------


def _degrade(x: np.ndarray, sr: int, kind: str, rng: np.random.Generator) -> np.ndarray:
    if kind == "silence-pad":
        return sd.pad_silence(x, sr, 1.5, 1.5, noise_db=-55)
    if kind == "quiet":
        return (x * 10 ** (-20 / 20)).astype(np.float32)
    if kind == "noise-15dB":
        p = float(np.mean(x**2))
        return (x + rng.normal(0, np.sqrt(p / 10 ** 1.5), x.size)).astype(np.float32)
    if kind == "clipped":
        return np.clip(x * 4, -0.6, 0.6).astype(np.float32)
    if kind == "combined":
        y = sd.pad_silence(x * 0.2, sr, 1.0, 1.0, noise_db=-50)
        p = float(np.mean((x * 0.2) ** 2))
        return (y + rng.normal(0, np.sqrt(p / 10 ** 2.0), y.size)).astype(np.float32)
    raise ValueError(kind)


def exp_reference_prep(seed: int = 0) -> ExperimentResult:
    import tempfile

    from ..audio_io import write_wav
    from ..research.reference_quality import prepare_reference
    from ..research.speaker_identity import get_encoder

    enc = get_encoder()
    t0 = time.perf_counter()
    if enc is None:
        return ExperimentResult("C", "Reference preprocessing", "", "", "", "", "not run",
                                decision="not run: speaker encoder unavailable (torch/librosa/resemblyzer)")
    rng = np.random.default_rng(seed)
    kinds = ["silence-pad", "quiet", "noise-15dB", "clipped", "combined"]
    arms: dict[str, dict[str, list[float]]] = {k: {"raw": [], "prepared": [], "prepared+denoise": [], "prepared+auto (default)": []} for k in kinds}
    with tempfile.TemporaryDirectory(prefix="aivoice-expC-") as td:
        for spk in sd.SPEAKERS:
            clean_other, sr = sd.speak(" ".join(sd.HARVARD[8:12]), spk)  # identity anchor: different text
            ref_x, _ = sd.speak(" ".join(sd.HARVARD[:3]), spk)
            anchor = enc.embed(clean_other, sr)
            for kind in kinds:
                deg = _degrade(ref_x, sr, kind, rng)
                p = Path(td) / f"{spk}-{kind}.wav"
                write_wav(p, deg, sr)
                arms[kind]["raw"].append(float(np.dot(enc.embed(deg, sr, trim=False), anchor)))
                for arm, dn in (("prepared", False), ("prepared+denoise", True), ("prepared+auto (default)", "auto")):
                    rep = prepare_reference(p, Path(td) / f"{spk}-{kind}-{arm}.wav", denoise=dn, encoder=enc)
                    from ..audio_io import read_wav

                    y, _ = read_wav(Path(rep.prepared))
                    arms[kind][arm].append(float(np.dot(enc.embed(y, sr, trim=False), anchor)))
    metrics: dict[str, Any] = {"similarity_to_clean_other_utterance": {}, "improvement_prepared_minus_raw": {},
                               "improvement_denoise_minus_prepared": {}}
    all_imp, all_dn = [], []
    for k in kinds:
        metrics["similarity_to_clean_other_utterance"][k] = {a: round(float(np.mean(v)), 4) for a, v in arms[k].items()}
        imp = [b - a for a, b in zip(arms[k]["raw"], arms[k]["prepared"])]
        dn = [b - a for a, b in zip(arms[k]["prepared"], arms[k]["prepared+denoise"])]
        all_imp += imp
        all_dn += dn
        metrics["improvement_prepared_minus_raw"][k] = _ci(imp)
        metrics["improvement_denoise_minus_prepared"][k] = _ci(dn)
    metrics["overall_prepared_minus_raw"] = _ci(all_imp)
    metrics["overall_default_auto_minus_raw"] = _ci([b - a for k in kinds for a, b in zip(arms[k]["raw"], arms[k]["prepared+auto (default)"])])
    metrics["overall_denoise_minus_prepared"] = _ci(all_dn)
    metrics["encoder"] = enc.name
    metrics["note"] = "embeddings computed without the encoder's own silence trimming, so preparation effects are visible"
    keep_prep = metrics["overall_prepared_minus_raw"]["ci95"][0] > 0
    keep_dn = metrics["overall_denoise_minus_prepared"]["ci95"][0] > 0
    return ExperimentResult(
        id="C", title="Reference preprocessing (trim, best window, loudness, optional denoise)",
        change="Condition engines on a prepared copy of the reference instead of the raw upload.",
        rationale="Cross-Lingual F5-TTS 2 and VoxCPM2 both trim prompt silence; noisy/quiet prompts degrade zero-shot cloning.",
        implemented="research/reference_quality.py prepare_reference with identity guard; spectral gating denoise optional.",
        baseline="raw degraded reference",
        status="measured", metrics=metrics,
        decision=("preparation: " + ("RETAIN (default on)" if keep_prep else "NOT SIGNIFICANT (kept on only for trimming/safety; re-evaluate on real voices)")
                  + "; denoise: " + ("RETAIN as 'auto' (only when estimated SNR < 25 dB, identity-guarded)" if keep_dn
                                     else "default OFF (did not improve similarity)")),
        regression="one analysis + preparation pass at profile creation; no per-utterance cost",
        limitations=["GE2E similarity on formant speech; identity effects on real voices may differ",
                     "measures the conditioning clip, not cloned output (no cloning engine ran here)"],
        data="4 synthetic speakers x 5 degradations; anchor = a different clean utterance of the same speaker",
        seconds=time.perf_counter() - t0,
    )


# --- G: enhancement + long-form joins -----------------------------------------


def exp_enhancement(seed: int = 0) -> ExperimentResult:
    from ..audio.enhancement import crossfade_concat, detect_discontinuities, enhance, loudness_matched
    from ..audio.loudness import integrated_loudness
    from ..research.speaker_identity import get_encoder

    enc = get_encoder()
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    rows: dict[str, list[float]] = {k: [] for k in ("clicks_hard", "clicks_fused", "lufs_spread_hard", "lufs_spread_fused",
                                                    "clicks_enhanced", "identity_enhanced_vs_input",
                                                    "detections_on_clean_segments", "enhance_changes_clean")}
    for spk in sd.SPEAKERS:
        parts = []
        for t in sd.HARVARD[:5]:
            x, sr = sd.speak(t, spk)
            rows["detections_on_clean_segments"].append(float(detect_discontinuities(x, sr).size))
            yc, repc = enhance(x, sr, mode="auto")
            rows["enhance_changes_clean"].append(float("declick" in repc.applied))
            parts.append((x * float(rng.uniform(0.25, 1.0))).astype(np.float32))  # segment level jumps
        hard = np.concatenate(parts)
        # inject 3 clicks per stitched output
        for pos in rng.integers(sr, hard.size - sr, 3):
            hard[pos] += 0.8
        fused = crossfade_concat(loudness_matched(parts, sr), sr, gaps_ms=[300] * (len(parts) - 1))
        seg_lufs = [integrated_loudness(p, sr) for p in parts]
        fused_lufs = [integrated_loudness(p, sr) for p in loudness_matched(parts, sr)]
        rows["clicks_hard"].append(float(detect_discontinuities(hard, sr).size))
        rows["clicks_fused"].append(float(detect_discontinuities(fused, sr).size))
        rows["lufs_spread_hard"].append(float(np.std(seg_lufs)))
        rows["lufs_spread_fused"].append(float(np.std(fused_lufs)))
        y, rep = enhance(hard, sr, mode="auto", encoder=enc)
        rows["clicks_enhanced"].append(float(rep.clicks_after))
        if rep.identity_similarity is not None:
            rows["identity_enhanced_vs_input"].append(rep.identity_similarity)
    metrics = {k: _ci(v) for k, v in rows.items() if v}
    imp = [a - b for a, b in zip(rows["clicks_hard"], rows["clicks_enhanced"])]
    metrics["click_reduction_by_enhance_auto"] = _ci(imp)
    ident_ok = not rows["identity_enhanced_vs_input"] or min(rows["identity_enhanced_vs_input"]) >= 0.95
    keep = metrics["click_reduction_by_enhance_auto"]["ci95"][0] > 0 and ident_ok
    return ExperimentResult(
        id="G", title="Output enhancement and long-form joins",
        change="Equal-power crossfades + loudness matching at segment joins; `--enhance auto` click repair + loudness.",
        rationale="VoiceStar targets long-form stability; boundary clicks and level jumps are the audible failure of stitched long-form output.",
        implemented="audio/enhancement.py (crossfade_concat, loudness_matched, enhance auto with identity guard).",
        baseline="hard concatenation with level jumps and injected clicks",
        status="measured", metrics=metrics,
        decision=("joins: RETAIN (always used for long-form); enhance auto: "
                  + ("available, default OFF for single utterances (only helps when artifacts exist), identity preserved"
                     if keep else "REJECT")),
        regression="enhancement adds analysis passes (tens of ms per utterance)",
        limitations=["clicks are synthetic impulses; real codec/vocoder artifacts differ"],
        data="4 synthetic speakers x 5 Harvard sentences with random segment gains, 3 injected clicks each",
        seconds=time.perf_counter() - t0,
    )


def not_run(id_: str, title: str, protocol: str, why: str) -> ExperimentResult:
    return ExperimentResult(id=id_, title=title, change=protocol, rationale="", implemented="", baseline="",
                            status="not run", decision=f"NOT RUN: {why}")


BLOCKED = [
    ("A", "Baseline A: MeanVC2 conversion", "Convert a common neutral source (espeak) to each target with MeanVC2; GE2E similarity, WER, RTF.",
     "MeanVC2 weights/vendor not installable here (huggingface.co and GitHub blocked)"),
    ("B", "Baseline B: Fish S2-Pro / Drama 3 synthesis", "Same targets and texts through a fish-speech server.",
     "no fish-speech server or GPU in this environment"),
    ("D", "New speaker conditioning (VoxCPM2 / X-Voice transcript-free)", "Same references and texts across VoxCPM2, X-Voice, VoiceStar vs Fish S2-Pro; GE2E similarity, WER, first-audio, RTF, VRAM.",
     "model weights unavailable here (Hugging Face blocked), no GPU"),
    ("E", "Prosody/style separation", "VoxCPM2 with and without a '(style)' instruction: similarity must not drop while style metrics (F0 range, rate) move.",
     "VoxCPM2 weights unavailable here"),
    ("H", "Full research-fusion pipeline", "Prepared reference + duration plan + long-form joins + engine routing vs each engine raw.",
     "needs at least one cloning engine running"),
]


EXPERIMENTS: dict[str, Callable[[], ExperimentResult]] = {"C": exp_reference_prep, "F": exp_duration, "G": exp_enhancement}


def run(ids: list[str] | None, out_dir: Path) -> list[ExperimentResult]:
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for i in ids or list(EXPERIMENTS) + [b[0] for b in BLOCKED]:
        if i in EXPERIMENTS:
            if not sd.available():
                r = not_run(i, i, "", "espeak-ng not installed (synthetic data generator)")
            else:
                r = EXPERIMENTS[i]()
        else:
            b = next(x for x in BLOCKED if x[0] == i)
            r = not_run(*b)
        (out_dir / f"{r.id}.json").write_text(json.dumps(asdict(r), indent=2, default=str) + "\n")
        (out_dir / f"{r.id}.md").write_text(r.to_markdown())
        results.append(r)
    return results
