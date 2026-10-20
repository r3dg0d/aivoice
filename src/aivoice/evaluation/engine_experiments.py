"""Engine experiments that need real cloning models (run on a GPU machine).

* **D** — speaker conditioning across engines: summarised from the research
  benchmark's ``results.json`` (same speakers, texts and seeds for every engine).
* **E** — prosody/style separation on VoxCPM2: the same voice with no
  instruction, a "slow and calm" and a "fast and excited" instruction. Identity
  must hold while speaking rate (and pitch movement) shift the instructed way.
* **H** — the research-fusion pipeline (prepared reference + duration plan) vs
  each engine fed the raw reference, on clean and on degraded references.

Data: an authorized dataset in the benchmark layout (see ``benchmark.py``).
Results use the ablation format of :mod:`aivoice.evaluation.experiments`.
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from .experiments import ExperimentResult, _ci, not_run, write_result

STYLES = {
    "slow-calm": ("speaking slowly and calmly", -1),
    "fast-excited": ("speaking quickly and excitedly", +1),
}
H_TEXTS = 3


def pitch_spread_semitones(x: np.ndarray, sr: int) -> float | None:
    """Std of F0 (semitones) over voiced, loud frames; librosa YIN. None without librosa."""
    try:
        import librosa  # type: ignore
    except ImportError:
        return None
    from ..audio.resample import resample

    y = resample(np.asarray(x, np.float32), sr, 16000) if sr != 16000 else np.asarray(x, np.float32)
    if y.size < 16000 * 0.5:
        return None
    f0 = librosa.yin(y, fmin=60, fmax=500, sr=16000, frame_length=1024, hop_length=256)
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0][: f0.size]
    keep = (rms > 0.1 * rms.max()) & (f0 > 61) & (f0 < 499)
    if keep.sum() < 10:
        return None
    st = 12 * np.log2(f0[keep] / 100.0)
    return float(np.std(st))


def _rate(x: np.ndarray, sr: int, text: str | None) -> float | None:
    from ..research.duration import speaking_rate

    r = speaking_rate(x, sr, transcript=text)
    return r.syl_per_s if r else None


def _profile(name: str, *, original: Path, prepared: Path | None, transcript: str | None) -> Any:
    from ..speakers import SpeakerProfile

    return SpeakerProfile(id=name, voice_name=name, original_audio=str(original),
                          prepared_audio=str(prepared) if prepared else None, transcript=transcript,
                          consent={"attested": True, "statement": "dataset consent.json: synthesis_authorized"})


def _save(out: Path | None, rel: str, x: np.ndarray, sr: int, engine: str, speaker: str) -> str | None:
    if out is None:
        return None
    from .metrics import write_labeled_wav

    write_labeled_wav(out / rel, x, sr, comment=f"Synthetic speech generated with aivoice ({engine}), cloned from an "
                                                 f"authorized reference. Not a recording of {speaker}.")
    return rel


# --- D ------------------------------------------------------------------------


def exp_engines(results_json: Path) -> ExperimentResult:
    t0 = time.perf_counter()
    if not results_json.is_file():
        return not_run("D", "New speaker conditioning across engines", "", f"no benchmark results at {results_json}: "
                       "run `aivoice benchmark research --dataset ...` first")
    data = json.loads(results_json.read_text())
    rows = [r for r in data["rows"] if r.get("status") == "ok"]
    engines = sorted({r["engine"] for r in rows if r["engine"] != "ground-truth"})
    if not engines:
        return not_run("D", "New speaker conditioning across engines", "", "the benchmark produced no engine output")
    metrics: dict[str, Any] = {}
    for e in engines + (["ground-truth"] if any(r["engine"] == "ground-truth" for r in rows) else []):
        er = [r for r in rows if r["engine"] == e]
        metrics[e] = {k: _ci([float(r[k]) for r in er if r.get(k) is not None])
                      for k in ("speaker_similarity", "wer", "cer", "rtf", "first_audio_ms") if any(r.get(k) is not None for r in er)}
    summaries = {r["engine"]: r for r in data["rows"] if r.get("status") == "summary"}
    for e in engines:
        s = summaries.get(e, {})
        metrics[e]["vram_engine_mib"] = s.get("vram_engine_mib")
        metrics[e]["load_ms"] = s.get("load_ms")

    def sim(e: str) -> float:
        return metrics[e].get("speaker_similarity", {}).get("mean", float("-inf"))

    def wer(e: str) -> float:
        return metrics[e].get("wer", {}).get("mean", 0.0)

    have_wer = any("wer" in metrics[e] for e in engines)
    intelligible = [e for e in engines if wer(e) <= 0.2]
    ranking = sorted(intelligible, key=sim, reverse=True) + sorted(set(engines) - set(intelligible), key=sim, reverse=True)
    metrics["ranking_by_similarity_among_wer_le_0.2"] = ranking
    rule = ("ranked by similarity among engines with mean WER <= 0.2" if have_wer
            else "ranked by similarity only: WER was not measured (no evaluation ASR)")
    return ExperimentResult(
        id="D", title="New speaker conditioning across engines (VoxCPM2 / X-Voice / VoiceStar / others)",
        change="Same authorized references, texts and seeds through every installed cloning engine.",
        rationale="VoxCPM2 and X-Voice clone without a transcript; VoiceStar conditions on reference + transcript.",
        implemented="evaluation/benchmark.py run_research_benchmark (prepared references)",
        baseline="each other; plus the real-speech ceiling (held-out recording of the same speaker) when provided",
        status="measured", metrics=metrics,
        decision=("router default on this machine: " + " > ".join(ranking) +
                  f" ({rule}; the router also records these measurements)"),
        regression="see RTF / VRAM per engine",
        limitations=["GE2E cosine is one encoder; WER depends on the evaluation ASR",
                     "X-Voice timings include a model load per call (CLI integration)"],
        data=f"{data['meta'].get('device')}; seeds {data['meta'].get('seeds')}; ASR {data['meta'].get('asr')}",
        seconds=time.perf_counter() - t0,
    )


# --- E ------------------------------------------------------------------------


def exp_style(dataset: Path, *, audio_out: Path | None = None, max_texts: int = 3, seed: int = 0) -> ExperimentResult:
    from ..audio_io import read_wav
    from ..engines.base import SynthesisOptions
    from ..engines.registry import get_engine
    from .benchmark import load_dataset
    from .metrics import speaker_similarity

    t0 = time.perf_counter()
    eng = get_engine("voxcpm2")
    why = eng.available()
    title = "Prosody/style separation (VoxCPM2 natural-language instructions)"
    if why:
        return not_run("E", title, "", f"voxcpm2 unavailable: {why}")
    speakers, texts, _ = load_dataset(dataset)
    if not speakers:
        return not_run("E", title, "", "no authorized speakers in the dataset")
    from ..research.speaker_identity import get_encoder

    enc = get_encoder()
    eng.load()
    d_sim: dict[str, list[float]] = {k: [] for k in STYLES}
    log_rate: dict[str, list[float]] = {k: [] for k in STYLES}
    d_pitch: dict[str, list[float]] = {k: [] for k in STYLES}
    files: list[str] = []
    try:
        for spk in speakers:
            from ..engines.base import SpeakerRef

            ref = SpeakerRef(spk["name"], Path(spk["audio"]), spk["transcript"])
            rx, rsr = read_wav(Path(spk["audio"]))
            for ti, text in enumerate(texts[:max_texts]):
                base = eng.synthesize(text, ref, SynthesisOptions(seed=seed))
                b_sim = speaker_similarity(base.audio, base.sample_rate, rx, rsr, encoder=enc).get("ge2e_cosine")
                b_rate, b_pitch = _rate(base.audio, base.sample_rate, text), pitch_spread_semitones(base.audio, base.sample_rate)
                if ti == 0:
                    f = _save(audio_out, f"E/{spk['name']}/neutral.wav", base.audio, base.sample_rate, "voxcpm2", spk["name"])
                    files += [f] if f else []
                for key, (instr, _) in STYLES.items():
                    r = eng.synthesize(text, ref, SynthesisOptions(seed=seed, style_prompt=instr))
                    s = speaker_similarity(r.audio, r.sample_rate, rx, rsr, encoder=enc).get("ge2e_cosine")
                    if s is not None and b_sim is not None:
                        d_sim[key].append(s - b_sim)
                    rt = _rate(r.audio, r.sample_rate, text)
                    if rt and b_rate:
                        log_rate[key].append(float(np.log(rt / b_rate)))
                    p = pitch_spread_semitones(r.audio, r.sample_rate)
                    if p is not None and b_pitch is not None:
                        d_pitch[key].append(p - b_pitch)
                    if ti == 0:
                        f = _save(audio_out, f"E/{spk['name']}/{key}.wav", r.audio, r.sample_rate, "voxcpm2", spk["name"])
                        files += [f] if f else []
    finally:
        eng.unload()
    metrics: dict[str, Any] = {}
    ok = True
    notes = []
    for key, (instr, direction) in STYLES.items():
        sim_ci, rate_ci = _ci(d_sim[key]), _ci(log_rate[key])
        metrics[key] = {"instruction": instr, "similarity_delta": sim_ci, "log_rate_ratio": rate_ci,
                        "pitch_spread_delta_semitones": _ci(d_pitch[key])}
        identity_ok = sim_ci["n"] > 0 and sim_ci["mean"] >= -0.01
        moved = rate_ci["n"] > 0 and ((rate_ci["ci95"][1] < 0) if direction < 0 else (rate_ci["ci95"][0] > 0))
        notes.append(f"{key}: identity {'held' if identity_ok else 'DROPPED'}, rate {'moved as instructed' if moved else 'did not move reliably'}")
        ok &= identity_ok and moved
    metrics["audio_files"] = files
    return ExperimentResult(
        id="E", title=title,
        change="Same voice and text with no instruction vs a '(slow, calm)' and a '(fast, excited)' instruction.",
        rationale="FlexiVoice / FC-TTS: style should be controllable without changing timbre; VoxCPM2 exposes "
                  "natural-language control natively.",
        implemented="--style -> VoxCPM2 '(instruction)text' prefix (engines/research_engines.py)",
        baseline="the same engine, voice, text and seed without an instruction",
        status="measured", metrics=metrics,
        decision=("RETAIN --style for VoxCPM2: " if ok else "KEEP OPTIONAL (does not meet the rule): ") + "; ".join(notes),
        regression="none at synthesis time (a text prefix)",
        limitations=["speaking rate counts transcript syllables over VAD speech; pitch spread uses YIN",
                     "instruction wording affects the result; only two instructions tested"],
        data=f"{len(speakers)} speakers x {min(max_texts, len(texts))} texts, seed {seed}",
        seconds=time.perf_counter() - t0,
    )


# --- H ------------------------------------------------------------------------


def degrade(x: np.ndarray, sr: int, *, seed: int = 0) -> np.ndarray:
    """A realistic bad upload: 1.5 s silence each side, -20 dB gain, white noise at ~15 dB SNR."""
    rng = np.random.default_rng(seed)
    pad = np.zeros(int(1.5 * sr), np.float32)
    y = np.concatenate([pad, np.asarray(x, np.float32) * 0.1, pad])
    speech_rms = float(np.sqrt(np.mean((np.asarray(x, np.float32) * 0.1) ** 2)) + 1e-9)
    noise = rng.normal(0, speech_rms / (10 ** (15 / 20)), y.size).astype(np.float32)
    return np.clip(y + noise, -1, 1)


def exp_fusion(dataset: Path, engines: list[str] | None = None, *, audio_out: Path | None = None,
               max_texts: int = H_TEXTS, seed: int = 0) -> ExperimentResult:
    from ..audio_io import read_wav, write_wav
    from ..engines.base import SynthesisOptions
    from ..engines.registry import get_engine, names
    from ..fusion.pipeline import FusionConfig, synthesize
    from ..research.reference_quality import prepare_reference
    from ..research.speaker_identity import get_encoder
    from .benchmark import load_dataset
    from .metrics import available_asr, intelligibility, speaker_similarity

    t0 = time.perf_counter()
    title = "Full research-fusion pipeline vs raw engine input"
    speakers, texts, _ = load_dataset(dataset)
    texts = texts[:max_texts]
    cands = engines or [n for n in names() if n in ("voxcpm2", "xvoice", "voicestar")]
    usable = []
    for n in cands:
        e = get_engine(n)
        if not e.available():
            usable.append(n)
    if not usable or not speakers:
        return not_run("H", title, "", "no cloning engine available" if not usable else "no authorized speakers")
    enc, asr = get_encoder(), available_asr()
    diffs: dict[str, dict[str, list[float]]] = {}
    per_engine: dict[str, dict[str, Any]] = {}
    files: list[str] = []
    raw_cfg = FusionConfig(use_prepared_reference=False, plan_duration=False, evaluate=False)
    fus_cfg = FusionConfig(evaluate=False)
    with tempfile.TemporaryDirectory(prefix="aivoice-H-") as td:
        refs: dict[tuple[str, str], tuple[Path, Path]] = {}
        for spk in speakers:
            x, sr = read_wav(Path(spk["audio"]))
            for cond in ("clean", "degraded"):
                raw = Path(td) / f"{spk['name']}-{cond}.wav"
                write_wav(raw, x if cond == "clean" else degrade(x, sr, seed=seed), sr)
                prep = prepare_reference(raw, Path(td) / f"{spk['name']}-{cond}-prepared.wav", encoder=enc)
                refs[(spk["name"], cond)] = (raw, Path(prep.prepared))
        for n in usable:
            eng = get_engine(n)
            eng.load()
            try:
                for spk in speakers:
                    anchor_path = Path(spk["heldout"] or spk["audio"])
                    ax, asr_ = read_wav(anchor_path)
                    cx, csr = read_wav(Path(spk["audio"]))
                    ref_rate = _rate(cx, csr, spk["transcript"])
                    for cond in ("clean", "degraded"):
                        raw, prepared = refs[(spk["name"], cond)]
                        out: dict[str, dict[str, Any]] = {}
                        for variant, cfg, prof in (
                            ("raw", raw_cfg, _profile(spk["name"], original=raw, prepared=None, transcript=spk["transcript"])),
                            ("fusion", fus_cfg, _profile(spk["name"], original=raw, prepared=prepared, transcript=spk["transcript"])),
                        ):
                            for ti, text in enumerate(texts):
                                try:
                                    res = synthesize(text, prof, SynthesisOptions(seed=seed), engine_obj=eng, config=cfg)
                                except Exception as e:  # noqa: BLE001
                                    per_engine.setdefault(n, {}).setdefault("errors", []).append(f"{variant}/{cond}: {e}"[:300])
                                    continue
                                sim = speaker_similarity(res.audio, res.sample_rate, ax, asr_, encoder=enc).get("ge2e_cosine") if enc else None
                                w = intelligibility(res.audio, res.sample_rate, text, asr=asr).get("wer")
                                rt = _rate(res.audio, res.sample_rate, text)
                                rate_dev = abs(float(np.log(rt / ref_rate))) if rt and ref_rate else None
                                out[f"{variant}/{ti}"] = {"sim": sim, "wer": w, "rate_dev": rate_dev}
                                if ti == 0:
                                    f = _save(audio_out, f"H/{n}/{spk['name']}/{cond}-{variant}.wav", res.audio,
                                              res.sample_rate, n, spk["name"])
                                    files += [f] if f else []
                        for ti in range(len(texts)):
                            a, b = out.get(f"raw/{ti}"), out.get(f"fusion/{ti}")
                            if not a or not b:
                                continue
                            for k in ("sim", "wer", "rate_dev"):
                                if a[k] is not None and b[k] is not None:
                                    diffs.setdefault(f"{n}/{cond}", {}).setdefault(k, []).append(b[k] - a[k])
                                    diffs.setdefault(f"all/{cond}", {}).setdefault(k, []).append(b[k] - a[k])
            finally:
                eng.unload()
    metrics: dict[str, Any] = {k: {m: _ci(v) for m, v in d.items()} for k, d in sorted(diffs.items())}
    metrics["errors"] = {k: v.get("errors") for k, v in per_engine.items() if v.get("errors")}
    metrics["audio_files"] = files
    metrics["similarity_anchor"] = "held-out recording when provided, else the clean original reference"

    def verdict(cond: str) -> str:
        d = metrics.get(f"all/{cond}", {})
        sim, wer, rd = (d.get(k) or {"n": 0} for k in ("sim", "wer", "rate_dev"))
        if not (sim["n"] or wer["n"] or rd["n"]):
            return f"{cond}: no paired results"
        better = (sim["n"] and sim["ci95"][0] > 0) or (rd["n"] and rd["ci95"][1] < 0) or (wer["n"] and wer["ci95"][1] < 0)
        worse = (sim["n"] and sim["mean"] < -0.01) or (wer["n"] and wer["mean"] > 0.02)
        return f"{cond}: " + ("RETAIN" if better and not worse else "REJECT" if worse else "NEUTRAL (no significant change)")

    v = [verdict("clean"), verdict("degraded")]
    return ExperimentResult(
        id="H", title=title,
        change="Prepared reference + silence-aware duration plan (fusion pipeline defaults) vs the same engine given the "
               "raw reference with no plan; on clean and on degraded (padded, quiet, noisy) references.",
        rationale="Combines C (reference preparation) and F (duration planning) end to end on real engines.",
        implemented="fusion/pipeline.py synthesize(FusionConfig()) vs FusionConfig(use_prepared_reference=False, plan_duration=False)",
        baseline="raw reference, no duration plan, same engine/text/seed",
        status="measured", metrics=metrics,
        decision="; ".join(v) + " (fusion minus raw: similarity up, WER down, |log rate deviation| down are improvements)",
        regression="reference preparation once per profile; duration plan ~0.2 s per utterance on CPU",
        limitations=["the degradation is synthetic (padding, gain, white noise)",
                     "VoxCPM2 has no duration input, so only reference preparation differs for it"],
        data=f"{len(speakers)} speakers x {len(texts)} texts x 2 conditions, engines {usable}, seed {seed}",
        seconds=time.perf_counter() - t0,
    )


def run_engine_experiments(ids: list[str], *, dataset: Path | None, results_json: Path | None, out_dir: Path,
                           engines: list[str] | None = None, audio_out: Path | None = None) -> list[ExperimentResult]:
    out: list[ExperimentResult] = []
    for i in ids:
        if i == "D":
            r = exp_engines(results_json or Path("benchmark-results/results.json"))
        elif dataset is None:
            r = not_run(i, i, "", "needs --dataset (authorized speakers)")
        elif i == "E":
            r = exp_style(dataset, audio_out=audio_out)
        elif i == "H":
            r = exp_fusion(dataset, engines, audio_out=audio_out)
        else:
            continue
        write_result(r, out_dir)
        out.append(r)
    return out
