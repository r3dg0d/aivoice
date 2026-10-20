"""`aivoice benchmark research`: every engine on the same references, texts and seeds.

Dataset layout (authorized material only)::

    dataset/<speaker>/reference.wav
    dataset/<speaker>/reference.txt        optional transcript
    dataset/<speaker>/consent.json         {"synthesis_authorized": true, "publication_authorized": false}
    dataset/<speaker>/heldout.wav          optional: another real utterance of the speaker
    dataset/<speaker>/heldout.txt          its transcript (gives a real-speech ceiling for similarity / WER)
    dataset/texts.txt                      one target sentence per line (optional)

Speakers without ``synthesis_authorized: true`` are skipped. ``--synthetic``
builds a rights-free dataset from espeak-ng voices instead.

TTS engines synthesize the texts; voice-conversion engines convert a common
neutral source (espeak-ng reading the same text) into each target, so both are
evaluated on matched content but reported separately (different tasks).
"""

from __future__ import annotations

import csv
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..engines.base import TASK_VC, ConversionOptions, SpeakerRef, SynthesisOptions

DEFAULT_TEXTS = [
    "Please call Stella and ask her to bring these things with her from the store.",
    "The quick brown fox jumps over the lazy dog near the riverbank.",
    "We will meet at seven thirty tomorrow, unless the train is late again.",
]


def load_dataset(root: Path) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    speakers, skipped = [], []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        ref = next((d / n for n in ("reference.wav", "reference.flac") if (d / n).is_file()), None)
        consent = d / "consent.json"
        ok = False
        if consent.is_file():
            try:
                ok = bool(json.loads(consent.read_text()).get("synthesis_authorized"))
            except json.JSONDecodeError:
                ok = False
        if ref is None or not ok:
            skipped.append(f"{d.name}: {'no reference.wav' if ref is None else 'no synthesis authorization in consent.json'}")
            continue
        txt = d / "reference.txt"
        held, held_txt = d / "heldout.wav", d / "heldout.txt"
        speakers.append({"name": d.name, "audio": ref, "transcript": txt.read_text().strip() if txt.is_file() else None,
                         "heldout": held if held.is_file() else None,
                         "heldout_text": held_txt.read_text().strip() if held_txt.is_file() else None,
                         "publishable": bool(json.loads(consent.read_text()).get("publication_authorized"))})
    texts_file = root / "texts.txt"
    texts = [t.strip() for t in texts_file.read_text().splitlines() if t.strip()] if texts_file.is_file() else DEFAULT_TEXTS
    return speakers, texts, skipped


def synthetic_dataset(root: Path, n_speakers: int = 2) -> Path:
    from ..audio_io import write_wav
    from . import synthetic_data as sd

    root.mkdir(parents=True, exist_ok=True)
    for spk in list(sd.SPEAKERS)[:n_speakers]:
        d = root / spk
        d.mkdir(exist_ok=True)
        text = " ".join(sd.HARVARD[:3])
        x, sr = sd.speak(text, spk)
        write_wav(d / "reference.wav", x, sr)
        (d / "reference.txt").write_text(text + "\n")
        (d / "consent.json").write_text(json.dumps({"synthesis_authorized": True, "publication_authorized": True,
                                                    "note": "espeak-ng synthetic voice; imitates no real person"}) + "\n")
    return root


def run_research_benchmark(
    engines: list[str] | None,
    dataset: Path,
    output: Path,
    *,
    seeds: tuple[int, ...] = (0, 1),
    prepare: bool = True,
    device: str | None = None,
    save_audio: bool = False,
) -> dict[str, Any]:
    from ..engines.registry import get_engine, names
    from ..fusion.router import record_measurement
    from ..hardware import detect, fingerprint, query_nvidia_smi
    from ..realtime.bench import ResourceSampler
    from ..research.reference_quality import prepare_reference
    from ..research.speaker_identity import get_encoder
    from .metrics import acoustic_metrics, available_asr, intelligibility, speaker_similarity

    output.mkdir(parents=True, exist_ok=True)
    speakers, texts, skipped = load_dataset(dataset)
    enc = get_encoder()
    asr = available_asr()
    hw = detect()
    rows: list[dict[str, Any]] = []
    meta = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "fingerprint": fingerprint(hw),
        "device": ", ".join(g.name for g in hw.gpus) or f"CPU: {hw.cpu_model}", "python": platform.python_version(),
        "similarity_encoder": enc.name if enc else None, "asr": asr.get_capabilities().name if asr else None,
        "skipped_speakers": skipped, "prepare_reference": prepare, "seeds": list(seeds),
    }
    prepared: dict[str, Path] = {}
    for spk in speakers:
        if prepare:
            rep = prepare_reference(spk["audio"], output / "_prepared" / f"{spk['name']}.wav", encoder=enc)
            prepared[spk["name"]] = Path(rep.prepared)
        else:
            prepared[spk["name"]] = Path(spk["audio"])
    neutral = get_engine("espeak")
    from ..audio_io import read_wav

    # Real-speech ceiling: a held-out recording of the same speaker scored like a synthesis.
    for spk in speakers:
        if spk["heldout"] is None:
            continue
        hx, hsr = read_wav(Path(spk["heldout"]))
        rx, rsr = read_wav(Path(spk["audio"]))
        sim = speaker_similarity(hx, hsr, rx, rsr, encoder=enc) if enc else {}
        intel = intelligibility(hx, hsr, spk["heldout_text"], asr=asr) if spk["heldout_text"] else {}
        rows.append({"engine": "ground-truth", "task": "real speech", "speaker": spk["name"], "text": spk["heldout_text"],
                     "status": "ok", "speaker_similarity": sim.get("ge2e_cosine"), "wer": intel.get("wer"),
                     "cer": intel.get("cer"), "hypothesis": intel.get("hypothesis")})

    def gpu_used() -> int | None:
        gs = query_nvidia_smi()
        return sum(g.total_mib - g.free_mib for g in gs) if gs else None

    for name in engines or names():
        eng = get_engine(name)
        caps = eng.capabilities()
        why = eng.available()
        if why or not (caps.zero_shot or TASK_VC in caps.tasks):
            rows.append({"engine": name, "status": f"unavailable: {why}" if why else "not a cloning engine"})
            continue
        agg: dict[str, list[float]] = {"speaker_similarity": [], "wer": [], "cer": [], "first_audio_ms": [], "rtf": []}
        vram_before = gpu_used()
        with ResourceSampler() as rs:
            t_load = time.perf_counter()
            try:
                eng.load()
            except Exception as e:  # noqa: BLE001
                rows.append({"engine": name, "status": f"load failed: {e}"})
                continue
            load_ms = (time.perf_counter() - t_load) * 1000
            for spk in speakers:
                ref = SpeakerRef(spk["name"], prepared[spk["name"]], spk["transcript"])
                ref_x, ref_sr = read_wav(Path(spk["audio"]))
                for ti, text in enumerate(texts):
                    for seed in seeds:
                        row: dict[str, Any] = {"engine": name, "task": "vc" if TASK_VC in caps.tasks else "tts",
                                               "speaker": spk["name"], "text": text, "seed": seed, "load_ms": round(load_ms, 1)}
                        try:
                            if TASK_VC in caps.tasks:
                                src = neutral.synthesize(text, None, SynthesisOptions())
                                import tempfile

                                from ..audio_io import write_wav

                                with tempfile.NamedTemporaryFile(suffix=".wav") as f:
                                    write_wav(Path(f.name), src.audio, src.sample_rate)
                                    r = eng.convert(Path(f.name), ref, ConversionOptions(device=device or "cuda"))
                            else:
                                r = eng.synthesize(text, ref, SynthesisOptions(seed=seed))
                        except Exception as e:  # noqa: BLE001
                            row["status"] = f"error: {e}"
                            rows.append(row)
                            continue
                        sim = speaker_similarity(r.audio, r.sample_rate, ref_x, ref_sr, encoder=enc) if enc else {}
                        intel = intelligibility(r.audio, r.sample_rate, text, asr=asr)
                        row.update(status="ok", first_audio_ms=r.first_audio_ms, total_ms=r.total_ms, rtf=r.rtf,
                                   native_sample_rate=r.native_sample_rate, speaker_similarity=sim.get("ge2e_cosine"),
                                   mfcc_similarity=sim.get("mfcc_stats_cosine"), wer=intel.get("wer"), cer=intel.get("cer"),
                                   hypothesis=intel.get("hypothesis"), warnings="; ".join(r.warnings) or None,
                                   **{f"acoustic_{k}": v for k, v in acoustic_metrics(r.audio, r.sample_rate).items()})
                        if save_audio:
                            from .metrics import write_labeled_wav

                            rel = Path("audio") / name / spk["name"] / f"text{ti:02d}-seed{seed}.wav"
                            write_labeled_wav(output / rel, r.audio, r.sample_rate,
                                              comment=f"Synthetic speech generated with aivoice ({name}), cloned from an "
                                                      f"authorized reference. Not a recording of {spk['name']}.")
                            row["audio_file"] = str(rel)
                        rows.append(row)
                        for k in agg:
                            if row.get(k) is not None:
                                agg[k].append(float(row[k]))
            eng.unload()
        summary = {k: (float(np.mean(v)) if v else None) for k, v in agg.items()}
        summary.update(peak_vram_mib=rs.gpu_peak_mib, peak_rss_mib=rs.peak_rss_mib, samples=len(agg["rtf"]),
                       vram_before_mib=vram_before, load_ms=round(load_ms, 1),
                       vram_engine_mib=(rs.gpu_peak_mib - vram_before) if rs.gpu_peak_mib is not None and vram_before is not None else None)
        rows.append({"engine": name, "status": "summary", **summary})
        if summary["samples"]:
            record_measurement(name, summary)
    (output / "results.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=2, default=str) + "\n")
    (output / "report.md").write_text(markdown_report(meta, rows))
    keys = sorted({k for r in rows for k in r})
    with (output / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return {"meta": meta, "rows": rows}


def _fmt(v: Any, d: int = 3) -> str:
    return "-" if v is None else (f"{v:.{d}f}" if isinstance(v, float) else str(v))


def markdown_report(meta: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    """Per-engine summary table (means over speakers x texts x seeds) plus the real-speech ceiling."""
    lines = ["# aivoice research benchmark", "",
             f"- created: {meta['created']}", f"- device: {meta['device']}", f"- python: {meta['python']}",
             f"- similarity encoder: {meta['similarity_encoder']}", f"- ASR for WER: {meta['asr'] or 'none'}",
             f"- seeds: {meta['seeds']}; prepared references: {meta['prepare_reference']}", ""]
    gt = [r for r in rows if r.get("engine") == "ground-truth"]
    if gt:
        sims = [r["speaker_similarity"] for r in gt if r.get("speaker_similarity") is not None]
        wers = [r["wer"] for r in gt if r.get("wer") is not None]
        lines += [f"Real-speech ceiling (held-out recording of the same speaker): similarity "
                  f"{_fmt(float(np.mean(sims)) if sims else None)}, WER {_fmt(float(np.mean(wers)) if wers else None)} "
                  f"over {len(gt)} speakers.", ""]
    lines += ["| Engine | Samples | Similarity | WER | CER | RTF | First audio ms | Load ms | VRAM (engine) MiB | Peak RSS MiB |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r.get("status") == "summary":
            lines.append(f"| {r['engine']} | {r.get('samples')} | {_fmt(r.get('speaker_similarity'))} | {_fmt(r.get('wer'))} | "
                         f"{_fmt(r.get('cer'))} | {_fmt(r.get('rtf'))} | {_fmt(r.get('first_audio_ms'), 0)} | "
                         f"{_fmt(r.get('load_ms'), 0)} | {_fmt(r.get('vram_engine_mib'))} | {_fmt(r.get('peak_rss_mib'))} |")
    bad = [r for r in rows if str(r.get("status", "")).startswith(("unavailable", "load failed", "error"))]
    if bad:
        lines += ["", "## Not measured / errors", ""]
        seen = set()
        for r in bad:
            key = (r["engine"], r["status"][:200])
            if key not in seen:
                seen.add(key)
                lines.append(f"- **{r['engine']}**: {r['status'][:500]}")
    lines += ["", "Similarity is GE2E cosine to the speaker's original reference (evidence, not proof of identity). "
              "WER is an ASR proxy for intelligibility. RTF = synthesis time / audio duration (lower is faster).", ""]
    return "\n".join(lines)
