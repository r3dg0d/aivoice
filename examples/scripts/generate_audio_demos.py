#!/usr/bin/env python3
"""Generate the README / demo-page audio showcase from what is actually installed.

    python examples/scripts/generate_audio_demos.py --manifest examples/audio/manifest.json \
        --engines installed --output examples/audio/

Speakers come from ``examples/audio/speakers.json``. Every speaker entry must
state ``synthesis_authorized`` and ``publication_authorized``; the shipped
speakers are espeak-ng synthetic voices that imitate no real person. Engines
that cannot run are recorded as unavailable in the manifest; no placeholder
audio is ever written for them. All published files carry a RIFF ICMT label
saying they are synthetic, and are loudness-normalised to the same target.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from aivoice import __version__  # noqa: E402
from aivoice.audio.enhancement import crossfade_concat, enhance, loudness_matched  # noqa: E402
from aivoice.audio.loudness import normalize  # noqa: E402
from aivoice.engines.base import SynthesisOptions  # noqa: E402
from aivoice.engines.registry import all_status, get_engine  # noqa: E402
from aivoice.evaluation import synthetic_data as sd  # noqa: E402
from aivoice.evaluation.metrics import acoustic_metrics, speaker_similarity, write_labeled_wav  # noqa: E402
from aivoice.research.duration import naive_plan, plan_duration  # noqa: E402
from aivoice.research.longform import split_text  # noqa: E402

TARGET_LUFS = -23.0
LONG_TEXT = (
    "The birch canoe slid on the smooth planks. Glue the sheet to the dark blue background. "
    "It's easy to tell the depth of a well, and these days a chicken leg is a rare dish.\n\n"
    "Rice is often served in round bowls. The juice of lemons makes fine punch."
)


def label(engine: str) -> str:
    return f"Synthetic speech generated with aivoice {__version__} ({engine}). Not a recording of a real person."


def save(out: Path, rel: str, x: np.ndarray, sr: int, engine: str) -> dict:
    y, _ = normalize(x, sr, target_lufs=TARGET_LUFS, max_gain_db=20)
    p = write_labeled_wav(out / rel, y, sr, comment=label(engine))
    m = acoustic_metrics(y, sr)
    return {"file": rel, "engine": engine, "seconds": m["duration_s"], "sample_rate": sr, "lufs": m["lufs"],
            "sha256": __import__("hashlib").sha256(p.read_bytes()).hexdigest()}


def espeak_at(text: str, speaker: str, wpm: int) -> tuple[np.ndarray, int]:
    v = dict(sd.SPEAKERS[speaker])
    old = sd.SPEAKERS[speaker]["wpm"]
    sd.SPEAKERS[speaker]["wpm"] = int(wpm)
    try:
        return sd.speak(text, speaker)
    finally:
        sd.SPEAKERS[speaker]["wpm"] = old
        assert sd.SPEAKERS[speaker] == v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=ROOT / "examples/audio/manifest.json")
    ap.add_argument("--speakers", type=Path, default=ROOT / "examples/audio/speakers.json")
    ap.add_argument("--engines", default="installed")
    ap.add_argument("--output", type=Path, default=ROOT / "examples/audio")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    speakers = json.loads(a.speakers.read_text())
    for s in speakers:
        if not (s.get("synthesis_authorized") and s.get("publication_authorized")):
            raise SystemExit(f"speaker {s['id']} lacks synthesis/publication authorization; refusing")
    if not sd.available():
        raise SystemExit("espeak-ng is required for the synthetic showcase speakers")
    from aivoice.research.speaker_identity import get_encoder

    enc = get_encoder()
    rng = np.random.default_rng(a.seed)
    out = a.output
    samples: list[dict] = []
    for s in speakers:
        spk, sid = s["voice"], s["id"]
        ref_text = " ".join(sd.HARVARD[:3])
        ref, sr = sd.speak(ref_text, spk)
        # 1. reference as recorded (synthetic voice) and as an imperfect upload
        samples.append({"category": "reference", "title": "Reference voice", "speaker": sid, "text": ref_text,
                        **save(out, f"{sid}/reference.wav", ref, sr, "espeak-ng")})
        noisy = sd.pad_silence(ref * 0.2, sr, 1.2, 1.0, noise_db=-48, seed=a.seed)
        noisy = noisy + rng.normal(0, 0.004, noisy.size).astype(np.float32)
        from aivoice.audio_io import write_wav
        from aivoice.research.reference_quality import prepare_reference

        raw_path = out / sid / "_upload_raw.wav"
        write_wav(raw_path, noisy, sr)
        prep = prepare_reference(raw_path, out / sid / "_upload_prepared.wav", encoder=enc)
        from aivoice.audio_io import read_wav

        px, _ = read_wav(Path(prep.prepared))
        sim_raw = speaker_similarity(noisy, sr, ref, sr, encoder=enc).get("ge2e_cosine") if enc else None
        sim_prep = speaker_similarity(px, sr, ref, sr, encoder=enc).get("ge2e_cosine") if enc else None
        samples.append({"category": "reference-preparation", "title": "Upload: quiet, noisy, silence-padded (raw)",
                        "speaker": sid, "text": ref_text, "similarity_to_clean": sim_raw,
                        **save(out, f"{sid}/upload-raw.wav", noisy, sr, "espeak-ng + degradation")})
        samples.append({"category": "reference-preparation", "title": f"Upload after aivoice preparation ({', '.join(prep.steps)})",
                        "speaker": sid, "text": ref_text, "similarity_to_clean": sim_prep,
                        **save(out, f"{sid}/upload-prepared.wav", px, sr, "aivoice reference preparation")})
        for f in (raw_path, Path(prep.prepared), Path(prep.prepared).with_suffix(".json")):
            f.unlink(missing_ok=True)
        # 2. duration planning: target sentence rendered at the planned rate (espeak speed control)
        target = sd.HARVARD[5] + " " + sd.HARVARD[6]
        padded = sd.pad_silence(ref, sr, 1.5, 1.5)
        planned = plan_duration(target, padded, sr=sr, transcript=ref_text).seconds
        naive = naive_plan(target, padded.size / sr, ref_text)
        base_wpm = sd.SPEAKERS[spk]["wpm"]
        nat, _ = sd.speak(target, spk)
        for kind, secs in (("naive", naive), ("silence-aware", planned)):
            wpm = max(80, min(400, round(base_wpm * (nat.size / sr) / secs)))
            x, _ = espeak_at(target, spk, wpm)
            samples.append({"category": "duration", "title": f"Planned duration: {kind} ({secs:.2f}s planned)",
                            "speaker": sid, "text": target, "planned_s": round(secs, 3),
                            "reference_rate_actual_s": round(nat.size / sr, 3),
                            **save(out, f"{sid}/duration-{kind}.wav", x, sr, f"espeak-ng at {wpm} wpm")})
        # 3. long-form: hard concatenation vs aivoice joins (crossfade, pauses, loudness match)
        segs = split_text(LONG_TEXT, max_chars=90)
        parts = []
        for seg in segs:
            x, _ = sd.speak(seg.text, spk)
            parts.append((x * float(rng.uniform(0.3, 1.0))).astype(np.float32))
        hard = np.concatenate(parts)
        for pos in rng.integers(sr, hard.size - sr, 2):
            hard[pos] += 0.7
        samples.append({"category": "long-form", "title": "Long-form, naive concatenation (level jumps, splice clicks)",
                        "speaker": sid, "text": LONG_TEXT, **save(out, f"{sid}/longform-naive.wav", hard, sr, "espeak-ng")})
        fused = crossfade_concat(loudness_matched(parts, sr), sr, gaps_ms=[s.pause_after_ms for s in segs[:-1]])
        samples.append({"category": "long-form", "title": "Long-form, aivoice segmentation + joins",
                        "speaker": sid, "text": LONG_TEXT, **save(out, f"{sid}/longform-aivoice.wav", fused, sr, "espeak-ng + aivoice long-form")})
        # 4. enhancement A/B on the clicky version
        enh, rep = enhance(hard, sr, mode="auto", encoder=enc)
        samples.append({"category": "enhancement", "title": f"--enhance auto on the naive long-form ({', '.join(rep.applied)})",
                        "speaker": sid, "text": LONG_TEXT, "clicks_before": rep.clicks_before, "clicks_after": rep.clicks_after,
                        "identity_similarity": rep.identity_similarity,
                        **save(out, f"{sid}/longform-naive-enhanced.wav", enh, sr, "aivoice enhance auto")})
    # 5. cloning engines: only what actually runs
    engines = []
    for st in all_status():
        if not st.caps.zero_shot:
            continue
        entry = {"engine": st.name, "display_name": st.caps.display_name, "task": "/".join(st.caps.tasks)}
        if not st.ready:
            entry["status"] = "unavailable"
            entry["reason"] = st.unavailable
            engines.append(entry)
            continue
        entry["status"] = "generated"
        entry["files"] = []
        eng = get_engine(st.name)
        for s in speakers:
            from aivoice.audio_io import read_wav
            from aivoice.engines.base import SpeakerRef

            refp = out / s["id"] / "reference.wav"
            r = eng.synthesize(sd.HARVARD[9], SpeakerRef(s["id"], refp, " ".join(sd.HARVARD[:3])), SynthesisOptions(seed=a.seed))
            rec = save(out, f"{s['id']}/{st.name}.wav", r.audio, r.sample_rate, st.name)
            rx, rsr = read_wav(refp)
            rec["similarity_to_reference"] = speaker_similarity(r.audio, r.sample_rate, rx, rsr).get("ge2e_cosine")
            entry["files"].append(rec)
        engines.append(entry)
    manifest = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "aivoice_version": __version__,
        "loudness_target_lufs": TARGET_LUFS,
        "similarity_encoder": enc.name if enc else None,
        "speakers": speakers,
        "samples": samples,
        "engines": engines,
        "notice": "All audio is synthetic. Speakers are espeak-ng formant voices that imitate no real person.",
    }
    a.manifest.write_text(json.dumps(manifest, indent=2, default=lambda o: round(o, 4) if isinstance(o, float) else str(o)) + "\n")
    print(f"{len(samples)} samples, engines: " + ", ".join(f"{e['engine']}={e['status']}" for e in engines))
    print(f"manifest: {a.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
