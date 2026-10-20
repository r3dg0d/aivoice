#!/usr/bin/env python3
"""Build an authorized benchmark dataset of *real* voices (benchmark layout, see aivoice.evaluation.benchmark).

Sources:
  --record NAME            record your own voice now (PipeWire pw-record, or parecord / arecord)
  --reference F --reference-text T [--heldout F2 --heldout-text T2] --name NAME
                           an existing recording of your own voice (or of someone who agreed)
  --librispeech N          N speakers from LibriSpeech test-clean (CC BY 4.0, public-domain LibriVox readings),
                           downloaded from openslr.org. Benchmark use only: publication_authorized is false.

Each speaker gets reference.wav (+ .txt), heldout.wav (+ .txt, a different utterance for the real-speech
ceiling) and consent.json. Nothing is sent anywhere; LibriSpeech is downloaded once into --cache.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

LIBRISPEECH_URLS = [
    "https://www.openslr.org/resources/12/test-clean.tar.gz",
    "https://us.openslr.org/resources/12/test-clean.tar.gz",
    "https://openslr.trmal.net/resources/12/test-clean.tar.gz",
]
# Harvard sentences (IEEE 1969, public domain): phonetically balanced.
READ_REFERENCE = (
    "The birch canoe slid on the smooth planks. Glue the sheet to the dark blue background. "
    "It's easy to tell the depth of a well. These days a chicken leg is a rare dish. "
    "Rice is often served in round bowls. The juice of lemons makes fine punch."
)
READ_HELDOUT = "The box was thrown beside the parked truck. The hogs were fed chopped corn and garbage."


def write_speaker(out: Path, name: str, ref: np.ndarray, ref_text: str, held: np.ndarray | None, held_text: str | None,
                  sr: int, consent: dict) -> None:
    from aivoice.audio_io import write_wav

    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    write_wav(d / "reference.wav", ref, sr)
    if ref_text.strip():
        (d / "reference.txt").write_text(ref_text.strip() + "\n")
    if held is not None and held_text:
        write_wav(d / "heldout.wav", held, sr)
        (d / "heldout.txt").write_text(held_text.strip() + "\n")
    (d / "consent.json").write_text(json.dumps(consent, indent=2) + "\n")
    print(f"  {name}: reference {ref.size / sr:.1f} s" + (f", held-out {held.size / sr:.1f} s" if held is not None else ""))


# --- your own voice ------------------------------------------------------------


def _recorder(path: Path, seconds: int) -> list[str]:
    if shutil.which("pw-record"):
        return ["timeout", str(seconds), "pw-record", "--rate", "48000", "--channels", "1", "--format", "s16", str(path)]
    if shutil.which("parecord"):
        return ["timeout", str(seconds), "parecord", "--rate=48000", "--channels=1", "--format=s16le",
                "--file-format=wav", str(path)]
    if shutil.which("arecord"):
        return ["arecord", "-q", "-f", "S16_LE", "-r", "48000", "-c", "1", "-d", str(seconds), str(path)]
    raise SystemExit("no recorder found (pw-record, parecord or arecord)")


def record(path: Path, text: str, seconds: int) -> tuple[np.ndarray, int]:
    from aivoice.audio_io import read_wav

    print("\nRead this aloud, at your normal pace, in a quiet room:\n")
    print("   " + text + "\n")
    input(f"Press Enter, then start reading. Recording stops after {seconds} s. ")
    subprocess.run(_recorder(path, seconds), check=False)
    x, sr = read_wav(path)
    if x.size < sr:
        raise SystemExit(f"recording at {path} is empty: check the default input device (pactl get-default-source)")
    return x, sr


def ask(q: str) -> bool:
    return input(q + " [y/N] ").strip().lower() in ("y", "yes")


def own_voice(out: Path, name: str, args: argparse.Namespace) -> None:
    from aivoice.audio_io import read_wav

    print(f"\n== {name}: your own voice")
    if not ask("Is this your own voice (or do you have the speaker's explicit permission to clone it)?"):
        raise SystemExit("stopping: only clone voices you have the rights to")
    publish = ask("May the cloned samples be published (e.g. on your GitHub Pages)?")
    tmp = out / f"_{name}-raw"
    tmp.mkdir(parents=True, exist_ok=True)
    if args.reference:
        x, sr = read_wav(Path(args.reference))
        ref_text = args.reference_text or ""
        h = None
        if args.heldout:
            from aivoice.audio.resample import resample

            h, hsr = read_wav(Path(args.heldout))
            h = h if hsr == sr else resample(h, hsr, sr)
        held_text = args.heldout_text
    else:
        x, sr = record(tmp / "reference.wav", READ_REFERENCE, args.seconds)
        ref_text = READ_REFERENCE
        h, _ = record(tmp / "heldout.wav", READ_HELDOUT, 8)
        held_text = READ_HELDOUT
    shutil.rmtree(tmp, ignore_errors=True)
    write_speaker(out, name, x, ref_text, h, held_text, sr,
                  {"synthesis_authorized": True, "publication_authorized": publish, "source": "own voice (attested)"})


# --- LibriSpeech -----------------------------------------------------------------


def fetch_librispeech(cache: Path) -> Path:
    root = cache / "LibriSpeech"
    if (root / "test-clean").is_dir():
        return root
    cache.mkdir(parents=True, exist_ok=True)
    tgz = cache / "test-clean.tar.gz"
    if not tgz.is_file():
        for url in LIBRISPEECH_URLS:
            try:
                print(f"downloading {url} (346 MB, CC BY 4.0)")
                tmp = tgz.with_suffix(".part")
                with urllib.request.urlopen(url, timeout=60) as r, tmp.open("wb") as f:
                    shutil.copyfileobj(r, f, length=1 << 20)
                tmp.rename(tgz)
                break
            except OSError as e:
                print(f"  failed: {e}")
        else:
            raise SystemExit("could not download LibriSpeech test-clean; pass --no-librispeech or download it to " + str(tgz))
    with tarfile.open(tgz) as t:
        t.extractall(cache, filter="data")
    return root


def _sentence_case(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\bi\b", "I", s)
    return (s[:1].upper() + s[1:] + ".") if s else s


def librispeech(out: Path, n: int, cache: Path, seed: int = 0) -> None:
    import soundfile as sf  # type: ignore

    root = fetch_librispeech(cache)
    sex = {}
    for line in (root / "SPEAKERS.TXT").read_text(errors="replace").splitlines():
        if line.startswith(";") or "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 3 and parts[2] == "test-clean":
            sex[parts[0]] = parts[1]
    spk_dirs = sorted(d for d in (root / "test-clean").iterdir() if d.is_dir() and d.name in sex)
    rng = random.Random(seed)
    females = [d for d in spk_dirs if sex[d.name] == "F"]
    males = [d for d in spk_dirs if sex[d.name] == "M"]
    rng.shuffle(females)
    rng.shuffle(males)
    chosen = [x for pair in zip(females, males) for x in pair][:n]
    print(f"\n== LibriSpeech test-clean: {len(chosen)} speakers (CC BY 4.0; benchmark only, not for publication)")
    for d in chosen:
        utts = []
        for chap in sorted(c for c in d.iterdir() if c.is_dir()):
            trans = {}
            for line in next(chap.glob("*.trans.txt")).read_text().splitlines():
                k, _, t = line.partition(" ")
                trans[k] = t
            for f in sorted(chap.glob("*.flac")):
                x, sr = sf.read(f, dtype="float32")
                utts.append((chap.name, x, sr, _sentence_case(trans.get(f.stem, ""))))
        if not utts:
            continue
        sr = utts[0][2]
        gap = np.zeros(int(0.25 * sr), np.float32)
        first_chap = utts[0][0]
        ref, texts, used = [], [], 0
        for i, (chap, x, _, t) in enumerate(utts):
            if chap != first_chap or 3 > x.size / sr or x.size / sr > 12:
                continue
            ref += [x, gap]
            texts.append(t)
            used = i
            if sum(a.size for a in ref) / sr >= 10:
                break
        held = next(((x, t) for chap, x, _, t in utts if chap != first_chap and 3 <= x.size / sr <= 10), None)
        if held is None:
            held = next(((x, t) for i, (chap, x, _, t) in enumerate(utts) if i > used and 3 <= x.size / sr <= 10), (None, None))
        write_speaker(out, f"libri-{d.name}-{sex[d.name].lower()}", np.concatenate(ref[:-1]), " ".join(texts),
                      held[0], held[1], sr,
                      {"synthesis_authorized": True, "publication_authorized": False,
                       "source": f"LibriSpeech test-clean speaker {d.name}",
                       "license": "CC BY 4.0 (Panayotov et al. 2015, LibriVox public-domain recordings)",
                       "note": "research benchmark only; do not publish clones of this speaker"})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--record", metavar="NAME", help="record your own voice as speaker NAME")
    ap.add_argument("--seconds", type=int, default=20)
    ap.add_argument("--name", default="me")
    ap.add_argument("--reference", type=Path)
    ap.add_argument("--reference-text")
    ap.add_argument("--heldout", type=Path)
    ap.add_argument("--heldout-text")
    ap.add_argument("--librispeech", type=int, default=0, metavar="N")
    ap.add_argument("--librispeech-root", type=Path, help="an already extracted LibriSpeech/ directory")
    ap.add_argument("--cache", type=Path, default=Path.home() / ".cache" / "aivoice" / "datasets")
    ap.add_argument("--texts", type=Path, default=Path(__file__).with_name("texts.txt"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.record:
        own_voice(a.out, a.record, a)
    elif a.reference:
        own_voice(a.out, a.name, a)
    if a.librispeech:
        if a.librispeech_root:
            dst = a.cache / "LibriSpeech"
            if not dst.exists():
                a.cache.mkdir(parents=True, exist_ok=True)
                dst.symlink_to(a.librispeech_root.resolve())
        librispeech(a.out, a.librispeech, a.cache)
    if a.texts.is_file():
        shutil.copy(a.texts, a.out / "texts.txt")
    n = sum(1 for d in a.out.iterdir() if (d / "consent.json").is_file())
    if not n:
        raise SystemExit("no speakers prepared: use --record, --reference and/or --librispeech")
    print(f"\ndataset: {a.out} ({n} speakers)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
