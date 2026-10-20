#!/usr/bin/env python3
"""Check the showcase: every referenced file exists, matches its hash, is labelled
synthetic, comes from an authorized speaker, and has consistent loudness; README
and demo-page links resolve. Exit code 1 on any problem."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from aivoice.evaluation.metrics import read_wav_comment  # noqa: E402


def main() -> int:
    audio = ROOT / "examples" / "audio"
    m = json.loads((audio / "manifest.json").read_text())
    problems: list[str] = []
    speakers = {s["id"]: s for s in m["speakers"]}
    files = list(m["samples"]) + [f for e in m["engines"] for f in e.get("files", [])]
    for e in m["engines"]:
        if e["status"] == "unavailable" and e.get("files"):
            problems.append(f"{e['engine']}: marked unavailable but lists files")
    lufs = []
    for s in files:
        p = audio / s["file"]
        if not p.is_file():
            problems.append(f"missing {s['file']}")
            continue
        if hashlib.sha256(p.read_bytes()).hexdigest() != s["sha256"]:
            problems.append(f"hash mismatch {s['file']}")
        if "Synthetic speech" not in (read_wav_comment(p) or ""):
            problems.append(f"no synthetic label in {s['file']}")
        spk = speakers.get(s.get("speaker") or s["file"].split("/")[0])
        if not spk or not (spk.get("synthesis_authorized") and spk.get("publication_authorized")):
            problems.append(f"unauthorized speaker for {s['file']}")
        if s.get("lufs") is not None:
            lufs.append(s["lufs"])
    if lufs and max(lufs) - min(lufs) > 1.5:
        problems.append(f"loudness spread {max(lufs) - min(lufs):.1f} LU > 1.5 (A/B comparisons would be biased)")
    for doc in (ROOT / "README.md", ROOT / "docs" / "audio-showcase.md"):
        if not doc.is_file():
            continue
        for link in re.findall(r"\]\((examples/audio/[^)]+|\.\./examples/audio/[^)]+)\)", doc.read_text()):
            target = (doc.parent / link).resolve()
            if not target.is_file():
                problems.append(f"{doc.name}: broken link {link}")
    page = ROOT / "docs" / "demo" / "index.html"
    if page.is_file() and "manifest.json" not in page.read_text():
        problems.append("demo page does not load the manifest")
    for p in problems:
        print("PROBLEM:", p)
    print(f"checked {len(files)} files: {'OK' if not problems else f'{len(problems)} problem(s)'}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
