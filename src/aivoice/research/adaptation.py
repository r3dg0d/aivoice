"""Optional speaker adaptation (fine-tuning) dataset preparation.

Zero-shot cloning needs no training. Adaptation trains a speaker-specific
model and is only offered where the engine supports it: VoxCPM2 ships LoRA
fine-tuning (``LoRAConfig``, manifests with ``audio``/``text``/``ref_audio``
columns; trainer ``scripts/train_voxcpm_finetune.py`` in the VoxCPM repository).
This module validates and splits an authorised dataset and writes that
manifest. It never starts training on its own.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

AUDIO_EXT = (".wav", ".flac")


@dataclass
class ClipCheck:
    audio: str
    text: str | None
    duration_s: float
    issues: list[str] = field(default_factory=list)
    similarity_to_centroid: float | None = None


@dataclass
class DatasetReport:
    root: str
    clips: int
    usable: int
    total_minutes: float
    duplicates: list[list[str]]
    issues: dict[str, list[str]]
    outliers: list[str]
    train: list[str]
    val: list[str]
    ready: bool
    notes: list[str] = field(default_factory=list)


def _load_pairs(root: Path) -> list[tuple[Path, str | None]]:
    meta = root / "metadata.csv"
    pairs: list[tuple[Path, str | None]] = []
    if meta.is_file():
        for line in meta.read_text(encoding="utf-8").splitlines():
            if "|" in line:
                name, text = line.split("|", 1)
                p = root / name if Path(name).suffix else root / f"{name}.wav"
                pairs.append((p, text.strip()))
        return pairs
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in AUDIO_EXT:
            txt = p.with_suffix(".txt")
            pairs.append((p, txt.read_text(encoding="utf-8").strip() if txt.is_file() else None))
    return pairs


def validate_dataset(root: str | Path, *, encoder: Any | None = None, val_fraction: float = 0.1, seed: int = 0,
                     min_minutes: float = 5.0) -> DatasetReport:
    from ..audio_io import read_wav

    root = Path(root)
    pairs = _load_pairs(root)
    checks: list[ClipCheck] = []
    hashes: dict[str, list[str]] = {}
    embeds: dict[str, np.ndarray] = {}
    for p, text in pairs:
        c = ClipCheck(str(p), text, 0.0)
        if not p.is_file():
            c.issues.append("missing file")
            checks.append(c)
            continue
        hashes.setdefault(hashlib.sha256(p.read_bytes()).hexdigest(), []).append(str(p))
        try:
            x, sr = read_wav(p)
        except Exception as e:  # noqa: BLE001
            c.issues.append(f"unreadable: {e}")
            checks.append(c)
            continue
        c.duration_s = x.size / sr
        if c.duration_s < 1.0 or c.duration_s > 30:
            c.issues.append(f"duration {c.duration_s:.1f}s outside 1-30 s")
        if np.mean(np.abs(x) >= 0.999) > 1e-4:
            c.issues.append("clipping")
        if sr < 16000:
            c.issues.append(f"sample rate {sr} Hz below 16 kHz")
        if not text:
            c.issues.append("no transcript")
        elif not re.search(r"\w", text):
            c.issues.append("transcript has no words")
        else:
            cps = len(text) / max(c.duration_s, 1e-3)
            if cps < 4 or cps > 30:
                c.issues.append(f"transcript length implausible for audio ({cps:.0f} chars/s)")
        if encoder is not None and not c.issues:
            try:
                embeds[c.audio] = encoder.embed(x, sr)
            except ValueError:
                pass
        checks.append(c)
    outliers: list[str] = []
    if len(embeds) >= 3:
        centroid = np.mean(list(embeds.values()), axis=0)
        centroid /= np.linalg.norm(centroid)
        sims = {k: float(np.dot(v, centroid)) for k, v in embeds.items()}
        med = float(np.median(list(sims.values())))
        for c in checks:
            if c.audio in sims:
                c.similarity_to_centroid = sims[c.audio]
                if sims[c.audio] < med - 0.1:
                    c.issues.append(f"speaker inconsistent with the rest (similarity {sims[c.audio]:.2f})")
                    outliers.append(c.audio)
    dups = [v for v in hashes.values() if len(v) > 1]
    dup_extra = {p for group in dups for p in group[1:]}
    usable = [c for c in checks if not c.issues and c.audio not in dup_extra]
    rnd = random.Random(seed)
    names = [c.audio for c in usable]
    rnd.shuffle(names)
    n_val = max(1, int(len(names) * val_fraction)) if len(names) >= 10 else 0
    minutes = sum(c.duration_s for c in usable) / 60
    notes = []
    if minutes < min_minutes:
        notes.append(f"only {minutes:.1f} usable minutes; VoxCPM's guide suggests 5-10 minutes for LoRA")
    if n_val == 0:
        notes.append("fewer than 10 usable clips: no validation split (overfitting cannot be monitored)")
    return DatasetReport(
        root=str(root), clips=len(checks), usable=len(usable), total_minutes=round(minutes, 2),
        duplicates=dups, issues={c.audio: c.issues for c in checks if c.issues}, outliers=outliers,
        train=sorted(names[n_val:]), val=sorted(names[:n_val]),
        ready=minutes >= min_minutes and len(usable) >= 10, notes=notes,
    )


def write_voxcpm_manifests(report: DatasetReport, out_dir: str | Path, *, ref_audio: str | None = None) -> dict[str, str]:
    """JSONL manifests with the columns VoxCPM's loader reads (audio, text, optional ref_audio)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    texts: dict[str, str] = {}
    for p, t in _load_pairs(Path(report.root)):
        if t:
            texts[str(p)] = t
    paths = {}
    for split, names in (("train", report.train), ("val", report.val)):
        f = out / f"{split}.jsonl"
        with f.open("w", encoding="utf-8") as fh:
            for n in names:
                row = {"audio": n, "text": texts.get(n, "")}
                if ref_audio:
                    row["ref_audio"] = ref_audio
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        paths[split] = str(f)
    (out / "report.json").write_text(json.dumps(asdict(report), indent=2) + "\n", encoding="utf-8")
    (out / "README.txt").write_text(
        "LoRA fine-tuning for VoxCPM2 (run it yourself; aivoice does not train automatically):\n"
        "  git clone https://github.com/OpenBMB/VoxCPM && cd VoxCPM\n"
        "  # set train/val manifest paths in conf/voxcpm_v2/voxcpm_finetune_lora.yaml to:\n"
        f"  #   {paths['train']}\n  #   {paths['val']}\n"
        "  python scripts/train_voxcpm_finetune.py --config_path conf/voxcpm_v2/voxcpm_finetune_lora.yaml\n"
        "Then: aivoice clone synthesize --engine voxcpm2 ... with engines.voxcpm2.lora_path set.\n",
        encoding="utf-8",
    )
    return paths
