"""Dataset tooling: manifests for LibriTTS / VCTK / LibriSpeech / custom folders, filters, resampling cache,
segment datasets and a resumable sampler.

A manifest is JSON lines, one utterance per line::

    {"id": "libritts/84_121123_000007_000000", "path": "/data/LibriTTS/...wav", "speaker": "libritts/84",
     "dataset": "libritts", "duration": 4.3, "sr": 24000, "text": "...", "split": "train"}

Paths are absolute so manifests can be shared between prepare runs; audio is never copied unless
it is resampled into the cache. Nothing is downloaded: point ``--dataset`` at a local copy (the
corpora have their own licenses: LibriTTS / LibriSpeech CC BY 4.0, VCTK CC BY 4.0).
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, Sampler

AUDIO_EXT = (".wav", ".flac", ".mp3", ".ogg")
DATASETS = ("libritts", "vctk", "librispeech", "custom")


@dataclass
class Filters:
    min_duration: float = 1.0
    max_duration: float = 20.0
    min_sr: int = 16000
    max_clip_fraction: float = 0.001  # samples at |x| >= 0.999
    min_rms_db: float = -50.0

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _info(path: Path) -> tuple[float, int]:
    i = sf.info(str(path))
    return i.frames / i.samplerate, i.samplerate


def scan(root: str | Path, kind: str) -> Iterator[dict[str, Any]]:
    """Yield manifest entries (without audio-quality checks) for a corpus laid out as published."""
    root = Path(root).expanduser().resolve()
    if kind not in DATASETS:
        raise ValueError(f"unknown dataset kind {kind!r}; expected one of {DATASETS}")
    if kind == "libritts":  # <subset>/<spk>/<chapter>/<spk>_<chapter>_<utt>.wav + .normalized.txt
        for p in sorted(root.rglob("*.wav")):
            spk = p.name.split("_")[0]
            txt = p.with_suffix(".normalized.txt")
            subset = next((q.name for q in p.parents if q.name.startswith(("train-", "dev-", "test-"))), "")
            yield {"id": f"libritts/{p.stem}", "path": str(p), "speaker": f"libritts/{spk}", "dataset": kind,
                   "text": txt.read_text().strip() if txt.exists() else None,
                   "split": "test" if subset.startswith("test") else "dev" if subset.startswith("dev") else "train"}
    elif kind == "vctk":  # wav48_silence_trimmed/pXXX/pXXX_YYY_mic1.flac + txt/pXXX/pXXX_YYY.txt
        for p in sorted(root.rglob("*.flac")) or sorted(root.rglob("*.wav")):
            if "_mic2" in p.stem:
                continue  # one microphone per utterance
            spk = p.parent.name
            utt = p.stem.replace("_mic1", "")
            txt = root / "txt" / spk / f"{utt}.txt"
            yield {"id": f"vctk/{utt}", "path": str(p), "speaker": f"vctk/{spk}", "dataset": kind,
                   "text": txt.read_text().strip() if txt.exists() else None, "split": "train"}
    elif kind == "librispeech":  # <subset>/<spk>/<chapter>/<spk>-<chapter>-<utt>.flac + <spk>-<chapter>.trans.txt
        trans: dict[str, str] = {}
        for t in root.rglob("*.trans.txt"):
            for line in t.read_text().splitlines():
                k, _, v = line.partition(" ")
                trans[k] = v
        for p in sorted(root.rglob("*.flac")):
            spk = p.name.split("-")[0]
            subset = next((q.name for q in p.parents if q.name.startswith(("train-", "dev-", "test-"))), "")
            yield {"id": f"librispeech/{p.stem}", "path": str(p), "speaker": f"librispeech/{spk}", "dataset": kind,
                   "text": trans.get(p.stem),
                   "split": "test" if subset.startswith("test") else "dev" if subset.startswith("dev") else "train"}
    else:  # custom: <root>/<speaker>/**.wav, or flat files (speaker = "custom/<root name>")
        for p in sorted(q for q in root.rglob("*") if q.suffix.lower() in AUDIO_EXT):
            rel = p.relative_to(root)
            spk = rel.parts[0] if len(rel.parts) > 1 else root.name
            txt = p.with_suffix(".txt")
            yield {"id": f"custom/{rel.with_suffix('').as_posix()}", "path": str(p), "speaker": f"custom/{spk}",
                   "dataset": kind, "text": txt.read_text().strip() if txt.exists() else None, "split": "train"}


def quality_check(entry: dict[str, Any], filters: Filters, full: bool = True) -> str | None:
    """Return a rejection reason, or None. ``full`` reads the audio for clipping/silence checks."""
    try:
        dur, sr = _info(Path(entry["path"]))
    except Exception as e:  # noqa: BLE001 - unreadable file
        return f"unreadable: {e}"
    entry["duration"], entry["sr"] = round(dur, 4), sr
    if dur < filters.min_duration:
        return "too_short"
    if dur > filters.max_duration:
        return "too_long"
    if sr < filters.min_sr:
        return "low_sample_rate"
    if full:
        x, _ = sf.read(entry["path"], dtype="float32", always_2d=True)
        x = x.mean(1)
        if np.mean(np.abs(x) >= 0.999) > filters.max_clip_fraction:
            return "clipping"
        rms = 20 * math.log10(float(np.sqrt(np.mean(x ** 2))) + 1e-12)
        if rms < filters.min_rms_db:
            return "silent"
    return None


def build_manifest(root: str | Path, kind: str, out: str | Path, filters: Filters | None = None,
                   full_check: bool = True, limit: int | None = None) -> dict[str, Any]:
    """Scan + filter, write ``out`` (JSONL) and ``out.stats.json``; returns the stats."""
    filters = filters or Filters()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    kept, rejected = 0, {}
    speakers: dict[str, int] = {}
    hours = 0.0
    with out.open("w") as f:
        for i, e in enumerate(scan(root, kind)):
            if limit is not None and i >= limit:
                break
            why = quality_check(e, filters, full_check)
            if why:
                rejected[why.split(":")[0]] = rejected.get(why.split(":")[0], 0) + 1
                continue
            f.write(json.dumps(e) + "\n")
            kept += 1
            speakers[e["speaker"]] = speakers.get(e["speaker"], 0) + 1
            hours += e["duration"] / 3600
    stats = {"root": str(Path(root).resolve()), "kind": kind, "kept": kept, "rejected": rejected,
             "speakers": len(speakers), "hours": round(hours, 3), "filters": filters.as_dict()}
    Path(str(out) + ".stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    return stats


def read_manifest(path: str | Path, split: str | None = None) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return [r for r in rows if split is None or r.get("split", "train") == split]


def load_audio(path: str | Path, sr: int, cache_dir: str | Path | None = None) -> np.ndarray:
    """Mono float32 at ``sr``. Resampled audio is cached as .npy (keyed by path, mtime and sr)."""
    path = Path(path)
    if cache_dir is not None:
        st = path.stat()
        key = hashlib.sha1(f"{path.resolve()}|{st.st_mtime_ns}|{st.st_size}|{sr}".encode()).hexdigest()
        c = Path(cache_dir) / f"sr{sr}" / key[:2] / f"{key}.npy"
        if c.exists():
            return np.load(c)
    x, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(1)
    if file_sr != sr:
        import soxr

        x = soxr.resample(x, file_sr, sr, quality="HQ").astype(np.float32)
    if cache_dir is not None:
        c.parent.mkdir(parents=True, exist_ok=True)
        tmp = c.with_suffix(".tmp.npy")
        np.save(tmp, x)
        tmp.replace(c)
    return x


class SegmentDataset(Dataset):
    """Fixed-length random crops (padded if short). Item = dict(audio [T], speaker_idx, id, + extras).

    ``with_reference`` adds a second crop of a *different* utterance of the same speaker (timbre
    reference, as in zero-shot VC training) when the speaker has one, else another crop of the same file.
    Cropping is driven by a per-item seed so a resumed run sees the same crops as an uninterrupted one.
    """

    def __init__(self, manifest: str | Path | list[dict[str, Any]], sr: int, segment_s: float,
                 cache_dir: str | Path | None = None, with_reference: bool = False, reference_s: float | None = None,
                 split: str | None = "train", seed: int = 0, hop: int = 1) -> None:
        self.rows = read_manifest(manifest, split) if not isinstance(manifest, list) else manifest
        if not self.rows:
            raise ValueError("empty manifest (after split filter)")
        self.sr, self.cache_dir, self.seed = sr, cache_dir, seed
        self.seg = int(round(segment_s * sr)) // hop * hop
        self.ref = int(round((reference_s or segment_s) * sr)) // hop * hop
        self.with_reference = with_reference
        self.speakers = sorted({r["speaker"] for r in self.rows})
        self.spk_index = {s: i for i, s in enumerate(self.speakers)}
        self.by_spk: dict[str, list[int]] = {}
        for i, r in enumerate(self.rows):
            self.by_spk.setdefault(r["speaker"], []).append(i)
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.rows)

    def _crop(self, x: np.ndarray, n: int, rng: random.Random) -> np.ndarray:
        if len(x) <= n:
            return np.pad(x, (0, n - len(x)))
        s = rng.randrange(0, len(x) - n + 1)
        return x[s:s + n]

    def __getitem__(self, i: int) -> dict[str, Any]:
        rng = random.Random(hash((self.seed, self.epoch, i)) & 0xFFFFFFFF)
        r = self.rows[i]
        x = load_audio(r["path"], self.sr, self.cache_dir)
        item: dict[str, Any] = {"audio": torch.from_numpy(self._crop(x, self.seg, rng)),
                                "speaker_idx": self.spk_index[r["speaker"]], "id": r["id"]}
        if self.with_reference:
            others = [j for j in self.by_spk[r["speaker"]] if j != i]
            j = rng.choice(others) if others else i
            y = x if j == i else load_audio(self.rows[j]["path"], self.sr, self.cache_dir)
            item["reference"] = torch.from_numpy(self._crop(y, self.ref, rng))
        for k in ("emotion", "text"):
            if r.get(k) is not None:
                item[k] = r[k]
        return item


def collate(items: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in items[0]:
        v = [it[k] for it in items]
        if isinstance(v[0], torch.Tensor):
            out[k] = torch.stack(v)
        elif isinstance(v[0], (int, float)):
            out[k] = torch.tensor(v)
        else:
            out[k] = v
    return out


class ResumableSampler(Sampler[int]):
    """Seeded shuffling with an exact resume point (epoch, position). DDP-aware via rank/world_size."""

    def __init__(self, n: int, seed: int = 0, shuffle: bool = True, rank: int = 0, world_size: int = 1) -> None:
        self.n, self.seed, self.shuffle, self.rank, self.world = n, seed, shuffle, rank, world_size
        self.epoch, self.pos = 0, 0

    def _order(self) -> list[int]:
        idx = list(range(self.n))
        if self.shuffle:
            random.Random(self.seed * 1000003 + self.epoch).shuffle(idx)
        per = math.ceil(self.n / self.world)
        idx += idx[: per * self.world - self.n]  # pad so every rank has the same count
        return idx[self.rank::self.world]

    def __iter__(self) -> Iterator[int]:
        order = self._order()
        while self.pos < len(order):
            i = order[self.pos]
            self.pos += 1
            yield i
        self.epoch, self.pos = self.epoch + 1, 0

    def __len__(self) -> int:
        return math.ceil(self.n / self.world)

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self.epoch, "pos": self.pos, "seed": self.seed, "n": self.n}

    def load_state_dict(self, st: dict[str, int]) -> None:
        if st.get("n", self.n) != self.n:
            raise ValueError(f"sampler state is for {st['n']} items, dataset has {self.n}")
        self.epoch, self.pos = int(st["epoch"]), int(st["pos"])


__all__ = ["DATASETS", "Filters", "ResumableSampler", "SegmentDataset", "build_manifest", "collate", "load_audio",
           "quality_check", "read_manifest", "scan"]
