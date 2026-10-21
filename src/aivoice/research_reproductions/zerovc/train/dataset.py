"""Zero-VC training data: hop-aligned crops of the source, its anonymised copy (content input), a timbre
reference from another utterance of the same speaker, and optional cached teacher features."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from ...common.data import load_audio, read_manifest
from ...common.features import FeatureCache
from ..config import HOP, SAMPLE_RATE, ZeroVCConfig
from ..model.anonymizer import Anonymizer


class ZeroVCDataset(Dataset):
    def __init__(self, manifest: str | list[dict[str, Any]], cfg: ZeroVCConfig, split: str | None = "train",
                 cache_dir: str | None = None, teacher_cache: FeatureCache | None = None,
                 timbre_cache: FeatureCache | None = None, seed: int = 0, anonymize: bool = True,
                 fixed_crops: bool = False) -> None:
        rows = read_manifest(manifest, split) if isinstance(manifest, str) else manifest
        self.rows = [r for r in rows if r.get("duration", 1e9) >= cfg.min_utt_s]
        if not self.rows and cfg.test_standin:  # tiny-overfit clips may be shorter than the paper's 4 s filter
            self.rows = rows
        if not self.rows:
            raise ValueError(f"no utterances >= {cfg.min_utt_s} s in the manifest (paper discards shorter ones)")
        self.cfg, self.cache_dir, self.seed, self.epoch = cfg, cache_dir, seed, 0
        self.fixed = fixed_crops  # overfit gate: identical crops / anonymisation every step
        self.seg = int(cfg.segment_s * SAMPLE_RATE) // HOP * HOP
        self.ref = int(cfg.timbre.reference_s * SAMPLE_RATE) // HOP * HOP
        self.anon = Anonymizer(cfg.anonymizer.kind, cfg.anonymizer.alpha_range, cfg.anonymizer.lpc_order) \
            if anonymize and cfg.stage == "vc" else None
        self.teacher_cache, self.timbre_cache = teacher_cache, timbre_cache
        self.by_spk: dict[str, list[int]] = {}
        for i, r in enumerate(self.rows):
            self.by_spk.setdefault(r["speaker"], []).append(i)

    def __len__(self) -> int:
        return len(self.rows)

    def _crop(self, x: np.ndarray, n: int, rng: random.Random) -> tuple[np.ndarray, int]:
        if len(x) < n:
            return np.pad(x, (0, n - len(x))), 0
        start = rng.randrange(0, (len(x) - n) // HOP + 1) * HOP
        return x[start:start + n], start

    def __getitem__(self, i: int) -> dict[str, Any]:
        rng = random.Random(hash((self.seed, 0 if self.fixed else self.epoch, i)) & 0xFFFFFFFF)
        r = self.rows[i]
        x = load_audio(r["path"], SAMPLE_RATE, self.cache_dir)
        y, start = self._crop(x, self.seg, rng)
        item: dict[str, Any] = {"audio": torch.from_numpy(y), "id": r["id"]}
        if self.teacher_cache is not None:  # [frames, D] or [frames] codes for the whole utterance
            f = np.asarray(self.teacher_cache.get(r["id"]))
            nf = self.seg // HOP
            sl = f[start // HOP:start // HOP + nf]
            if len(sl) < nf:
                sl = np.concatenate([sl, np.repeat(sl[-1:], nf - len(sl), 0)]) if len(sl) else np.zeros((nf,) + f.shape[1:], f.dtype)
            item["teacher"] = torch.from_numpy(np.ascontiguousarray(sl))
        if self.anon is not None:
            ext = None
            if r.get("anonymized_path"):
                full = load_audio(r["anonymized_path"], SAMPLE_RATE, self.cache_dir)
                ext = np.pad(full[start:start + self.seg], (0, max(0, self.seg - len(full[start:start + self.seg]))))
            a = self.anon(y, np.random.default_rng(rng.randrange(2**31)), ext)
            item["content_in"] = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
        if self.cfg.stage == "vc":
            others = [j for j in self.by_spk[r["speaker"]] if j != i]
            j = rng.choice(others) if others else i
            rr = self.rows[j]
            if self.timbre_cache is not None:
                f = np.asarray(self.timbre_cache.get(rr["id"]))
                nf = self.ref // HOP
                s0 = rng.randrange(0, max(1, len(f) - nf + 1))
                item["ref_feats"] = torch.from_numpy(np.ascontiguousarray(f[s0:s0 + nf]))
            else:
                ref = x if j == i else load_audio(rr["path"], SAMPLE_RATE, self.cache_dir)
                item["reference"] = torch.from_numpy(self._crop(ref, self.ref, rng)[0])
        return item
