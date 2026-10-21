"""TVTSyn training data: 320-aligned crops; HuBERT-L9 k-means codes (content stage targets and token
durations for pitch/energy averaging); log-F0 / log-energy targets; [x-vector; ECAPA] of a reference
utterance of the same speaker (cached; frozen speechbrain teachers)."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from ...common.data import load_audio, read_manifest
from ...common.dsp import f0_yin
from ...common.features import FeatureCache
from ..config import HOP, SAMPLE_RATE, TVTSynConfig


def token_average(values: np.ndarray, tokens: np.ndarray) -> np.ndarray:
    """Average non-zero values over runs of identical consecutive tokens (official recipe, training targets)."""
    out = np.zeros_like(values)
    s = 0
    for e in list(np.nonzero(np.diff(tokens))[0] + 1) + [len(tokens)]:
        seg = values[s:e]
        nz = seg[seg != 0]
        out[s:e] = nz.mean() if len(nz) else 0.0
        s = e
    return out


class TVTSynDataset(Dataset):
    def __init__(self, manifest, cfg: TVTSynConfig, split: str | None = "train", cache_dir: str | None = None,
                 code_cache: FeatureCache | None = None, spk_cache: FeatureCache | None = None, seed: int = 0,
                 fixed_crops: bool = False, standin_codes=None, standin_spk=None) -> None:
        self.rows = read_manifest(manifest, split) if isinstance(manifest, str) else manifest
        self.cfg, self.cache_dir, self.seed, self.epoch, self.fixed = cfg, cache_dir, seed, 0, fixed_crops
        self.code_cache, self.spk_cache = code_cache, spk_cache
        self.standin_codes, self.standin_spk = standin_codes, standin_spk
        if not cfg.test_standin and (code_cache is None or (cfg.stage == "synth" and spk_cache is None)):
            raise ValueError("TVTSyn needs the k-means code cache (and x-vector/ECAPA cache for stage synth): run "
                             "`aivoice reproduce prepare tvtsyn-repro --teacher hubert-base=DIR --teacher xvector=DIR "
                             "--teacher ecapa=DIR` first")
        self.seg = int(cfg.segment_s * SAMPLE_RATE) // HOP * HOP
        self.by_spk: dict[str, list[int]] = {}
        for i, r in enumerate(self.rows):
            self.by_spk.setdefault(r["speaker"], []).append(i)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict[str, Any]:
        rng = random.Random(hash((self.seed, 0 if self.fixed else self.epoch, i)) & 0xFFFFFFFF)
        r = self.rows[i]
        x = load_audio(r["path"], SAMPLE_RATE, self.cache_dir)
        if len(x) < self.seg:
            x = np.pad(x, (0, self.seg - len(x)))
        start = rng.randrange(0, (len(x) - self.seg) // HOP + 1) * HOP
        y = np.ascontiguousarray(x[start:start + self.seg])
        nf = self.seg // HOP
        if self.code_cache is not None:
            codes = np.asarray(self.code_cache.get(r["id"]))[start // HOP:start // HOP + nf].astype(np.int64)
            codes = np.pad(codes, (0, nf - len(codes)), mode="edge") if len(codes) else np.zeros(nf, np.int64)
        else:
            codes = self.standin_codes(y)
        item: dict[str, Any] = {"audio": torch.from_numpy(y), "codes": torch.from_numpy(codes), "id": r["id"]}
        if self.cfg.stage == "synth":
            f0, _ = f0_yin(y, SAMPLE_RATE, HOP)
            f0 = np.pad(f0, (0, nf - len(f0)))[:nf]
            lf0 = np.where(f0 > 0, np.log(np.maximum(f0, 1.0)), 0.0).astype(np.float32)
            fr = y[: nf * HOP].reshape(nf, HOP)
            le = np.log(np.sqrt((fr ** 2).mean(1)) + 1e-5).astype(np.float32)
            item["pitch"] = torch.from_numpy(token_average(lf0, codes))[None]
            item["energy"] = torch.from_numpy(token_average(le, codes))[None]
            others = [j for j in self.by_spk[r["speaker"]] if j != i]
            j = rng.choice(others) if others else i
            g = np.asarray(self.spk_cache.get(self.rows[j]["id"])) if self.spk_cache is not None else \
                self.standin_spk(load_audio(self.rows[j]["path"], SAMPLE_RATE, self.cache_dir))
            item["g"] = torch.from_numpy(np.asarray(g, np.float32).reshape(-1))
        return item
