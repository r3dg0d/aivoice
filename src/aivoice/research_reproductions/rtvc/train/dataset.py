"""RT-VC training data: 80-sample-aligned crops with pitch / periodicity targets (YIN teacher or cache),
EMA pseudo-labels (cache, or the test stand-in), and a reference crop of another utterance of the speaker."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from ...common.data import load_audio, read_manifest
from ...common.dsp import f0_yin
from ...common.features import FeatureCache
from ..config import FRAME, SAMPLE_RATE, RTVCConfig


class RTVCDataset(Dataset):
    def __init__(self, manifest, cfg: RTVCConfig, split: str | None = "train", cache_dir: str | None = None,
                 ema_cache: FeatureCache | None = None, pitch_cache: FeatureCache | None = None, seed: int = 0,
                 fixed_crops: bool = False) -> None:
        self.rows = read_manifest(manifest, split) if isinstance(manifest, str) else manifest
        if not self.rows:
            raise ValueError("empty manifest")
        self.cfg, self.cache_dir, self.seed, self.epoch, self.fixed = cfg, cache_dir, seed, 0, fixed_crops
        self.ema_cache, self.pitch_cache = ema_cache, pitch_cache
        self.seg = int(cfg.segment_s * SAMPLE_RATE) // FRAME * FRAME
        self.ref = int(cfg.reference_s * SAMPLE_RATE) // FRAME * FRAME
        self.by_spk: dict[str, list[int]] = {}
        for i, r in enumerate(self.rows):
            self.by_spk.setdefault(r["speaker"], []).append(i)
        if cfg.stage in ("ema", "vocoder") and ema_cache is None and not cfg.test_standin:
            raise ValueError("RT-VC stages ema/vocoder need EMA pseudo-labels: run `aivoice reproduce prepare rtvc "
                             "--teacher sparc=DIR` first (SPARC articulatory inversion)")

    def __len__(self) -> int:
        return len(self.rows)

    def _crop(self, x: np.ndarray, n: int, rng: random.Random) -> tuple[np.ndarray, int]:
        if len(x) < n:
            return np.pad(x, (0, n - len(x))), 0
        s = rng.randrange(0, (len(x) - n) // FRAME + 1) * FRAME
        return x[s:s + n], s

    def _frames(self, cache: FeatureCache, item_id: str, start: int, n: int) -> np.ndarray:
        f = np.asarray(cache.get(item_id))
        sl = f[start // FRAME:start // FRAME + n]
        if len(sl) < n:
            sl = np.concatenate([sl, np.zeros((n - len(sl),) + f.shape[1:], f.dtype)])
        return np.ascontiguousarray(sl)

    def _augment(self, x: np.ndarray, rng: random.Random) -> np.ndarray:
        a = self.cfg.augment
        if rng.random() >= a.prob:
            return x
        y = x * 10 ** (rng.uniform(-a.gain_db, a.gain_db) / 20)
        snr = rng.uniform(*a.noise_snr_db)
        p = float(np.mean(y ** 2)) + 1e-10
        y = y + np.random.default_rng(rng.randrange(2**31)).standard_normal(len(y)).astype(np.float32) * np.sqrt(p / 10 ** (snr / 10))
        return np.clip(y, -1, 1).astype(np.float32)

    def __getitem__(self, i: int) -> dict[str, Any]:
        rng = random.Random(hash((self.seed, 0 if self.fixed else self.epoch, i)) & 0xFFFFFFFF)
        r = self.rows[i]
        x = load_audio(r["path"], SAMPLE_RATE, self.cache_dir)
        y, start = self._crop(x, self.seg, rng)
        nf = self.seg // FRAME
        item: dict[str, Any] = {"audio": torch.from_numpy(np.ascontiguousarray(y)), "id": r["id"]}
        if self.cfg.stage == "pitch":
            item["input"] = torch.from_numpy(self._augment(y, rng))
        if self.cfg.stage in ("pitch", "vocoder"):
            if self.pitch_cache is not None:
                fp = self._frames(self.pitch_cache, r["id"], start, nf)  # [n, 2]: f0, periodicity
                f0, per = fp[:, 0], fp[:, 1]
            else:
                f0, per = f0_yin(y, SAMPLE_RATE, FRAME)
                f0, per = np.pad(f0, (0, nf - len(f0)))[:nf], np.pad(per, (0, nf - len(per)))[:nf]
            item["f0"], item["periodicity"] = torch.from_numpy(f0), torch.from_numpy(per)
        if self.cfg.stage == "ema":
            item["input"] = torch.from_numpy(self._augment(y, rng))
        if self.cfg.stage in ("ema", "vocoder") and self.ema_cache is not None:
            item["ema"] = torch.from_numpy(self._frames(self.ema_cache, r["id"], start, nf))  # [n, 12]
        if self.cfg.stage == "vocoder":
            others = [j for j in self.by_spk[r["speaker"]] if j != i]
            j = rng.choice(others) if others else i
            ref = x if j == i else load_audio(self.rows[j]["path"], SAMPLE_RATE, self.cache_dir)
            item["reference"] = torch.from_numpy(np.ascontiguousarray(self._crop(ref, self.ref, rng)[0]))
        return item
