"""StreamVoiceAnon+ pairs: a *neutral* prompt and an *emotional* source utterance of the same speaker.

Inputs are precomputed by ``prepare`` with the official StreamVoiceAnon encoders (semantic ASR-BSQ codes,
firefly acoustic codes, CampPlus style vector, Spark timbre latents) plus frame-level emotion2vec features
(training-only teacher) resampled to the codec frame rate. Manifest rows need ``emotion`` (e.g. ESD:
Neutral / Angry / Happy / Sad / Surprise) and ``speaker``.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from ...common.data import read_manifest
from ...common.features import FeatureCache

NEUTRAL = {"neutral", "neu", "n"}


def build_pairs(rows: list[dict[str, Any]], seed: int = 0) -> list[tuple[int, int]]:
    """(prompt_index, source_index): every emotional utterance paired with a neutral one of its speaker."""
    rng = random.Random(seed)
    neutral: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        if str(r.get("emotion", "")).lower() in NEUTRAL:
            neutral.setdefault(r["speaker"], []).append(i)
    pairs = []
    for i, r in enumerate(rows):
        emo = str(r.get("emotion", "")).lower()
        if emo and emo not in NEUTRAL and neutral.get(r["speaker"]):
            pairs.append((rng.choice(neutral[r["speaker"]]), i))
    return pairs


class PairDataset(Dataset):
    def __init__(self, manifest, cfg, caches: dict[str, FeatureCache] | None = None, split: str | None = "train",
                 seed: int = 0, synthetic: dict[str, Any] | None = None, fixed_crops: bool = False) -> None:
        self.cfg, self.seed, self.epoch, self.fixed = cfg, seed, 0, fixed_crops
        self.synthetic = synthetic
        if synthetic is not None:  # TEST: random tokens with the official shapes
            self.pairs = [(0, 0)] * synthetic.get("n", 4)
            return
        self.rows = read_manifest(manifest, split) if isinstance(manifest, str) else manifest
        self.pairs = build_pairs(self.rows, seed)
        if not self.pairs:
            raise ValueError("no (neutral prompt, emotional source) pairs: manifest rows need `emotion` labels "
                             "with at least one neutral utterance per speaker")
        need = ("semantic", "acoustic", "style", "timbre", "emotion2vec")
        missing = [k for k in need if not caches or k not in caches]
        if missing:
            raise ValueError(f"missing feature caches {missing}: run `aivoice reproduce prepare streamvoiceanon-plus` "
                             "in the StreamVoiceAnon environment first")
        self.c = caches

    def __len__(self) -> int:
        return len(self.pairs)

    def _crop(self, n_total: int, n: int, rng: random.Random) -> int:
        return 0 if n_total <= n or self.fixed else rng.randrange(0, n_total - n + 1)

    def __getitem__(self, k: int) -> dict[str, Any]:
        rng = random.Random(hash((self.seed, 0 if self.fixed else self.epoch, k)) & 0xFFFFFFFF)
        cfg = self.cfg
        P, T = cfg.prompt_frames, cfg.source_frames
        if self.synthetic is not None:
            s = self.synthetic
            g = torch.Generator().manual_seed(k if self.fixed else k + 1000 * self.epoch)
            return {"sem_p": torch.randint(0, s["vocab"], (P,), generator=g),
                    "ac_p": torch.randint(0, s["codebook"], (s["K"], P), generator=g),
                    "sem_s": torch.randint(0, s["vocab"], (T,), generator=g),
                    "ac_s": torch.randint(0, s["codebook"], (s["K"], T), generator=g), "src_lens": T,
                    "timbre": torch.randn(32, s["ctx"], generator=g), "style": torch.randn(s["style"], generator=g),
                    "emo": torch.randn(T, s["emo"], generator=g)}
        pi, si = self.pairs[k]
        rp, rs = self.rows[pi], self.rows[si]
        sem_p, ac_p = np.asarray(self.c["semantic"].get(rp["id"])), np.asarray(self.c["acoustic"].get(rp["id"]))
        sem_s, ac_s = np.asarray(self.c["semantic"].get(rs["id"])), np.asarray(self.c["acoustic"].get(rs["id"]))
        a = self._crop(len(sem_p), P, rng)
        b = self._crop(len(sem_s), T, rng)
        n_s = min(T, len(sem_s) - b)

        def pad(x, n, axis=0):
            w = [(0, 0)] * x.ndim
            w[axis] = (0, max(0, n - x.shape[axis]))
            return np.pad(x, w)

        emo = np.asarray(self.c["emotion2vec"].get(rs["id"]))[b:b + T]
        return {"sem_p": torch.from_numpy(pad(sem_p[a:a + P], P).astype(np.int64)),
                "ac_p": torch.from_numpy(pad(ac_p[a:a + P].T, P, 1).astype(np.int64)),
                "sem_s": torch.from_numpy(pad(sem_s[b:b + T], T).astype(np.int64)),
                "ac_s": torch.from_numpy(pad(ac_s[b:b + T].T, T, 1).astype(np.int64)), "src_lens": n_s,
                "timbre": torch.from_numpy(np.asarray(self.c["timbre"].get(rs["id"]), np.float32)),
                "style": torch.from_numpy(np.asarray(self.c["style"].get(rs["id"]), np.float32).reshape(-1)),
                "emo": torch.from_numpy(pad(emo.astype(np.float32), T))}
