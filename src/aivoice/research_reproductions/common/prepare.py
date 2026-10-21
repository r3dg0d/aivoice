"""Preparation steps shared by recipes: manifests, teacher-feature caches, k-means pseudo-labels."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data import Filters, build_manifest, load_audio, read_manifest
from .features import FeatureCache, FeatureSpec


def ensure_manifest(dataset: str, out_dir: Path, filters: Filters, limit: int | None = None) -> tuple[Path, dict]:
    """``dataset`` is ``KIND:PATH`` (libritts:/data/LibriTTS) or an existing manifest .jsonl."""
    if dataset.endswith(".jsonl") and Path(dataset).exists():
        return Path(dataset), {"manifest": dataset, "reused": True}
    kind, _, root = dataset.partition(":")
    if not root:
        raise ValueError("--dataset must be KIND:PATH (libritts|vctk|librispeech|custom) or a manifest .jsonl")
    out = out_dir / f"manifest-{kind}.jsonl"
    stats = build_manifest(root, kind, out, filters, limit=limit)
    return out, stats


def teacher_spec(teacher: Any, name: str, extra: dict[str, Any] | None = None) -> FeatureSpec:
    return FeatureSpec(name=name, extractor_version="1", source_model=getattr(teacher, "source", None),
                       source_revision=getattr(teacher, "revision", None), sample_rate=16000, hop=320,
                       dim=getattr(teacher, "dim", None), config={"layer": getattr(teacher, "layer", None),
                                                                   "standin": bool(getattr(teacher, "standin", False)),
                                                                   **(extra or {})})


@torch.no_grad()
def cache_teacher_features(teacher: Any, manifest: Path, cache_root: Path, name: str, device: str = "cpu",
                           audio_cache: Path | None = None, max_s: float = 30.0) -> dict[str, Any]:
    teacher = teacher.to(device)
    cache = FeatureCache(cache_root, teacher_spec(teacher, name))
    rows = read_manifest(manifest)
    t0 = time.time()

    def fn(r: dict[str, Any]) -> np.ndarray:
        x = load_audio(r["path"], 16000, audio_cache)[: int(max_s * 16000)]
        return teacher(torch.from_numpy(x)[None].to(device))[0].float().cpu().numpy()

    res = cache.fill(((r["id"], r) for r in rows), fn)
    return {"cache": str(cache.dir), **res, "seconds": round(time.time() - t0, 1)}


def learn_kmeans(cache: FeatureCache, ids: list[str], k: int, seed: int = 0, max_frames: int = 2_000_000,
                 out: Path | None = None) -> np.ndarray:
    """MiniBatch k-means on cached frames (the HuBERT / TVTSyn pseudo-label recipe)."""
    from sklearn.cluster import MiniBatchKMeans

    rng = np.random.default_rng(seed)
    feats, n = [], 0
    for i in rng.permutation(len(ids)):
        f = np.asarray(cache.get(ids[i]))
        feats.append(f)
        n += len(f)
        if n >= max_frames:
            break
    X = np.concatenate(feats)[:max_frames]
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=10000, n_init=3, max_iter=100).fit(X)
    c = km.cluster_centers_.astype(np.float32)
    if out is not None:
        np.save(out, c)
        Path(str(out) + ".json").write_text(json.dumps({"k": k, "frames": len(X), "seed": seed,
                                                        "source_cache": str(cache.dir), "inertia": float(km.inertia_)},
                                                       indent=2) + "\n")
    return c


def assign_codes(feats: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    f = torch.from_numpy(np.asarray(feats, np.float32))
    c = torch.from_numpy(centroids)
    return torch.cdist(f, c).argmin(1).numpy().astype(np.int16)


def cache_codes(src: FeatureCache, ids: list[str], centroids: np.ndarray, cache_root: Path, name: str) -> FeatureCache:
    spec = FeatureSpec(name=name, extractor_version="1", source_model=src.spec.source_model,
                       source_revision=src.spec.source_revision, sample_rate=16000, hop=320, dim=None, dtype="int16",
                       config={"from": src.spec.key(), "k": len(centroids),
                               "centroids_sha": hashlib.sha256(centroids.tobytes()).hexdigest()[:16]})
    out = FeatureCache(cache_root, spec)
    out.fill(((i, i) for i in ids), lambda i: assign_codes(src.get(i), centroids))
    return out


def write_prepared(out_dir: Path, info: dict[str, Any]) -> Path:
    p = out_dir / "prepared.json"
    p.write_text(json.dumps({**info, "time": time.time()}, indent=2, default=str) + "\n")
    return p


__all__ = ["assign_codes", "cache_codes", "cache_teacher_features", "ensure_manifest", "learn_kmeans", "teacher_spec",
           "write_prepared"]
