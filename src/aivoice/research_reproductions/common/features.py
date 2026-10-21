"""Versioned precomputed-feature caches (SSL features, pseudo-labels, F0, loudness, speaker embeddings).

A cache directory holds ``meta.json`` and one ``.npy`` per utterance. ``meta.json`` records what made
the features (extractor name and version, source model id, its revision/commit or file hash, the
extractor config, sample rate, hop, feature dim and dtype) and a cache *key* derived from all of it.
Opening a cache with a different key is an error, not a silent reuse, so a changed teacher model or
layer can never mix with old features.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

FEATURE_CACHE_FORMAT = "aivoice-feature-cache/1"


class FeatureCacheError(RuntimeError):
    pass


@dataclass
class FeatureSpec:
    name: str                 # e.g. "wavlm-large-l7", "hubert-base-l9-km200", "f0-yin", "loudness-a"
    extractor_version: str    # bump when the extraction code changes
    source_model: str | None  # e.g. "microsoft/wavlm-large" (None for DSP features)
    source_revision: str | None  # commit hash / file sha256 of the model actually used
    sample_rate: int
    hop: int
    dim: int | None           # None for scalar-per-frame tracks
    dtype: str = "float32"
    config: dict[str, Any] = field(default_factory=dict)

    def key(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]


class FeatureCache:
    def __init__(self, root: str | Path, spec: FeatureSpec, create: bool = True) -> None:
        self.spec = spec
        self.dir = Path(root) / f"{spec.name}-{spec.key()}"
        meta = self.dir / "meta.json"
        if meta.exists():
            m = json.loads(meta.read_text())
            if m.get("key") != spec.key():
                raise FeatureCacheError(f"{self.dir}: cache key {m.get('key')} != {spec.key()} (spec changed)")
        elif create:
            self.dir.mkdir(parents=True, exist_ok=True)
            meta.write_text(json.dumps({"format": FEATURE_CACHE_FORMAT, "key": spec.key(), "spec": asdict(spec),
                                        "created": time.time()}, indent=2) + "\n")
        else:
            raise FeatureCacheError(f"no feature cache {self.dir}")

    @classmethod
    def open(cls, directory: str | Path) -> FeatureCache:
        """Open an existing cache directory (as recorded in prepared.json), verifying its key."""
        d = Path(directory)
        m = json.loads((d / "meta.json").read_text())
        spec = FeatureSpec(**m["spec"])
        c = cls(d.parent, spec, create=False)
        if c.dir != d:
            raise FeatureCacheError(f"{d}: directory name does not match its spec key")
        return c

    def _path(self, item_id: str) -> Path:
        h = hashlib.sha1(item_id.encode()).hexdigest()
        return self.dir / h[:2] / f"{h}.npy"

    def has(self, item_id: str) -> bool:
        return self._path(item_id).exists()

    def put(self, item_id: str, arr: np.ndarray) -> None:
        arr = np.asarray(arr, dtype=self.spec.dtype)
        if self.spec.dim is not None and (arr.ndim != 2 or arr.shape[1] != self.spec.dim):
            raise FeatureCacheError(f"{self.spec.name}: expected [frames, {self.spec.dim}], got {arr.shape}")
        p = self._path(item_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp.npy")
        np.save(tmp, arr)
        tmp.replace(p)

    def get(self, item_id: str, mmap: bool = True) -> np.ndarray:
        p = self._path(item_id)
        if not p.exists():
            raise FeatureCacheError(f"{self.spec.name}: no features for {item_id!r}; run `aivoice reproduce prepare`")
        return np.load(p, mmap_mode="r" if mmap else None)

    def fill(self, items: Iterable[tuple[str, Any]], fn: Callable[[Any], np.ndarray],
             progress: Callable[[int], None] | None = None) -> dict[str, int]:
        done = skipped = 0
        for i, (item_id, x) in enumerate(items):
            if self.has(item_id):
                skipped += 1
            else:
                self.put(item_id, fn(x))
                done += 1
            if progress:
                progress(i + 1)
        return {"computed": done, "cached": skipped}


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class SSLExtractor:
    """Hidden states of a local Hugging Face SSL model (WavLM / HuBERT) at one layer.

    The model must already be on disk (``model_dir``); aivoice does not download it. Revision is the
    sha256 of the weight file so the cache key changes if the weights do.
    """

    def __init__(self, model_dir: str | Path, layer: int, device: str = "cpu") -> None:
        import torch
        from transformers import AutoModel

        self.dir = Path(model_dir)
        weights = next((p for p in (self.dir / "model.safetensors", self.dir / "pytorch_model.bin") if p.exists()), None)
        if weights is None:
            raise FeatureCacheError(f"{model_dir}: no model.safetensors / pytorch_model.bin")
        self.revision = file_sha256(weights)
        self.model = AutoModel.from_pretrained(str(self.dir)).to(device).eval()
        self.layer, self.device, self.torch = layer, device, torch
        cfg = json.loads((self.dir / "config.json").read_text())
        self.source = cfg.get("_name_or_path") or self.dir.name
        self.dim = int(cfg["hidden_size"])

    def spec(self, name: str) -> FeatureSpec:
        return FeatureSpec(name=name, extractor_version="1", source_model=self.source, source_revision=self.revision,
                           sample_rate=16000, hop=320, dim=self.dim, config={"layer": self.layer})

    def __call__(self, audio_16k: np.ndarray) -> np.ndarray:
        torch = self.torch
        with torch.inference_mode():
            x = torch.from_numpy(np.asarray(audio_16k, np.float32))[None].to(self.device)
            out = self.model(x, output_hidden_states=True)
            return out.hidden_states[self.layer][0].float().cpu().numpy()


__all__ = ["FEATURE_CACHE_FORMAT", "FeatureCache", "FeatureCacheError", "FeatureSpec", "SSLExtractor", "file_sha256"]
