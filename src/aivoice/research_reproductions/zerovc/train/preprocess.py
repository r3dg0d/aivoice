"""Zero-VC preparation: manifest (LibriTTS etc., >= 4 s), optional teacher caches.

    stage content / variant A : w2v-BERT 2.0 hidden_states[16] cache
    stage content / variant B : HuBERT-base hidden_states[9] cache -> k-means 200 -> code cache
    stage vc                  : WavLM-large hidden_states[7] cache (timbre references)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...common.data import Filters, read_manifest
from ...common.features import FeatureCache
from ...common.prepare import (
    cache_codes,
    cache_teacher_features,
    ensure_manifest,
    learn_kmeans,
    teacher_spec,
    write_prepared,
)
from ...common.teachers import load_teacher
from ..config import ZeroVCConfig


def prepare(cfg: ZeroVCConfig, dataset: str, out: str | Path, teacher_dirs: dict[str, str] | None = None,
            device: str = "cpu", limit: int | None = None, features: bool = True) -> dict[str, Any]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    teacher_dirs = teacher_dirs or {}
    man, stats = ensure_manifest(dataset, out, Filters(min_duration=cfg.min_utt_s, max_duration=30.0), limit)
    info: dict[str, Any] = {"engine": "zerovc-repro", "manifest": str(man), "manifest_stats": stats, "caches": {}}
    if features:
        cr = out / "features"
        audio_cache = out / "audio-cache"
        ids = [r["id"] for r in read_manifest(man)]
        c = cfg.content
        if cfg.stage == "content" and c.variant == "A":
            t = load_teacher("w2v-bert-2.0", c.teacher_layer, c.teacher_dim, cfg.test_standin,
                             teacher_dirs.get("w2v-bert-2.0"))
            info["caches"]["teacher"] = cache_teacher_features(t, man, cr, f"w2vbert2-l{c.teacher_layer}", device,
                                                               audio_cache)
        elif cfg.stage == "content":
            t = load_teacher("hubert-base", c.km_layer, 768, cfg.test_standin, teacher_dirs.get("hubert-base"))
            r = cache_teacher_features(t, man, cr, f"hubert-base-l{c.km_layer}", device, audio_cache)
            src = FeatureCache(cr, teacher_spec(t, f"hubert-base-l{c.km_layer}"), create=False)
            cent = learn_kmeans(src, ids, c.km_codes, out=out / f"kmeans{c.km_codes}.npy")
            codes = cache_codes(src, ids, cent, cr, f"hubert-base-l{c.km_layer}-km{c.km_codes}")
            info["caches"]["hubert"] = r
            info["caches"]["teacher"] = {"cache": str(codes.dir)}
            info["kmeans"] = str(out / f"kmeans{c.km_codes}.npy")
        else:
            t = load_teacher("wavlm-large", cfg.timbre.ssl_layer, cfg.timbre.ssl_dim, cfg.test_standin,
                             teacher_dirs.get("wavlm-large"))
            info["caches"]["timbre"] = cache_teacher_features(t, man, cr, f"wavlm-large-l{cfg.timbre.ssl_layer}",
                                                              device, audio_cache)
    write_prepared(out, info)
    return info
