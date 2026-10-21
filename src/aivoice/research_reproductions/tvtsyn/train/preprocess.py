"""TVTSyn preparation: manifest; HuBERT-base L9 features -> k-means 200 -> code cache; x-vector + ECAPA cache.

The speechbrain speaker encoders (spkrec-xvect-voxceleb 512-d, spkrec-ecapa-voxceleb 192-d) are frozen
teachers loaded from local snapshots (``--teacher xvector=DIR --teacher ecapa=DIR``); speechbrain must be
installed. Nothing is downloaded by aivoice.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from ...common.data import Filters, load_audio, read_manifest
from ...common.features import FeatureCache, FeatureSpec
from ...common.prepare import (
    cache_codes,
    cache_teacher_features,
    ensure_manifest,
    learn_kmeans,
    teacher_spec,
    write_prepared,
)
from ...common.teachers import load_teacher, teacher_dir
from ..config import SAMPLE_RATE, TVTSynConfig


class SpeakerTeachers:
    def __init__(self, xvec_dir: str | None, ecapa_dir: str | None, device: str = "cpu") -> None:
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError as e:
            raise RuntimeError("speechbrain is not installed (x-vector / ECAPA teachers): pip install speechbrain") from e
        self.x = EncoderClassifier.from_hparams(source=str(teacher_dir("xvector", xvec_dir)), run_opts={"device": device})
        self.e = EncoderClassifier.from_hparams(source=str(teacher_dir("ecapa", ecapa_dir)), run_opts={"device": device})

    @torch.no_grad()
    def __call__(self, wav: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(wav)[None]
        return np.concatenate([self.x.encode_batch(t).reshape(-1).cpu().numpy(),
                               self.e.encode_batch(t).reshape(-1).cpu().numpy()]).astype(np.float32)


def prepare(cfg: TVTSynConfig, dataset: str, out: str | Path, teacher_dirs: dict[str, str] | None = None,
            device: str = "cpu", limit: int | None = None, features: bool = True) -> dict[str, Any]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    td = teacher_dirs or {}
    man, stats = ensure_manifest(dataset, out, Filters(min_duration=1.0, max_duration=30.0), limit)
    info: dict[str, Any] = {"engine": "tvtsyn-repro", "manifest": str(man), "manifest_stats": stats, "caches": {}}
    if features and not cfg.test_standin:
        cr, ac = out / "features", out / "audio-cache"
        ids = [r["id"] for r in read_manifest(man)]
        hub = load_teacher("hubert-base", cfg.content.km_layer, 768, False, td.get("hubert-base"))
        name = f"hubert-base-l{cfg.content.km_layer}"
        info["caches"]["hubert"] = cache_teacher_features(hub, man, cr, name, device, ac)
        src = FeatureCache(cr, teacher_spec(hub, name), create=False)
        cent = learn_kmeans(src, ids, cfg.content.num_codes, out=out / f"kmeans{cfg.content.num_codes}.npy")
        codes = cache_codes(src, ids, cent, cr, f"{name}-km{cfg.content.num_codes}")
        info["caches"]["codes"] = {"cache": str(codes.dir)}
        try:
            spk = SpeakerTeachers(td.get("xvector"), td.get("ecapa"), device)
            sc = FeatureCache(cr, FeatureSpec("xvector512-ecapa192", "1", "speechbrain xvect+ecapa", None, SAMPLE_RATE,
                                              0, None, config={"dim": 704}))
            rows = read_manifest(man)
            info["caches"]["speaker"] = {"cache": str(sc.dir), **sc.fill(
                ((r["id"], r) for r in rows), lambda r: spk(load_audio(r["path"], SAMPLE_RATE, ac)))}
        except RuntimeError as e:
            info["caches"]["speaker"] = {"error": str(e)}
    write_prepared(out, info)
    return info
