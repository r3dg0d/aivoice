"""StreamVoiceAnon+ preparation (run with the StreamVoiceAnon interpreter, CUDA recommended).

For each manifest row: semantic codes (official ASR-BSQ content encoder), acoustic codes (firefly codec),
style vector (CampPlus) and timbre latents (Spark speaker encoder) via the official ``InferenceWrapper``
helpers, plus frame-level emotion2vec features mean-pooled to codec frames. Each goes into a versioned
feature cache. Not runnable in the aivoice build environment (no CUDA, no checkpoints)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ...common.data import Filters, read_manifest
from ...common.features import FeatureCache, FeatureSpec
from ...common.prepare import ensure_manifest, write_prepared
from ..config import CODEC_HOP, CODEC_SR, PlusConfig


def pool_to_codec(frames: np.ndarray, src_rate: float, n_codec: int) -> np.ndarray:
    """Mean-pool teacher frames (src_rate Hz) onto codec frames (44100/2048 Hz)."""
    t = (np.arange(len(frames)) + 0.5) / src_rate
    idx = np.minimum((t * CODEC_SR / CODEC_HOP).astype(int), n_codec - 1)
    out = np.zeros((n_codec, frames.shape[1]), np.float32)
    cnt = np.zeros(n_codec)
    np.add.at(out, idx, frames)
    np.add.at(cnt, idx, 1)
    nz = cnt > 0
    out[nz] /= cnt[nz, None]
    for i in np.nonzero(~nz)[0]:  # frames with no teacher frame: copy the previous one
        out[i] = out[i - 1] if i else 0
    return out


def prepare(cfg: PlusConfig, dataset: str, out: str | Path, teacher_dirs: dict[str, str] | None = None,
            device: str = "cuda", limit: int | None = None, features: bool = True, repo_dir: str | None = None) -> dict[str, Any]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    man, stats = ensure_manifest(dataset, out, Filters(min_duration=1.0, max_duration=20.0, min_sr=16000), limit)
    info: dict[str, Any] = {"engine": "streamvoiceanon-plus", "manifest": str(man), "manifest_stats": stats, "caches": {}}
    rows = read_manifest(man)
    info["emotion_labels"] = sorted({str(r.get("emotion")) for r in rows})
    if features:
        import librosa

        from ..model.official import import_modules
        from ..model.official import repo_dir as _repo

        import_modules(repo_dir)
        import sys

        sys.path.insert(0, str(_repo(repo_dir) / "evaluations"))
        from funasr import AutoModel
        from infer_arvc import InferenceWrapper  # type: ignore

        w = InferenceWrapper(cfg.base_config, cfg.base_checkpoint)
        emo = AutoModel(model=(teacher_dirs or {}).get("emotion2vec", "emotion2vec/emotion2vec_plus_base"))
        cr = out / "features"
        specs = {k: FeatureCache(cr, FeatureSpec(f"svanon-{k}", "1", "Plachta/StreamVoiceAnon", cfg.base_checkpoint,
                                                 CODEC_SR, CODEC_HOP, None, "float32" if k in ("style", "timbre") else "int32"))
                 for k in ("semantic", "acoustic", "style", "timbre")}
        specs["emotion2vec"] = FeatureCache(cr, FeatureSpec("emotion2vec-codec-frames", "1", "emotion2vec_plus_base", None,
                                                            CODEC_SR, CODEC_HOP, cfg.emo_dim))
        for r in rows:
            wav, _ = librosa.load(r["path"], sr=CODEC_SR)
            enc = w.encode_for_training(wav)  # requires the official helper; see SPEC P-I1
            for k in ("semantic", "acoustic", "style", "timbre"):
                specs[k].put(r["id"], enc[k])
            e = emo.generate(librosa.resample(wav, orig_sr=CODEC_SR, target_sr=16000), granularity="frame",
                             extract_embedding=True)[0]["feats"]
            specs["emotion2vec"].put(r["id"], pool_to_codec(np.asarray(e, np.float32), cfg.emo_frame_rate,
                                                            len(enc["semantic"])))
        info["caches"] = {k: str(v.dir) for k, v in specs.items()}
    write_prepared(out, info)
    return info
