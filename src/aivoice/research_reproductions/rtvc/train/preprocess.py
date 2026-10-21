"""RT-VC preparation: manifest, pitch/periodicity cache (YIN at 200 Hz), EMA pseudo-label cache (SPARC).

SPARC (Speech Articulatory Coding, Cho et al. 2024; pip ``speech-articulatory-coding``) is the
articulatory-inversion teacher assumed for the EMA pseudo-labels (R-T1). It is loaded only if installed;
its checkpoints must already be on disk (``--teacher sparc=DIR``). Its 50 Hz EMA is upsampled to 200 Hz.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ...common.data import Filters, load_audio, read_manifest
from ...common.dsp import f0_yin
from ...common.features import FeatureCache, FeatureSpec
from ...common.prepare import ensure_manifest, write_prepared
from ..config import FRAME, SAMPLE_RATE, RTVCConfig


def pitch_spec() -> FeatureSpec:
    return FeatureSpec("rtvc-yin-f0-periodicity", "1", None, None, SAMPLE_RATE, FRAME, 2,
                       config={"fmin": 50, "fmax": 800, "frame": 640})


def ema_spec(revision: str | None) -> FeatureSpec:
    return FeatureSpec("rtvc-sparc-ema12-200hz", "1", "sparc", revision, SAMPLE_RATE, FRAME, 12,
                       config={"upsample": "linear 50->200 Hz"})


class SparcTeacher:
    def __init__(self, directory: str | None, device: str = "cpu") -> None:
        try:
            from sparc import load_model
        except ImportError as e:
            raise RuntimeError("SPARC is not installed: pip install speech-articulatory-coding "
                               "(the EMA pseudo-label teacher; training only)") from e
        self.model = load_model("en", device=device, ckpt=directory) if directory else load_model("en", device=device)
        self.revision = str(directory)

    def __call__(self, wav: np.ndarray) -> np.ndarray:
        code = self.model.encode(wav)
        ema = np.asarray(code["ema"], np.float32)  # [t@50Hz, 12]
        n = len(ema) * 4
        xs = np.arange(n) / 4.0
        return np.stack([np.interp(xs, np.arange(len(ema)), ema[:, c]) for c in range(ema.shape[1])], 1).astype(np.float32)


def prepare(cfg: RTVCConfig, dataset: str, out: str | Path, teacher_dirs: dict[str, str] | None = None,
            device: str = "cpu", limit: int | None = None, features: bool = True) -> dict[str, Any]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    man, stats = ensure_manifest(dataset, out, Filters(min_duration=1.0, max_duration=30.0), limit)
    info: dict[str, Any] = {"engine": "rtvc", "manifest": str(man), "manifest_stats": stats, "caches": {}}
    if features:
        rows = read_manifest(man)
        ac = out / "audio-cache"
        pc = FeatureCache(out / "features", pitch_spec())
        info["caches"]["pitch"] = {"cache": str(pc.dir), **pc.fill(
            ((r["id"], r) for r in rows),
            lambda r: np.stack(f0_yin(load_audio(r["path"], SAMPLE_RATE, ac), SAMPLE_RATE, FRAME), 1))}
        if not cfg.test_standin:
            try:
                sparc = SparcTeacher((teacher_dirs or {}).get("sparc"), device)
                ec = FeatureCache(out / "features", ema_spec(sparc.revision))
                info["caches"]["ema"] = {"cache": str(ec.dir), **ec.fill(
                    ((r["id"], r) for r in rows), lambda r: sparc(load_audio(r["path"], SAMPLE_RATE, ac)))}
            except RuntimeError as e:
                info["caches"]["ema"] = {"error": str(e)}
    write_prepared(out, info)
    return info
