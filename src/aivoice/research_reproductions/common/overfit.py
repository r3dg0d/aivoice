"""Mandatory tiny-overfit gate (validation phase B) and a rights-free synthetic corpus for it.

Before any large run, the exact recipe must drive its training loss on 1-4 utterances well below its
starting value. ``reproduce train`` refuses to start a run whose config fingerprint has no passing
``overfit.json`` in the output directory, and runs the gate itself first.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .checkpoint import config_fingerprint
from .trainer import TrainConfig, Trainer

SENTENCES = [
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "Please call Stella and ask her to bring these things with her from the store.",
    "A rainbow is a division of white light into many beautiful colors.",
    "Six spoons of fresh snow peas, five thick slabs of blue cheese.",
]
VOICES = ["en-us", "en-us+f3", "en-gb", "en-us+m3"]


def synthetic_corpus(out_dir: str | Path, n: int = 4, sr: int = 16000, speakers: int = 2) -> Path:
    """espeak-ng utterances (or a formant test signal if espeak-ng is missing) + a manifest."""
    from scipy.signal import resample_poly

    from .harness import test_signal

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    espeak = shutil.which("espeak-ng")
    for i in range(n):
        spk = i % speakers
        p = out / f"spk{spk}_{i:02d}.wav"
        if espeak:
            with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
                subprocess.run([espeak, "-v", VOICES[spk % len(VOICES)], "-s", "150", "-w", tmp.name,
                                SENTENCES[i % len(SENTENCES)]], check=True, capture_output=True)
                x, file_sr = sf.read(tmp.name, dtype="float32")
            if file_sr != sr:
                x = resample_poly(x, sr, file_sr).astype(np.float32)
        else:
            x = test_signal(sr, 3.0, seed=i)
        x = (0.5 * x / (np.max(np.abs(x)) + 1e-9)).astype(np.float32)
        sf.write(p, x, sr)
        rows.append({"id": f"synthetic/{p.stem}", "path": str(p.resolve()), "speaker": f"synthetic/spk{spk}",
                     "dataset": "synthetic-espeak" if espeak else "synthetic-formant", "duration": len(x) / sr,
                     "sr": sr, "text": SENTENCES[i % len(SENTENCES)], "split": "train"})
    man = out / "manifest.jsonl"
    man.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return man


def run_overfit(make_recipe: Callable[[], Any], out_dir: str | Path, steps: int = 300, batch_size: int = 2,
                loss_key: str | None = None, threshold: float = 0.5, device: str = "auto",
                precision: str = "fp32", lr_override: float | None = None) -> dict[str, Any]:
    """Train ``steps`` on the recipe's (tiny) dataset; pass if final loss <= threshold * initial loss."""
    out = Path(out_dir)
    recipe = make_recipe()
    tc = TrainConfig(output=str(out / "run"), device=device, precision=precision, batch_size=batch_size, workers=0,
                     max_steps=steps, checkpoint_every=0, eval_every=0, log_every=10, keep_checkpoints=1)
    res = Trainer(recipe, tc).fit()
    rows = [json.loads(line) for line in (out / "run" / "metrics.jsonl").read_text().splitlines()]
    key = loss_key or getattr(recipe, "overfit_loss_key", None) or next(k for k in rows[0] if k not in ("step", "time") and not k.startswith(("perf/", "lr/", "grad")))
    first = float(np.mean([r[key] for r in rows[:2] if key in r]))
    last = float(np.mean([r[key] for r in rows[-3:] if key in r]))
    passed = bool(np.isfinite(last) and last <= threshold * first)
    rep = {"phase": "B (tiny overfit)", "passed": passed, "loss_key": key, "initial": first, "final": last,
           "ratio": last / first if first else None, "threshold": threshold, "steps": steps,
           "config_fingerprint": config_fingerprint(recipe.model_config()), "engine": recipe.engine,
           "step_times": res.get("step_times"), "curve": [(r["step"], r.get(key)) for r in rows]}
    (out / "overfit.json").write_text(json.dumps(rep, indent=2) + "\n")
    return rep


def overfit_passed(out_dir: str | Path, config: dict[str, Any]) -> bool:
    p = Path(out_dir) / "overfit.json"
    if not p.exists():
        return False
    rep = json.loads(p.read_text())
    return bool(rep.get("passed")) and rep.get("config_fingerprint") == config_fingerprint(config)


__all__ = ["overfit_passed", "run_overfit", "synthetic_corpus"]
