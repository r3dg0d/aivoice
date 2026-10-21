"""Import the official StreamVoiceAnon model classes from your checkout (Apache-2.0; not vendored)."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def repo_dir(explicit: str | None = None) -> Path:
    d = explicit or os.environ.get("AIVOICE_STREAMVOICEANON_REPO_DIR")
    if not d:
        from aivoice.vc.engines import vendor

        d = str(vendor("streamvoiceanon"))
    p = Path(d)
    if not (p / "modules" / "dual_ar_stream.py").exists():
        raise FileNotFoundError(f"{p}: not a StreamVoiceAnon checkout (set AIVOICE_STREAMVOICEANON_REPO_DIR)")
    return p


def import_modules(explicit: str | None = None):
    p = repo_dir(explicit)
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
    from modules import arvc_wrapper, dual_ar_stream  # type: ignore

    return arvc_wrapper, dual_ar_stream


def tiny_official(dims: dict, explicit: str | None = None):
    """A tiny *official-architecture* ARVC (random init) for tests; never a usable model."""
    import torch

    aw, da = import_modules(explicit)
    cfg = da.DualARModelArgs(model_type="dual_ar", vocab_size=dims["vocab"], n_layer=dims["n_layer"], n_head=dims["n_head"],
                             dim=dims["dim"], head_dim=dims["head_dim"], intermediate_size=dims["dim"] * 3,
                             max_seq_len=512, codebook_size=dims["codebook"], num_codebooks=dims["num_codebooks"],
                             n_fast_layer=dims["n_fast_layer"], fast_dim=dims["dim"], fast_head_dim=dims["head_dim"],
                             fast_n_head=dims["n_head"], fast_intermediate_size=dims["dim"] * 3,
                             tie_word_embeddings=False, delay=[0, 1, 2])
    dec = da.DualARWrapper(da.DualARTransformer(cfg))
    return aw.ARVCWrapper(torch.nn.Embedding(dims["vocab"], dims["dim"]), dec, context_dim=16, style_dim=24,
                          model_dim=dims["dim"])


def load_base(cfg_path: str, ckpt_path: str, explicit: str | None = None, trust_pickle: bool = False):
    """Build the official ARVC from its hydra config and load the public base checkpoint."""
    import torch
    from hydra.utils import instantiate  # official dependency
    from omegaconf import OmegaConf

    p = repo_dir(explicit)
    import_modules(explicit)
    conf = OmegaConf.load(str(p / cfg_path))
    model = instantiate(OmegaConf.load(str(p / conf.model_params.config_path)) if "model_params" in conf else conf.model)
    sd = torch.load(str(p / ckpt_path), map_location="cpu", weights_only=not trust_pickle)
    sd = sd.get("net", sd.get("model", sd))
    model.load_state_dict(sd, strict=False)
    return model
