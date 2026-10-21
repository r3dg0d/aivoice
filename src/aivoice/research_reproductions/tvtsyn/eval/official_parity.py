"""Parity of aivoice's TVTSyn modules with the official VOSSA code (user's own checkout, CC BY-NC 4.0;
imported only to compare parameter counts/shapes, never copied or redistributed)."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

from ..config import TVTSynConfig


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def check(checkout: str | Path, cfg: TVTSynConfig | None = None) -> dict[str, Any]:
    cfg = cfg or TVTSynConfig()
    root = Path(checkout)
    if not (root / "src" / "components" / "tv_timbre.py").exists():
        raise FileNotFoundError(f"{checkout}: not a VOSSA checkout")
    # load the two files directly under a private package name so VOSSA's package __init__ (which pulls in
    # Lightning) never runs; their only intra-package import is the sibling `streaming` module
    pkg = "_vossa_official"
    saved = {k: v for k, v in sys.modules.items() if k == pkg or k.startswith(pkg + ".")}
    try:
        for name, path in ((pkg, None), (f"{pkg}.components", root / "src" / "components")):
            m = types.ModuleType(name)
            m.__path__ = [str(path)] if path else []
            sys.modules[name] = m
        comp = root / "src" / "components"
        _load(f"{pkg}.components.streaming", comp / "streaming.py")
        tvt = _load(f"{pkg}.components.tv_timbre", comp / "tv_timbre.py")
        fvq = _load(f"{pkg}.components.fvq", comp / "quantizer" / "factorized_vector_quantize.py")
        StreamingGlobalOnlyTVTimbre, FactorizedVectorQuantize = tvt.StreamingGlobalOnlyTVTimbre, fvq.FactorizedVectorQuantize
    finally:
        for k in [k for k in sys.modules if k == pkg or k.startswith(pkg + ".")]:
            sys.modules.pop(k, None)
        sys.modules.update(saved)
    from ..model.encoder import FactorizedVQ
    from ..model.timbre import TimeVaryingTimbre

    t, c = cfg.timbre, cfg.content
    pairs = {
        "time_varying_timbre": (StreamingGlobalOnlyTVTimbre(c.dim, t.g_dim, t.d_kv, t.d_out, t.num_tokens, t.heads, 0.0,
                                                            t.gate_hidden, t.token_dropout), TimeVaryingTimbre(t, c.dim)),
        "factorized_vq": (FactorizedVectorQuantize(c.dim, c.codebook_size, c.codebook_dim, c.commitment,
                                                   c.codebook_loss_weight, c.l2_normalize),
                          FactorizedVQ(c.dim, c.codebook_size, c.codebook_dim, c.commitment, c.codebook_loss_weight)),
    }
    out: dict[str, Any] = {}
    for k, (a, b) in pairs.items():
        sa = sorted(tuple(p.shape) for p in a.parameters())
        sb = sorted(tuple(p.shape) for p in b.parameters())
        out[k] = {"official_params": sum(p.numel() for p in a.parameters()),
                  "aivoice_params": sum(p.numel() for p in b.parameters()), "shapes_equal": sa == sb}
    out["passed"] = all(v["shapes_equal"] for v in out.values())
    return out
