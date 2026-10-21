"""Architecture parity against the official RT-VC model classes (user's own checkout; never vendored).

Builds the official ``Vocoder``, ``InverterCausalConv`` and ``RTPitchTracker`` with the same sizes as the
aivoice config and compares parameter shapes module by module. torchaudio is stubbed only for import
(the official files import it at module level; no torchaudio code runs during construction).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import torch

from ..config import RTVCConfig


def _stubs() -> dict[str, types.ModuleType]:
    """Import-time stand-ins for torchaudio / transformers (the official files import them at module level;
    nothing from them runs while the classes are constructed)."""
    ta = types.ModuleType("torchaudio")
    ta.transforms = types.ModuleType("torchaudio.transforms")
    ta.functional = types.ModuleType("torchaudio.functional")
    for n in ("MFCC", "MelSpectrogram", "Spectrogram"):
        setattr(ta.transforms, n, object)
    ta.functional.fftconvolve = None
    tf = types.ModuleType("transformers")
    tf.WavLMModel = object
    return {"torchaudio": ta, "torchaudio.transforms": ta.transforms, "torchaudio.functional": ta.functional,
            "transformers": tf}


def _shapes(m: torch.nn.Module) -> list[tuple[int, ...]]:
    return sorted(tuple(p.shape) for p in m.parameters())


def check(checkout: str | Path, cfg: RTVCConfig | None = None) -> dict[str, Any]:
    cfg = cfg or RTVCConfig()
    src = Path(checkout) / "src"
    if not (src / "vocoder" / "vocoder.py").exists():
        raise FileNotFoundError(f"{checkout}: not an RT-VC checkout (src/vocoder/vocoder.py missing)")
    stubs = _stubs()
    saved = {k: sys.modules.get(k) for k in list(stubs) + ["vocoder", "ema_inverter", "pitch_tracker"]}
    sys.modules.update(stubs)
    sys.path.insert(0, str(src))
    try:
        from ema_inverter.model import InverterCausalConv
        from pitch_tracker.model import RTPitchTracker
        from vocoder.vocoder import Vocoder
    finally:
        sys.path.remove(str(src))
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        for k in [k for k in sys.modules if k.split(".")[0] in ("vocoder", "ema_inverter", "pitch_tracker")]:
            sys.modules.pop(k, None)
    from ..model.decoder import DDSPVocoder
    from ..model.encoder import EMAInverter, PitchTracker

    v = cfg.vocoder
    off_voc = Vocoder(v.hidden_dim, v.nharmonics, v.nbands, v.attenuate, 16000, 80, cfg.speaker.spkr_emb_dim,
                      v.kernel_size, v.n_blocks, v.forward_expansion, v.reverb_len)
    ours_voc = DDSPVocoder(v, cfg.speaker.spkr_emb_dim, cfg.ema.channels)
    off_inv = InverterCausalConv(cfg.frontend.n_mfcc, cfg.ema.hidden_dim, cfg.ema.channels, cfg.ema.kernel_size)
    ours_inv = EMAInverter(cfg.ema, cfg.frontend.n_mfcc)
    p = cfg.pitch
    pcfg = {"convnet_config": {"in_channels": cfg.frontend.n_mels, "hidden_dim": p.hidden_dim, "out_channels": p.hidden_dim,
                               "kernel_size": p.kernel_size},
            "hidden_dim": p.hidden_dim, "nbins": p.nbins, "forward_expansion": p.forward_expansion,
            "n_midlayers": p.n_midlayers, "use_periodicity_head": True, "use_loudness_head": True,
            "fmin": p.fmin, "fmax": p.fmax, "f_base": p.f_base}
    off_pt = RTPitchTracker(pcfg)
    ours_pt = PitchTracker(p, cfg.frontend.n_mels)
    out = {}
    for name, a, b in (("vocoder", off_voc, ours_voc), ("ema_inverter", off_inv, ours_inv), ("pitch_tracker", off_pt, ours_pt)):
        sa, sb = _shapes(a), _shapes(b)
        out[name] = {"official_params": sum(x.numel() for x in a.parameters()),
                     "aivoice_params": sum(x.numel() for x in b.parameters()), "shapes_equal": sa == sb}
    out["passed"] = all(v["shapes_equal"] for v in out.values() if isinstance(v, dict))
    return out
