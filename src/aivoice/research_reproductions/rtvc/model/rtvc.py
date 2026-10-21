"""Full RT-VC inference graph: causal front-end -> pitch tracker + EMA inverter -> f0 shift -> DDSP vocoder.

The source f0 is scaled by ``target_median / source_median`` as in the official ``VC.vc``. The official
code takes the source median over the whole source file (offline); streaming has to fix it in advance:
``source_median`` comes from enrolling the user's own voice (or a running estimate, SPEC R-S1).
"""

from __future__ import annotations

import torch
from torch import nn

from ...common.streaming_nn import State
from ..config import FRAME, SAMPLE_RATE, RTVCConfig
from .decoder import DDSPVocoder
from .encoder import EMAInverter, PitchTracker
from .frontend import CausalMelMFCC


class RTVC(nn.Module):
    def __init__(self, cfg: RTVCConfig) -> None:
        super().__init__()
        self.cfg = cfg
        fc = cfg.frontend
        self.frontend = CausalMelMFCC(SAMPLE_RATE, fc.n_fft, FRAME, fc.n_mels, fc.n_mfcc)
        self.pitch = PitchTracker(cfg.pitch, fc.n_mels)
        self.ema = EMAInverter(cfg.ema, fc.n_mfcc)
        self.vocoder = DDSPVocoder(cfg.vocoder, cfg.speaker.spkr_emb_dim, cfg.ema.channels)

    def analyse(self, wav: torch.Tensor) -> dict[str, torch.Tensor]:
        logmel, mfcc = self.frontend(wav)
        p = self.pitch(logmel)
        return {"f0": self.pitch.decode(p["logits"]), "periodicity": p["periodicity"], "loudness": p["loudness"],
                "ema": self.ema(mfcc)}

    def forward(self, wav: torch.Tensor, g: torch.Tensor, f0_ratio: torch.Tensor | float = 1.0) -> torch.Tensor:
        a = self.analyse(wav)
        return self.vocoder(a["f0"] * f0_ratio, a["loudness"], a["periodicity"], a["ema"], g)

    def step(self, wav: torch.Tensor, g: torch.Tensor, f0_ratio, state: State | None) -> tuple[torch.Tensor, State]:
        st = state or {"fe": None, "pitch": None, "ema": None, "voc": None}
        (logmel, mfcc), s_fe = self.frontend.step(wav, st["fe"])
        if logmel.shape[-1] == 0:
            return wav.new_zeros(wav.shape[0], 0), {**st, "fe": s_fe}
        p, s_p = self.pitch.step(logmel, st["pitch"])
        e, s_e = self.ema.step(mfcc, st["ema"])
        f0 = self.pitch.decode(p["logits"]) * f0_ratio
        y, s_v = self.vocoder.step(f0, p["loudness"], p["periodicity"], e, g, st["voc"])
        return y, {"fe": s_fe, "pitch": s_p, "ema": s_e, "voc": s_v}
