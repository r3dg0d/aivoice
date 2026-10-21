"""Training-only speaker anonymisation (SA) of the content-encoder input.

Paper: an off-the-shelf SA module perturbs the source during training so the decoder cannot rely on
timbre leaking through the content features; it is discarded at inference. The paper does not name
the SA system in the material available to aivoice, so two interchangeable options exist:

* ``mcadams``: McAdams-coefficient LPC pole warping (VoicePrivacy Challenge 2020 baseline B2,
  Patino et al. 2021), pure DSP, no weights. A random coefficient per utterance in ``alpha_range``.
* ``external``: anonymised copies produced by any SA system you choose (e.g. StreamVoiceAnon), listed
  in the manifest as ``anonymized_path`` (same length as the original).

This module must never be imported by the inference path (tests/reproductions enforce it).
"""

from __future__ import annotations

import numpy as np
import scipy.signal as sps

TRAINING_ONLY = True


def mcadams(x: np.ndarray, sr: int, alpha: float, order: int = 20, win_ms: float = 20.0) -> np.ndarray:
    import librosa

    n = int(sr * win_ms / 1000)
    hop = n // 2
    win = np.hanning(n + 1)[:-1]
    x = np.asarray(x, np.float64)
    pad = np.concatenate([np.zeros(n), x, np.zeros(n)])
    out = np.zeros_like(pad)
    norm = np.zeros_like(pad)
    for s in range(0, len(pad) - n, hop):
        frame = pad[s:s + n] * win
        if not np.any(frame):
            continue
        a = librosa.lpc(frame + 1e-9 * np.random.default_rng(s).standard_normal(n), order=order)
        res = sps.lfilter(a, [1.0], frame)
        poles = np.roots(a)
        cplx = np.iscomplex(poles) & (np.imag(poles) > 0)
        ang = np.angle(poles[cplx])
        new = np.abs(poles[cplx]) * np.exp(1j * np.sign(ang) * np.abs(ang) ** alpha)
        p2 = np.concatenate([poles[~np.iscomplex(poles)], new, np.conj(new)])
        a2 = np.real(np.poly(p2))
        y = sps.lfilter([1.0], a2, res)
        out[s:s + n] += y * win
        norm[s:s + n] += win ** 2
    y = out / np.maximum(norm, 1e-8)
    y = y[n:n + len(x)]
    peak = np.max(np.abs(y)) + 1e-9
    return (y * min(1.0, np.max(np.abs(x)) / peak)).astype(np.float32)


class Anonymizer:
    def __init__(self, kind: str = "mcadams", alpha_range: tuple[float, float] = (0.7, 0.9), order: int = 20,
                 sr: int = 16000) -> None:
        if kind not in ("mcadams", "external"):
            raise ValueError(f"anonymizer kind must be mcadams or external, got {kind!r}")
        self.kind, self.alpha_range, self.order, self.sr = kind, alpha_range, order, sr

    def __call__(self, x: np.ndarray, rng: np.random.Generator, anonymized: np.ndarray | None = None) -> np.ndarray:
        if self.kind == "external":
            if anonymized is None:
                raise ValueError("anonymizer=external but the manifest row has no anonymized_path")
            return anonymized
        return mcadams(x, self.sr, float(rng.uniform(*self.alpha_range)), self.order)
