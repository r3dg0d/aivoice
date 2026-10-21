"""Sample buffering between arbitrary audio blocks and the model's 20 ms frames.

The network layers accept any number of samples (they keep their own sub-frame buffers), so the only
buffering needed here is float32/torch conversion; this module exists so every reproduction has the
same place for it and so block-size invariance is tested in one spot."""

from __future__ import annotations

import numpy as np
import torch


def to_tensor(block: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(block, dtype=np.float32))[None].to(device)


def to_numpy(y: torch.Tensor) -> np.ndarray:
    return y.reshape(-1).float().cpu().numpy()
