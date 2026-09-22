from __future__ import annotations

from ..models import is_ready
from ..presets import Mode, get_mode
from .placeholder import MissingBackend, PassthroughBackend


def create_backend(
    name: str | None = None,
    *,
    mode: str | Mode = "balanced",
    device: str = "cuda",
    allow_passthrough: bool = False,
):
    if isinstance(mode, str):
        mode_obj = get_mode(mode)
    else:
        mode_obj = mode
    backend = name or ("meanvc2" if is_ready("meanvc2") else None)
    if backend is None:
        if allow_passthrough:
            return PassthroughBackend()
        return MissingBackend()
    if backend == "passthrough":
        return PassthroughBackend()
    if backend == "meanvc2":
        if not is_ready("meanvc2"):
            return MissingBackend()
        from .meanvc2 import MeanVC2Backend

        return MeanVC2Backend(mode_obj, device=device)
    raise KeyError(f"unknown backend {backend!r}")
