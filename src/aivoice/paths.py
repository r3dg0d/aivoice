"""XDG paths for profiles, config, and model/vendor cache."""

from __future__ import annotations

import os
from pathlib import Path


def data_home() -> Path:
    base = os.environ.get("XDG_DATA_HOME")
    if base:
        return Path(base) / "aivoice"
    return Path.home() / ".local" / "share" / "aivoice"


def cache_home() -> Path:
    base = os.environ.get("XDG_CACHE_HOME")
    if base:
        return Path(base) / "aivoice"
    return Path.home() / ".cache" / "aivoice"


def config_home() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME")
    if base:
        return Path(base) / "aivoice"
    return Path.home() / ".config" / "aivoice"


def models_dir() -> Path:
    return cache_home() / "models"


def vendor_dir() -> Path:
    """Optional MeanVC2 git checkout after models install."""
    return cache_home() / "vendor" / "MeanVC2"


def profiles_dir() -> Path:
    return data_home() / "profiles"


def ensure_dirs() -> None:
    data_home().mkdir(parents=True, exist_ok=True)
    profiles_dir().mkdir(parents=True, exist_ok=True)
    models_dir().mkdir(parents=True, exist_ok=True)
    config_home().mkdir(parents=True, exist_ok=True)
    (cache_home() / "vendor").mkdir(parents=True, exist_ok=True)
