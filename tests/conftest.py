import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))


@pytest.fixture(autouse=True)
def _isolated_xdg(tmp_path, monkeypatch):
    """Every test gets its own config/data/cache/runtime dirs; nothing touches the real home."""
    for var, sub in (
        ("XDG_CONFIG_HOME", "config"),
        ("XDG_DATA_HOME", "data"),
        ("XDG_CACHE_HOME", "cache"),
        ("XDG_RUNTIME_DIR", "run"),
        ("HF_HOME", "hf"),
    ):
        d = tmp_path / sub
        d.mkdir(exist_ok=True)
        monkeypatch.setenv(var, str(d))
    for var in ("FISH_API_KEY", "FISH_AUDIO_API_KEY", "HF_HUB_OFFLINE", "AIVOICE_SILERO_ONNX"):
        monkeypatch.delenv(var, raising=False)
    # Pulse/PipeWire clients in tests must not reach a developer's real sound server.
    if os.environ.get("AIVOICE_TEST_REAL_PULSE") != "1":
        monkeypatch.setenv("PULSE_SERVER", str(tmp_path / "no-pulse"))
    yield
