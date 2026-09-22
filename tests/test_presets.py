import pytest

from aivoice.presets import MODES, get_mode


def test_modes():
    assert set(MODES) == {"lowest-latency", "balanced", "best-quality"}
    assert get_mode("balanced").sample_rate == 16000


def test_unknown():
    with pytest.raises(KeyError):
        get_mode("ultra")
