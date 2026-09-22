import os

from aivoice.devices import detect_audio_stack, format_devices, list_devices


def test_mock_devices(monkeypatch):
    monkeypatch.setenv("AIVOICE_MOCK_DEVICES", "1")
    devices = list_devices()
    assert len(devices) >= 2
    text = format_devices(devices)
    assert "mock" in text.lower() or "Mock" in text


def test_detect_stack():
    stack = detect_audio_stack()
    assert stack in {"pipewire", "pulse", "alsa", "none"}
