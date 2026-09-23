"""Provider registry."""

from __future__ import annotations

from .base import VoiceProvider
from .voice_models import VoiceModelsProvider

_PROVIDERS: dict[str, VoiceProvider] = {
    "voice-models": VoiceModelsProvider(),
}


def list_providers() -> dict[str, VoiceProvider]:
    return dict(_PROVIDERS)


def get_provider(name: str = "voice-models") -> VoiceProvider:
    if name not in _PROVIDERS:
        raise KeyError(f"unknown provider {name!r}; known: {', '.join(sorted(_PROVIDERS))}")
    return _PROVIDERS[name]
