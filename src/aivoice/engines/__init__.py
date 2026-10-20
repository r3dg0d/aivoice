"""Selectable synthesis and conversion engines behind capability-checked interfaces."""

from .base import (
    AudioResult,
    ConversionOptions,
    EngineCapabilities,
    EngineError,
    EngineUnavailable,
    SpeakerRef,
    SynthesisOptions,
    UnsupportedOption,
    validate_options,
)
from .registry import all_status, get_engine, names, status

__all__ = [
    "AudioResult",
    "ConversionOptions",
    "EngineCapabilities",
    "EngineError",
    "EngineUnavailable",
    "SpeakerRef",
    "SynthesisOptions",
    "UnsupportedOption",
    "all_status",
    "get_engine",
    "names",
    "status",
    "validate_options",
]
