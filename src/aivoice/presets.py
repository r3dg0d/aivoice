"""Latency / quality modes mapped to MeanVC2 chunk settings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ModeName = Literal["lowest-latency", "balanced", "best-quality"]


@dataclass(frozen=True)
class Mode:
    name: ModeName
    meanvc2_model: str  # 40ms | 120ms
    chunk_ms: int
    future_ms: int
    steps: int
    sample_rate: int
    buffer_chunks: int


MODES: dict[str, Mode] = {
    "lowest-latency": Mode(
        name="lowest-latency",
        meanvc2_model="40ms",
        chunk_ms=40,
        future_ms=40,
        steps=1,
        sample_rate=16000,
        buffer_chunks=2,
    ),
    "balanced": Mode(
        name="balanced",
        meanvc2_model="40ms",
        chunk_ms=40,
        future_ms=40,
        steps=2,
        sample_rate=16000,
        buffer_chunks=3,
    ),
    "best-quality": Mode(
        name="best-quality",
        meanvc2_model="120ms",
        chunk_ms=120,
        future_ms=40,
        steps=3,
        sample_rate=16000,
        buffer_chunks=4,
    ),
}


def get_mode(name: str) -> Mode:
    if name not in MODES:
        raise KeyError(f"unknown mode {name!r}; choose from {sorted(MODES)}")
    return MODES[name]
