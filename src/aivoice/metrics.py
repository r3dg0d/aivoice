"""Audio metrics: e2e latency, RTF, underruns, drops, CUDA util."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class AudioMetrics:
    e2e_latency_ms: float = 0.0
    rtf: float = 0.0
    sample_rate: int = 16000
    buffer_depth: int = 0
    underruns: int = 0
    dropped_chunks: int = 0
    cuda_util: float | None = None
    chunks: int = 0


@dataclass
class MetricsTracker:
    sample_rate: int = 16000
    underruns: int = 0
    dropped_chunks: int = 0
    buffer_depth: int = 0
    _proc_ms: list[float] = field(default_factory=list)
    _audio_ms: list[float] = field(default_factory=list)
    chunks: int = 0

    def mark_underrun(self, n: int = 1) -> None:
        self.underruns += n

    def mark_drop(self, n: int = 1) -> None:
        self.dropped_chunks += n

    def record(self, *, process_ms: float, audio_ms: float, buffer_depth: int = 0) -> AudioMetrics:
        self._proc_ms.append(process_ms)
        self._audio_ms.append(audio_ms)
        self.buffer_depth = buffer_depth
        self.chunks += 1
        if len(self._proc_ms) > 100:
            self._proc_ms.pop(0)
            self._audio_ms.pop(0)
        avg_proc = sum(self._proc_ms) / len(self._proc_ms)
        avg_audio = sum(self._audio_ms) / len(self._audio_ms) if self._audio_ms else 1.0
        rtf = avg_proc / avg_audio if avg_audio > 0 else 0.0
        return AudioMetrics(
            e2e_latency_ms=avg_proc,
            rtf=rtf,
            sample_rate=self.sample_rate,
            buffer_depth=buffer_depth,
            underruns=self.underruns,
            dropped_chunks=self.dropped_chunks,
            cuda_util=_cuda_util(),
            chunks=self.chunks,
        )


def _cuda_util() -> float | None:
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            # rough: allocated / reserved
            a = torch.cuda.memory_allocated()
            r = torch.cuda.memory_reserved() or 1
            return float(a) / float(r)
    except Exception:
        pass
    return None


def format_metrics(m: AudioMetrics) -> str:
    cu = f"{m.cuda_util:.2f}" if m.cuda_util is not None else "n/a"
    return (
        f"e2e={m.e2e_latency_ms:.1f}ms rtf={m.rtf:.3f} sr={m.sample_rate} "
        f"buf={m.buffer_depth} underrun={m.underruns} drop={m.dropped_chunks} cuda={cu}"
    )
