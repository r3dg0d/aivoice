"""Adaptive ASR backend selection from *measured* benchmarks.

Results are cached per hardware/runtime fingerprint plus backend configuration;
a driver, runtime or GPU change gives a new fingerprint, so old numbers are
never reused on a different setup. With no measurement the selector says so and
falls back to a named default instead of guessing which model is faster.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..hardware import fingerprint
from ..paths import benchmarks_dir

POLICIES = ("fastest", "accurate", "balanced", "low-memory", "manual")
UNMEASURED_DEFAULT = "moonshine-small"
MAX_AGE_S = 30 * 24 * 3600


@dataclass
class Measurement:
    backend: str
    final_p50_ms: float
    final_p95_ms: float | None
    rtf: float | None
    wer: float | None
    rss_mib: float | None
    clips: int
    measured_at: float
    config: dict[str, Any]


@dataclass
class Selection:
    backend: str
    policy: str
    measured: bool
    reason: str


def _cache_file() -> Path:
    return benchmarks_dir() / "asr_selection_cache.json"


def _key(backend: str, config: dict[str, Any]) -> str:
    return fingerprint(extra={"backend": backend, **{k: config[k] for k in sorted(config)}})


class ASRSelector:
    def __init__(self, cache_path: Path | None = None) -> None:
        self.path = cache_path or _cache_file()

    def _load(self) -> dict[str, Any]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def record(self, m: Measurement) -> None:
        data = self._load()
        data[_key(m.backend, m.config)] = asdict(m)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    def measurements(self, candidates: dict[str, dict[str, Any]]) -> dict[str, Measurement]:
        data = self._load()
        out = {}
        now = time.time()
        for name, cfg in candidates.items():
            raw = data.get(_key(name, cfg))
            if raw and now - raw.get("measured_at", 0) < MAX_AGE_S:
                out[name] = Measurement(**raw)
        return out

    def select(
        self,
        policy: str,
        candidates: dict[str, dict[str, Any]],
        *,
        latency_budget_ms: float = 500,
        manual: str | None = None,
    ) -> Selection:
        policy = policy.lower()
        if policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}")
        if policy == "manual":
            if not manual:
                raise ValueError("manual policy needs a backend")
            return Selection(manual, policy, False, "chosen by the user")
        ms = self.measurements(candidates)
        if not ms:
            fb = UNMEASURED_DEFAULT if UNMEASURED_DEFAULT in candidates else next(iter(candidates), UNMEASURED_DEFAULT)
            return Selection(
                fb,
                policy,
                False,
                "no benchmark results for this machine yet; using an unmeasured default. "
                "Run `aivoice experimental asr benchmark` to enable automatic selection.",
            )
        items = list(ms.values())
        if policy == "fastest":
            best = min(items, key=lambda m: m.final_p50_ms)
            why = f"lowest measured final latency ({best.final_p50_ms:.0f} ms p50)"
        elif policy == "low-memory":
            best = min(items, key=lambda m: m.rss_mib if m.rss_mib is not None else float("inf"))
            why = f"smallest measured memory ({best.rss_mib or 0:.0f} MiB RSS)"
        elif policy == "accurate":
            with_wer = [m for m in items if m.wer is not None]
            if not with_wer:
                best = min(items, key=lambda m: m.final_p50_ms)
                why = "no WER measured (benchmark had no reference transcripts); fell back to fastest"
            else:
                inside = [m for m in with_wer if m.final_p50_ms <= latency_budget_ms]
                pool = inside or with_wer
                best = min(pool, key=lambda m: m.wer)  # type: ignore[arg-type, return-value]
                why = f"lowest WER ({best.wer:.1%})" + (
                    f" within {latency_budget_ms:.0f} ms" if inside else f"; nothing met the {latency_budget_ms:.0f} ms budget"
                )
        else:  # balanced
            def score(m: Measurement) -> float:
                lat = m.final_p50_ms / max(latency_budget_ms, 1)
                return (m.wer if m.wer is not None else 0.0) * 4 + lat

            best = min(items, key=score)
            why = f"best WER/latency trade-off ({best.final_p50_ms:.0f} ms p50" + (
                f", WER {best.wer:.1%})" if best.wer is not None else ", WER not measured)"
            )
        missing = [c for c in candidates if c not in ms]
        if missing:
            why += f"; not yet measured: {', '.join(missing)}"
        return Selection(best.backend, policy, True, why)
