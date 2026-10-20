"""Engine registry: names -> factories, capability table, availability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .base import EngineCapabilities, EngineError

_FACTORIES: dict[str, Callable[..., Any]] = {}


def register(name: str, factory: Callable[..., Any]) -> None:
    _FACTORIES[name] = factory


def _builtin() -> None:
    if _FACTORIES:
        return
    from .builtin_engines import EspeakEngine, FishEngine, MeanVC2Engine
    from .research_engines import VoiceStarEngine, VoxCPM2Engine, XVoiceEngine

    register("voxcpm2", VoxCPM2Engine)
    register("xvoice", XVoiceEngine)
    register("voicestar", VoiceStarEngine)
    register("meanvc2", MeanVC2Engine)
    register("fish-server", lambda **kw: FishEngine("server"))
    register("fish-api", lambda **kw: FishEngine("api"))
    register("espeak", EspeakEngine)


ALIASES = {"vox-cpm2": "voxcpm2", "voxcpm": "voxcpm2", "x-voice": "xvoice", "fish": "fish-server", "espeak-ng": "espeak"}


def names() -> list[str]:
    _builtin()
    return list(_FACTORIES)


def get_engine(name: str, **kw: Any) -> Any:
    _builtin()
    key = ALIASES.get(name.lower(), name.lower())
    if key not in _FACTORIES:
        raise EngineError(f"unknown engine {name!r}; available: {', '.join(_FACTORIES)}")
    return _FACTORIES[key](**kw)


@dataclass
class EngineStatus:
    name: str
    caps: EngineCapabilities
    unavailable: str | None

    @property
    def ready(self) -> bool:
        return self.unavailable is None


def status(name: str, *, check: bool = True, **kw: Any) -> EngineStatus:
    eng = get_engine(name, **kw)
    why: str | None = None
    if check:
        try:
            why = eng.available()
        except Exception as e:  # noqa: BLE001
            why = f"{type(e).__name__}: {e}"
    return EngineStatus(name, eng.capabilities(), why)


def all_status(*, check: bool = True) -> list[EngineStatus]:
    return [status(n, check=check) for n in names()]


FEATURES = [
    ("tts", lambda c: "tts" in c.tasks),
    ("vc", lambda c: "vc" in c.tasks),
    ("zero-shot", lambda c: c.zero_shot),
    ("no-transcript", lambda c: c.transcript_free_prompt),
    ("cross-lingual", lambda c: c.cross_lingual),
    ("style-prompt", lambda c: c.style_prompt),
    ("style-ref", lambda c: c.style_reference),
    ("duration", lambda c: c.duration_control),
    ("streaming", lambda c: c.streaming),
    ("adapt", lambda c: bool(c.adaptation)),
]


def capability_table(statuses: list[EngineStatus]) -> str:
    head = f"{'ENGINE':<12} " + " ".join(f"{f:<13}" for f, _ in FEATURES) + " RATE    READY"
    lines = [head]
    for s in statuses:
        c = s.caps
        cells = " ".join(f"{('yes' if fn(c) else '-'):<13}" for _, fn in FEATURES)
        rate = f"{c.native_sample_rate // 1000 if c.native_sample_rate else '?'}k"
        lines.append(f"{s.name:<12} {cells} {rate:<7} {'yes' if s.ready else 'no'}")
    return "\n".join(lines)
