"""Modulation presets: attribute deltas at full strength, scaled by a 0..1 strength.

A preset says *which* attributes move and how far at strength 1.0; ``strength`` scales the
deltas (semitones and dB linearly, ratios in the log domain), so 0 is always the unmodified
voice. User presets are stored as plain JSON in ``$XDG_CONFIG_HOME/aivoice/modulation_presets.json``.

The masculine/feminine deltas are grounded in measured sex differences in adult speech
(F0 roughly 100-130 Hz vs 190-220 Hz, i.e. ~9 semitones; formants ~15-20 % lower in men,
following a longer vocal tract; e.g. Peterson & Barney 1952, Hillenbrand et al. 1995,
Fitch & Giedd 1999), but deliberately use only a *fraction* of them: the goal is the same
speaker with altered attributes, not a different person. How far a given voice can move before
it stops sounding like its owner is measured, not assumed (experiments L/M/N).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

from ..paths import config_home
from .dsp import ModulationParams

IDENTITY = ModulationParams()


@dataclass(frozen=True)
class Preset:
    name: str
    description: str
    delta: dict[str, Any]  # parameter -> value at strength 1.0
    default_strength: float = 1.0
    category: str = "identity-preserving"  # identity-preserving | character | utility
    window_ms: float = 32.0
    notes: str = ""
    builtin: bool = True
    fixed: dict[str, Any] = field(default_factory=dict)  # applied regardless of strength

    def params(self, strength: float | None = None, overrides: dict[str, Any] | None = None) -> ModulationParams:
        s = self.default_strength if strength is None else float(strength)
        if not 0.0 <= s <= 1.5:
            raise ValueError("strength must be between 0 and 1.5 (1.0 = the preset's full delta)")
        base = IDENTITY.to_dict()
        for k, v in self.delta.items():
            if k not in base:
                raise ValueError(f"preset {self.name}: unknown parameter {k!r}")
            if isinstance(v, bool):
                base[k] = v
            elif k in ("formant_ratio", "intonation"):
                # ratios scale in the log domain; intonation 0 (monotone) is reached linearly
                base[k] = math.exp(s * math.log(v)) if v > 0 else max(0.0, 1.0 - s)
            elif k in ("band_low_hz", "band_high_hz"):
                base[k] = v if s > 0 else 0.0
            else:
                base[k] = base[k] + s * v if not isinstance(base[k], bool) else v
        base.update(self.fixed)
        if overrides:
            for k, v in overrides.items():
                if k not in base:
                    raise ValueError(f"unknown modulation parameter {k!r}")
                base[k] = v
        return ModulationParams.from_dict(base)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "delta": self.delta,
                "default_strength": self.default_strength, "category": self.category,
                "window_ms": self.window_ms, "notes": self.notes, "builtin": self.builtin, "fixed": self.fixed}


# Experiment M (docs/research/experiments/M.md) kept the formant warp (about 11x more masculinity per unit of
# identity lost than a larger pitch drop) and rejected body resonance, narrower intonation and darker brightness
# as defaults: none moved the measured masculinity proxy. They remain available as manual controls
# (--body, --intonation, --brightness) and in MASCULINE_CANDIDATES for re-testing on real voices.
MASCULINE_DELTA = {
    "pitch_semitones": -4.0,  # ~45 % of the typical female-male F0 gap
    "formant_ratio": 0.90,  # ~10 % lower formants: ~1.7 cm longer apparent vocal tract
}
MASCULINE_CANDIDATES = {"body_db": 3.0, "intonation": 0.85, "brightness_db": -1.5}
FEMININE_DELTA = {
    "pitch_semitones": 4.0,
    "formant_ratio": 1.10,
}

BUILTIN: dict[str, Preset] = {p.name: p for p in [
    Preset("natural", "No change (cleanup only): the reference for A/B", {}, 1.0, "utility"),
    Preset("masculine", "Masculine shift of your own voice: lower pitch AND formants (longer vocal tract)", MASCULINE_DELTA, 0.65),
    Preset("masculine-subtle", "Masculine at 35 %", MASCULINE_DELTA, 0.35),
    Preset("masculine-natural", "Masculine at 65 %", MASCULINE_DELTA, 0.65),
    Preset("masculine-strong", "Masculine at 100 %", MASCULINE_DELTA, 1.0),
    Preset("feminine", "Feminine shift of your own voice: higher pitch AND formants (shorter vocal tract)", FEMININE_DELTA, 0.65),
    Preset("feminine-subtle", "Feminine at 35 %", FEMININE_DELTA, 0.35),
    Preset("feminine-natural", "Feminine at 65 %", FEMININE_DELTA, 0.65),
    Preset("deep", "Lower pitch and slightly lower formants, more body", {"pitch_semitones": -3.0, "formant_ratio": 0.94, "body_db": 4.0}, 0.6),
    Preset("bright", "More presence and air", {"brightness_db": 4.0, "tilt_db_per_oct": 1.0}, 0.7),
    Preset("dark", "Less presence, steeper spectral tilt", {"brightness_db": -4.0, "tilt_db_per_oct": -1.5}, 0.7),
    Preset("warm", "Fuller lows and mids, softer highs", {"body_db": 2.5, "resonance_db": 2.0, "brightness_db": -2.0}, 0.7),
    Preset("thin", "Less body", {"body_db": -6.0, "resonance_db": -2.0}, 0.7),
    Preset("full", "More body and oral resonance", {"body_db": 3.5, "resonance_db": 2.5}, 0.7),
    Preset("rough", "Slow amplitude modulation on voiced sounds", {"roughness": 0.7}, 0.6),
    Preset("breathy", "Aspiration noise under the spectral envelope", {"breathiness": 0.6}, 0.6),
    Preset("broadcast", "Radio-presenter chain: body, compression, de-essing, gentle saturation",
           {"body_db": 2.0, "compress": 0.7, "deess": 0.5, "saturation": 0.15, "brightness_db": 1.0}, 0.8, "character"),
    Preset("radio", "Band-limited AM-radio character", {"compress": 0.6, "saturation": 0.35},
           1.0, "character", fixed={"band_low_hz": 250.0, "band_high_hz": 5000.0}),
    Preset("telephone", "300-3400 Hz narrowband line", {"saturation": 0.25},
           1.0, "character", fixed={"band_low_hz": 300.0, "band_high_hz": 3400.0}),
    Preset("robotic", "Monotone, phase-flattened synthetic voice", {"robotic": 0.85, "intonation": 0.0}, 1.0, "character"),
    Preset("anonymous", "Light disguise (pitch, formants, flatter intonation). NOT anonymisation: "
           "speaker-verification systems can still link a DSP-shifted voice to its owner",
           {"pitch_semitones": -2.0, "formant_ratio": 1.08, "intonation": 0.7}, 1.0, "character"),
    Preset("character", "Exaggerated cartoon-like shift", {"pitch_semitones": 7.0, "formant_ratio": 1.18}, 1.0, "character"),
    Preset("low-latency", "No change with a 16 ms analysis window (16 ms algorithmic delay)", {}, 1.0, "utility", window_ms=16.0),
]}

ALIASES = {"masculine-natural": "masculine-natural", "male": "masculine", "female": "feminine"}


def _user_file():
    return config_home() / "modulation_presets.json"


def user_presets() -> dict[str, Preset]:
    try:
        raw = json.loads(_user_file().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    out = {}
    for name, d in raw.items():
        out[name] = Preset(name=name, description=d.get("description", "user preset"), delta=d.get("delta", {}),
                           default_strength=float(d.get("default_strength", 1.0)), category=d.get("category", "user"),
                           window_ms=float(d.get("window_ms", 32.0)), notes=d.get("notes", ""), builtin=False,
                           fixed=d.get("fixed", {}))
    return out


def all_presets() -> dict[str, Preset]:
    return {**BUILTIN, **user_presets()}


def get_preset(name: str) -> Preset:
    key = ALIASES.get(name.lower(), name.lower())
    p = all_presets().get(key) or all_presets().get(name)
    if p is None:
        raise KeyError(f"unknown modulation preset {name!r}; see `aivoice modulate preset list`")
    return p


def save_user_preset(name: str, params: ModulationParams, *, description: str = "", window_ms: float = 32.0) -> Preset:
    if name in BUILTIN:
        raise ValueError(f"{name!r} is a built-in preset; choose another name")
    ident = IDENTITY.to_dict()
    delta = {k: v for k, v in params.to_dict().items() if v != ident[k]}
    p = _user_file()
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raw = {}
    raw[name] = {"description": description or "user preset", "delta": {}, "fixed": delta, "default_strength": 1.0,
                 "window_ms": window_ms, "category": "user"}
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    return user_presets()[name]


def delete_user_preset(name: str) -> bool:
    p = _user_file()
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if name not in raw:
        return False
    del raw[name]
    p.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    return True


# CLI flag -> parameter, with the unit conversion each flag uses
ADVANCED_FLAGS = {
    "pitch": ("pitch_semitones", "semitones"),
    "formants": ("formant_ratio", "percent change, e.g. -10 lowers formants 10 %"),
    "resonance": ("resonance_db", "dB at ~700 Hz"),
    "body": ("body_db", "dB at ~180 Hz"),
    "brightness": ("brightness_db", "dB high shelf from ~3.5 kHz"),
    "roughness": ("roughness", "0..1"),
    "breathiness": ("breathiness", "0..1"),
    "intonation": ("intonation", "pitch-range scale, 1 = unchanged"),
}


def overrides_from_flags(**flags: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for flag, value in flags.items():
        if value is None:
            continue
        key, _ = ADVANCED_FLAGS[flag]
        if flag == "formants":
            value = 1.0 + float(value) / 100.0
            if not 0.5 <= value <= 1.6:
                raise ValueError("--formants must be between -50 and +60 percent")
        if flag in ("roughness", "breathiness") and not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"--{flag} must be between 0 and 1")
        out[key] = float(value)
    return out


__all__ = ["ADVANCED_FLAGS", "BUILTIN", "Preset", "all_presets", "delete_user_preset", "get_preset",
           "overrides_from_flags", "save_user_preset", "user_presets"]
