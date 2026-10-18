"""Experimental-mode presets: named bundles of settings, plus user-saved ones."""

from __future__ import annotations

import copy
from typing import Any

from .. import settings as st

PRESETS: dict[str, dict[str, Any]] = {
    "ultra-low-latency": {
        "asr": {"backend": "moonshine-small", "policy": "manual"},
        "llm": {"enabled": False, "mode": "verbatim"},
        "stabilizer": {"agreement": 2, "min_commit_words": 2, "commit_on_punctuation": True, "speculative": True},
        "vad": {"min_silence_ms": 220, "speech_pad_ms": 100},
        "audio": {"prebuffer_ms": 40},
        "tts": {"latency": "balanced", "chunk_length": 100},
        "expression": {"intensity": 0.0},
    },
    "balanced": {
        "asr": {"backend": "auto", "policy": "balanced"},
        "llm": {"mode": "hybrid"},
        "stabilizer": {"agreement": 2, "min_commit_words": 3, "commit_on_punctuation": True, "speculative": True},
        "vad": {"min_silence_ms": 300, "speech_pad_ms": 120},
        "audio": {"prebuffer_ms": 60},
        "tts": {"latency": "balanced", "chunk_length": 200},
        "expression": {"intensity": 0.4},
    },
    "expressive": {
        "asr": {"backend": "moonshine-medium", "policy": "manual"},
        "llm": {"enabled": True, "mode": "expressive"},
        "stabilizer": {"agreement": 2, "min_commit_words": 4, "speculative": False},
        "vad": {"min_silence_ms": 350},
        "audio": {"prebuffer_ms": 80},
        "tts": {"latency": "normal", "chunk_length": 200},
        "expression": {"intensity": 0.7, "frequency": 0.6},
    },
    "character": {
        "llm": {"enabled": True, "mode": "character"},
        "stabilizer": {"speculative": False},
        "vad": {"min_silence_ms": 350},
        "audio": {"prebuffer_ms": 80},
        "expression": {"intensity": 0.6, "frequency": 0.5},
    },
    "accuracy": {
        "asr": {"backend": "auto", "policy": "accurate"},
        "stabilizer": {"agreement": 3, "speculative": False},
        "vad": {"min_silence_ms": 500, "speech_pad_ms": 200},
        "audio": {"prebuffer_ms": 100},
        "tts": {"latency": "normal"},
    },
}

PRESET_SECTIONS = ("asr", "llm", "stabilizer", "vad", "audio", "tts", "expression", "voice")


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def all_presets() -> dict[str, dict[str, Any]]:
    user = st.get("presets") or {}
    return {**PRESETS, **{k: v for k, v in user.items() if isinstance(v, dict)}}


def apply_preset(name: str) -> dict[str, Any]:
    presets = all_presets()
    if name not in presets:
        raise KeyError(f"unknown preset {name!r}; available: {', '.join(sorted(presets))}")
    data = _merge(st.load(), presets[name])
    data.setdefault("experimental", {})["preset"] = name
    st.save(data)
    return data


def save_preset(name: str) -> dict[str, Any]:
    if name in PRESETS:
        raise ValueError(f"{name!r} is a built-in preset; pick another name (e.g. custom-{name})")
    data = st.load()
    snap = {s: copy.deepcopy(data.get(s, {})) for s in PRESET_SECTIONS}
    data.setdefault("presets", {})[name] = snap
    data.setdefault("experimental", {})["preset"] = name
    st.save(data)
    return snap


def delete_preset(name: str) -> None:
    data = st.load()
    if name not in (data.get("presets") or {}):
        raise KeyError(f"no saved preset {name!r}")
    del data["presets"][name]
    st.save(data)
