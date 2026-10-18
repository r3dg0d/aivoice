"""Persistent user settings (``$XDG_CONFIG_HOME/aivoice/settings.json``).

Plain JSON, written atomically. Secrets are refused here on purpose: API keys and
tokens live in the environment, the system keyring, or Hugging Face's own
credential store (see :mod:`aivoice.credentials`), never in this file.
"""

from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path
from typing import Any

from .paths import config_home

# Whole key names (or *_suffix) that hold credentials; "max_tokens" is not one.
SECRET_KEY = re.compile(r"(^|[_-])(token|api[_-]?key|apikey|secret|password|passwd|bearer)$", re.I)

DEFAULTS: dict[str, Any] = {
    "experimental": {
        "enabled": False,
        "preset": "balanced",
    },
    "asr": {
        # auto | moonshine-tiny | moonshine-small | moonshine-medium | parakeet
        "backend": "auto",
        "policy": "balanced",  # fastest | accurate | balanced | low-memory | manual
        "language": "en",
        "latency_budget_ms": 500,
        "device": "cpu",
        "allow_model_download": False,
    },
    "stabilizer": {
        "agreement": 2,
        "min_commit_words": 3,
        "commit_on_punctuation": True,
        "speculative": True,  # speak stable phrases before the endpoint (matches the balanced preset)
    },
    "vad": {
        "engine": "auto",  # auto | silero | energy
        "threshold": 0.5,
        "min_silence_ms": 300,
        "speech_pad_ms": 120,
    },
    "llm": {
        "enabled": False,
        "model": None,  # local GGUF path or registered repo id
        "mode": "verbatim",  # verbatim | cleanup | natural | expressive | character | custom | hybrid
        "prompt": "",
        "character": "",
        "intensity": 0.5,
        "creativity": 0.3,
        "timeout_ms": 350,
        "max_tokens": 96,
        "n_threads": None,
        "n_gpu_layers": 0,
    },
    "expression": {
        "intensity": 0.5,
        "frequency": 0.5,
        "expression_only": False,
    },
    "tts": {
        # fish-server (local or remote fish-speech/Drama 3 server) | fish-api | espeak | command | piper | auto
        "backend": "auto",
        "server_url": "http://127.0.0.1:8080",
        "server_model": "auto",  # drama-3 | s2-pro | auto (ask the server)
        "api_model": "drama-3",
        "latency": "balanced",
        "fallback": "local",  # local (espeak/piper/command) | none
        "chunk_length": 200,
        "temperature": 0.7,
        "top_p": 0.7,
        "timeout_s": 30,
    },
    "voice": {"active": None},
    "audio": {
        "input_device": None,
        "output_device": None,
        "virtual_mic_name": "aivoice",
        "capture_rate": 16000,
        "block_ms": 20,
        "prebuffer_ms": 60,
        "input_mode": "vad",  # vad | ptt
        # queue: new speech waits for earlier output (voice-changer default);
        # interrupt: new speech cancels pending/playing output; duck: lower it while you talk
        "barge_in": "queue",
        "max_backlog_ms": 4000,  # drop unstarted output older than this to stay real-time
        "duck_db": -12.0,
        "output_driver": "auto",  # auto | sounddevice | pacat
        "output_gain_db": 0.0,
        "monitor": False,
        "echo_gate": True,
    },
    "privacy": {
        # Remote processing is opt-in per destination: {"fish-api": true, "http://gpu-box:8080": true}
        "remote_consent": {},
    },
    "defaults": {"asr": None, "tts": None, "llm": None, "vad": None},
    "hf": {
        "cache_dir": None,  # None = Hugging Face default cache
        "quota_gib": None,
        "confirm_over_gib": 2.0,
    },
    "presets": {},
}


class SettingsError(ValueError):
    pass


def settings_path() -> Path:
    return config_home() / "settings.json"


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _check_no_secrets(data: Any, path: str = "") -> None:
    if isinstance(data, dict):
        for k, v in data.items():
            here = f"{path}.{k}" if path else str(k)
            # remote_consent keys are URLs/provider names, values are booleans.
            if SECRET_KEY.search(str(k)) and not isinstance(v, (bool, type(None))):
                raise SettingsError(
                    f"refusing to store {here!r} in settings.json: secrets belong in the "
                    "environment or the system keyring (see `aivoice credentials`)"
                )
            _check_no_secrets(v, here)


def load() -> dict[str, Any]:
    p = settings_path()
    data: dict[str, Any] = {}
    if p.is_file():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}
    return _merge(DEFAULTS, data if isinstance(data, dict) else {})


def save(data: dict[str, Any]) -> Path:
    _check_no_secrets(data)
    p = settings_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, p)
    return p


def get(key: str, default: Any = None) -> Any:
    cur: Any = load()
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def set_value(key: str, value: Any) -> dict[str, Any]:
    if SECRET_KEY.search(key.split(".")[-1]) and not isinstance(value, (bool, type(None))):
        raise SettingsError(
            f"refusing to store secret-like key {key!r} in settings.json: use the environment or the "
            "system keyring (`aivoice credentials set fish-audio`)"
        )
    data = load()
    cur = data
    parts = key.split(".")
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value
    save(data)
    return data


def update(section: str, **values: Any) -> dict[str, Any]:
    data = load()
    data.setdefault(section, {}).update(values)
    save(data)
    return data


def parse_value(text: str) -> Any:
    """CLI helper: ``true``/``false``/``null``/numbers/JSON, else the raw string."""
    low = text.strip().lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if low in ("null", "none"):
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


# --- remote-processing consent -------------------------------------------------


def has_remote_consent(destination: str) -> bool:
    return bool((get("privacy.remote_consent") or {}).get(destination))


def grant_remote_consent(destination: str, granted: bool = True) -> None:
    data = load()
    consent = data.setdefault("privacy", {}).setdefault("remote_consent", {})
    if granted:
        consent[destination] = True
    else:
        consent.pop(destination, None)
    save(data)


def is_local_url(url: str) -> bool:
    from urllib.parse import urlparse

    host = (urlparse(url).hostname or "").lower()
    return host in ("localhost", "127.0.0.1", "::1") or host.startswith("127.")
