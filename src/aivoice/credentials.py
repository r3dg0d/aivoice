"""Credential lookup that never touches plaintext aivoice config.

* Fish Audio API key: ``FISH_API_KEY`` (or ``FISH_AUDIO_API_KEY``) in the
  environment, else the system keyring (``keyring`` package; Secret Service /
  KWallet on Linux) under service ``aivoice`` / user ``fish-audio``.
* Hugging Face token: delegated entirely to ``huggingface_hub`` (``HF_TOKEN`` or
  ``hf auth login``). aivoice never reads, copies, or logs it beyond asking the
  library whether one exists.
"""

from __future__ import annotations

import os

KEYRING_SERVICE = "aivoice"
FISH_USER = "fish-audio"
FISH_ENV = ("FISH_API_KEY", "FISH_AUDIO_API_KEY")


class CredentialError(RuntimeError):
    pass


def _keyring():
    try:
        import keyring  # type: ignore
    except ImportError:
        return None
    return keyring


def fish_api_key() -> str | None:
    for name in FISH_ENV:
        val = os.environ.get(name, "").strip()
        if val:
            return val
    kr = _keyring()
    if kr is None:
        return None
    try:
        return kr.get_password(KEYRING_SERVICE, FISH_USER)
    except Exception:  # noqa: BLE001 - no backend / locked keyring
        return None


def store_fish_api_key(value: str) -> str:
    kr = _keyring()
    if kr is None:
        raise CredentialError(
            "The `keyring` package is not installed, so there is no secure store to write to. "
            "pip install keyring (uses Secret Service/KWallet), or export FISH_API_KEY in your "
            "session instead. aivoice will not write the key to a plaintext file."
        )
    try:
        kr.set_password(KEYRING_SERVICE, FISH_USER, value)
    except Exception as e:  # noqa: BLE001
        raise CredentialError(f"keyring refused the write: {e}") from e
    return "system keyring"


def delete_fish_api_key() -> None:
    kr = _keyring()
    if kr is None:
        return
    try:
        kr.delete_password(KEYRING_SERVICE, FISH_USER)
    except Exception:  # noqa: BLE001
        pass


def hf_token_present() -> bool:
    try:
        from huggingface_hub import get_token  # type: ignore
    except ImportError:
        return bool(os.environ.get("HF_TOKEN"))
    try:
        return bool(get_token())
    except Exception:  # noqa: BLE001
        return False


def redact(text: str, *secrets: str | None) -> str:
    """Remove known secret values (and bearer headers) from a message before logging it."""
    out = text
    for s in secrets:
        if s and len(s) >= 6:
            out = out.replace(s, "***")
    import re

    out = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._\-]+", r"\1***", out)
    out = re.sub(r"hf_[A-Za-z0-9]{10,}", "hf_***", out)
    return out


def describe() -> dict[str, str]:
    fish = "set (env)" if any(os.environ.get(n) for n in FISH_ENV) else (
        "set (keyring)" if fish_api_key() else "not set"
    )
    return {
        "fish_audio_api_key": fish,
        "huggingface_token": "present (huggingface_hub store/env)" if hf_token_present() else "not set",
        "keyring_backend": "available" if _keyring() else "not installed",
    }
