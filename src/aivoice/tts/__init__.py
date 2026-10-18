"""Speech synthesis backends."""

from __future__ import annotations

from typing import Any, Iterator

from .base import AudioChunk, RemoteNotAllowed, TTSBackend, TTSCapabilities, TTSError

TTS_BACKENDS = ("auto", "fish-server", "fish-api", "local", "espeak", "piper", "command")


class FallbackTTS(TTSBackend):
    """Primary backend with a *local* fallback when it fails before producing audio.

    The fallback is never a remote service: falling back from a local server to
    the cloud would send speech off the machine without the user choosing it.
    """

    def __init__(self, primary: TTSBackend, fallback: TTSBackend | None) -> None:
        self.primary = primary
        self.fallback = fallback
        self.name = primary.name
        self.fallbacks = 0
        self.last_error: str | None = None
        if fallback is not None and fallback.capabilities().remote and not primary.capabilities().remote:
            raise TTSError("refusing a remote fallback for a local primary backend")

    def capabilities(self) -> TTSCapabilities:
        return self.primary.capabilities()

    def health(self) -> tuple[bool, str]:
        return self.primary.health()

    def synthesize(self, text, *, voice=None, cancel=None) -> Iterator[AudioChunk]:
        produced = False
        try:
            for ch in self.primary.synthesize(text, voice=voice, cancel=cancel):
                produced = True
                yield ch
            return
        except TTSError as e:
            self.last_error = str(e)
            if produced or self.fallback is None:
                raise
        self.fallbacks += 1
        yield from self.fallback.synthesize(text, voice=voice, cancel=cancel)


def create_tts(cfg: dict[str, Any] | None = None) -> TTSBackend:
    """Build the configured backend. Remote destinations need prior consent (see settings.privacy)."""
    from .. import settings as st

    cfg = cfg or st.load()["tts"]
    name = cfg.get("backend", "auto")
    fallback: TTSBackend | None = None
    if cfg.get("fallback", "local") == "local" and name not in ("local", "espeak", "piper", "command"):
        try:
            from .local import LocalEngineTTS

            fallback = LocalEngineTTS()
        except TTSError:
            fallback = None

    if name in ("auto", "fish-server"):
        from .fish import FishServerBackend

        url = cfg.get("server_url", "http://127.0.0.1:8080")
        primary = FishServerBackend(
            url,
            model=cfg.get("server_model", "auto"),
            chunk_length=int(cfg.get("chunk_length", 200)),
            temperature=float(cfg.get("temperature", 0.7)),
            top_p=float(cfg.get("top_p", 0.7)),
            latency=cfg.get("latency", "balanced"),
            timeout=float(cfg.get("timeout_s", 30)),
        )
        if name == "auto":
            ok, _ = primary.health()
            if not ok:
                if fallback is None:
                    raise TTSError(
                        f"no TTS available: fish-speech server at {url} is not answering and no local engine "
                        "is installed (espeak-ng / Piper)."
                    )
                return fallback
        return FallbackTTS(primary, fallback)
    if name == "fish-api":
        from .fish import FishAPIBackend

        primary = FishAPIBackend(
            model=cfg.get("api_model", "drama-3"),
            latency=cfg.get("latency", "balanced"),
            chunk_length=int(cfg.get("chunk_length", 200)),
            temperature=float(cfg.get("temperature", 0.7)),
            top_p=float(cfg.get("top_p", 0.7)),
            timeout=float(cfg.get("timeout_s", 30)),
        )
        return primary  # no automatic local fallback mid-conversation either way: report errors
    if name in ("local", "espeak", "piper", "command"):
        from .local import LocalEngineTTS

        return LocalEngineTTS(None if name == "local" else ("espeak-ng" if name == "espeak" else name))
    raise TTSError(f"unknown tts backend {name!r}; choose from {', '.join(TTS_BACKENDS)}")


__all__ = [
    "AudioChunk",
    "FallbackTTS",
    "RemoteNotAllowed",
    "TTSBackend",
    "TTSCapabilities",
    "TTSError",
    "create_tts",
]
