"""Fish Audio Drama 3 / S2-Pro backends.

Two transports, one request schema (fish-speech ``ServeTTSRequest``):

* :class:`FishServerBackend` - a fish-speech ``tools/api_server.py`` (or a
  compatible SGLang-Omni / vLLM-Omni deployment exposing ``/v1/tts``). On
  ``localhost`` this is the LOCAL mode; any other host is REMOTE and requires
  explicit consent, because text and reference audio leave the machine.
* :class:`FishAPIBackend` - the official hosted API at ``api.fish.audio``
  (``POST /v1/tts`` with a ``model`` header, as the official ``fish-audio-sdk``
  does). Always remote; needs consent and an API key from the environment or
  keyring.

Streaming: the server returns a WAV header followed by PCM as it is generated
(``streaming: true``); the hosted API streams the response body. Both are read
with ``read1`` so audio is handed on chunk by chunk.

Hardware note: Drama 3 is a 35B-parameter MoE that upstream lists at >=48 GB
of GPU memory in FP8; it does not fit a 24 GB RTX 4090. On such a card run
S2-Pro (4B) locally, or point this backend at a bigger GPU box / the hosted API.
"""

from __future__ import annotations

import base64
import json
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Iterator

from .. import __version__, settings
from ..credentials import fish_api_key, redact
from .base import AudioChunk, RemoteNotAllowed, TTSBackend, TTSCapabilities, TTSError
from .wavstream import WavStreamDecoder

HOSTED_URL = "https://api.fish.audio"
UA = f"aivoice/{__version__} (+https://github.com/r3dg0d/aivoice)"


def _packer():
    try:
        import ormsgpack  # type: ignore

        return "application/msgpack", ormsgpack.packb
    except ImportError:
        pass
    try:
        import msgpack  # type: ignore

        return "application/msgpack", lambda o: msgpack.packb(o, use_bin_type=True)
    except ImportError:
        pass

    def as_json(o: dict[str, Any]) -> bytes:
        def enc(v):
            if isinstance(v, (bytes, bytearray)):
                return base64.b64encode(v).decode()  # fish-speech base64-decodes long strings
            if isinstance(v, dict):
                return {k: enc(x) for k, x in v.items()}
            if isinstance(v, list):
                return [enc(x) for x in v]
            return v

        return json.dumps(enc(o)).encode()

    return "application/json", as_json


def build_request(
    text: str,
    *,
    voice: Any | None,
    streaming: bool = True,
    fmt: str = "wav",
    chunk_length: int = 200,
    temperature: float | None = 0.7,
    top_p: float | None = 0.7,
    latency: str = "balanced",
    hosted: bool = False,
    sample_rate: int | None = None,
) -> dict[str, Any]:
    req: dict[str, Any] = {
        "text": text,
        "format": fmt,
        "chunk_length": max(100, min(300 if hosted else 1000, int(chunk_length))),
        "normalize": True,
        "latency": latency if latency in ("normal", "balanced") else "balanced",
        "references": [],
        "reference_id": None,
    }
    if not hosted:
        req["streaming"] = streaming
        req["use_memory_cache"] = "on"  # server caches the encoded reference between requests
    if sample_rate and hosted:
        req["sample_rate"] = sample_rate
    if temperature is not None:
        req["temperature"] = float(temperature)
    if top_p is not None:
        req["top_p"] = float(top_p)
    if voice is not None:
        kind = getattr(voice, "kind", None)
        if kind == "hosted_fish_id":
            if not hosted:
                raise TTSError(
                    f"voice {voice.name!r} is a hosted Fish Audio id; it only works with tts.backend=fish-api. "
                    "Hosted ids cannot be used by a local Drama 3 server."
                )
            req["reference_id"] = voice.reference_id
        elif kind == "local_voice_id":
            if hosted:
                raise TTSError(f"voice {voice.name!r} is a local server reference id; it is unknown to the hosted API")
            req["reference_id"] = voice.reference_id
        elif kind == "local_voice":
            req["references"] = [{"audio": voice.reference_bytes(), "text": voice.reference_text or ""}]
    return {k: v for k, v in req.items() if v is not None}


def _stream_audio(
    req: urllib.request.Request,
    *,
    timeout: float,
    cancel: threading.Event | None,
    raw_pcm_rate: int | None = None,
    secrets: tuple[str | None, ...] = (),
) -> Iterator[AudioChunk]:
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read(600).decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            pass
        raise TTSError(redact(f"TTS HTTP {e.code}: {body.strip()[:400]}", *secrets)) from e
    except (urllib.error.URLError, OSError) as e:
        raise TTSError(redact(f"TTS endpoint unreachable: {e}", *secrets)) from e
    dec = WavStreamDecoder(raw_pcm_rate=raw_pcm_rate)
    with resp:
        while True:
            if cancel is not None and cancel.is_set():
                return
            data = resp.read1(16384) if hasattr(resp, "read1") else resp.read(16384)
            if not data:
                break
            pcm = dec.feed(data)
            if pcm.size:
                yield AudioChunk(pcm=pcm, sample_rate=int(dec.sample_rate or 44100))


class FishServerClient:
    """Small client for a fish-speech api_server (health, references)."""

    def __init__(self, url: str, *, api_key: str | None = None, timeout: float = 10) -> None:
        self.url = url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        h = {"User-Agent": UA}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _get(self, path: str) -> Any:
        req = urllib.request.Request(self.url + path, headers=self._headers())
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            raw = r.read()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw.decode("utf-8", "replace")

    def health(self) -> tuple[bool, str]:
        try:
            data = self._get("/v1/health")
        except Exception as e:  # noqa: BLE001
            return False, f"{self.url} unreachable: {e}"
        ok = isinstance(data, dict) and data.get("status") == "ok"
        return ok, json.dumps(data) if isinstance(data, dict) else str(data)[:200]

    def defaults(self) -> dict[str, Any]:
        try:
            d = self._get("/v1/tts/defaults")
            return d if isinstance(d, dict) else {}
        except Exception:  # noqa: BLE001
            return {}

    def detect_model(self) -> str | None:
        """'drama-3' / 's2-pro' when the server says so, else None (unknown)."""
        blob = json.dumps(self.defaults()).lower()
        if "drama" in blob or "single_chunk" in blob:
            return "drama-3"
        if "s2" in blob:
            return "s2-pro"
        return None

    def list_references(self) -> list[str]:
        data = self._get("/v1/references/list")
        return list(data.get("reference_ids", [])) if isinstance(data, dict) else []

    def add_reference(self, reference_id: str, audio: Path, text: str) -> dict[str, Any]:
        boundary = uuid.uuid4().hex
        audio = Path(audio)
        mime = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac", ".ogg": "audio/ogg"}.get(
            audio.suffix.lower(), "application/octet-stream"
        )
        parts = []
        for k, v in (("id", reference_id), ("text", text)):
            parts.append(
                f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
            )
        parts.append(
            (
                f'--{boundary}\r\nContent-Disposition: form-data; name="audio"; filename="{audio.name}"\r\n'
                f"Content-Type: {mime}\r\n\r\n"
            ).encode()
            + audio.read_bytes()
            + b"\r\n"
        )
        parts.append(f"--{boundary}--\r\n".encode())
        req = urllib.request.Request(
            self.url + "/v1/references/add",
            data=b"".join(parts),
            headers={**self._headers(), "Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=max(self.timeout, 60)) as r:
                data = json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            raise TTSError(f"server refused the reference: HTTP {e.code} {e.read(300)!r}") from e
        if isinstance(data, dict) and data.get("success") is False:
            raise TTSError(f"server refused the reference: {data.get('message')}")
        return data if isinstance(data, dict) else {}

    def delete_reference(self, reference_id: str) -> None:
        boundary = uuid.uuid4().hex
        body = (
            f'--{boundary}\r\nContent-Disposition: form-data; name="reference_id"\r\n\r\n{reference_id}\r\n--{boundary}--\r\n'
        ).encode()
        req = urllib.request.Request(
            self.url + "/v1/references/delete",
            data=body,
            headers={**self._headers(), "Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="DELETE",
        )
        with urllib.request.urlopen(req, timeout=self.timeout):
            pass


class FishServerBackend(TTSBackend):
    def __init__(
        self,
        url: str = "http://127.0.0.1:8080",
        *,
        model: str = "auto",
        api_key: str | None = None,
        chunk_length: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.7,
        latency: str = "balanced",
        timeout: float = 30.0,
        check_consent: bool = True,
    ) -> None:
        self.url = url.rstrip("/")
        self.remote = not settings.is_local_url(self.url)
        if self.remote and check_consent and not settings.has_remote_consent(self.url):
            raise RemoteNotAllowed(
                f"{self.url} is not on this machine: transcripts and reference audio would be sent there. "
                f"Allow it explicitly with `aivoice experimental remote allow {self.url}`."
            )
        self.client = FishServerClient(self.url, api_key=api_key)
        self._model = model
        self.chunk_length = chunk_length
        self.temperature = temperature
        self.top_p = top_p
        self.latency = latency
        self.timeout = timeout
        self.name = "fish-server" + ("(remote)" if self.remote else "")

    @property
    def model(self) -> str:
        if self._model == "auto":
            self._model = self.client.detect_model() or "unknown"
        return self._model

    def capabilities(self) -> TTSCapabilities:
        dialect = "drama3" if self.model == "drama-3" else "s2"
        return TTSCapabilities(
            name=self.name,
            streaming=True,
            dialect=dialect,
            remote=self.remote,
            destination=self.url,
            voice_kinds=("local_voice", "local_voice_id"),
            notes=f"fish-speech compatible server at {self.url} (model: {self.model})"
            + ("" if self._model != "unknown" else "; set tts.server_model to drama-3 to enable Drama 3 tags"),
        )

    def health(self) -> tuple[bool, str]:
        return self.client.health()

    def synthesize(self, text, *, voice=None, cancel=None) -> Iterator[AudioChunk]:
        ctype, pack = _packer()
        body = build_request(
            text,
            voice=voice,
            chunk_length=self.chunk_length,
            temperature=self.temperature,
            top_p=self.top_p,
            latency=self.latency,
        )
        req = urllib.request.Request(
            self.url + "/v1/tts",
            data=pack(body),
            headers={**self.client._headers(), "Content-Type": ctype},
            method="POST",
        )
        yield from _stream_audio(req, timeout=self.timeout, cancel=cancel, secrets=(self.client.api_key,))


class FishAPIBackend(TTSBackend):
    name = "fish-api"

    def __init__(
        self,
        *,
        model: str = "drama-3",
        api_key: str | None = None,
        base_url: str = HOSTED_URL,
        latency: str = "balanced",
        chunk_length: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.7,
        sample_rate: int = 44100,
        timeout: float = 30.0,
        check_consent: bool = True,
    ) -> None:
        if check_consent and not settings.has_remote_consent("fish-api"):
            raise RemoteNotAllowed(
                "The hosted Fish Audio API receives your transcribed text (and reference audio for local "
                "voices). Allow it explicitly with `aivoice experimental remote allow fish-api`."
            )
        self.api_key = api_key or fish_api_key()
        if not self.api_key:
            raise TTSError(
                "No Fish Audio API key: export FISH_API_KEY, or store it in the system keyring with "
                "`aivoice credentials set fish-audio`."
            )
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.latency = latency
        self.chunk_length = chunk_length
        self.temperature = temperature
        self.top_p = top_p
        self.sample_rate = sample_rate
        self.timeout = timeout

    def capabilities(self) -> TTSCapabilities:
        dialect = "drama3" if self.model == "drama-3" else ("s1" if self.model == "s1" else "s2")
        return TTSCapabilities(
            name=self.name,
            streaming=True,
            dialect=dialect,
            remote=True,
            destination=self.base_url,
            voice_kinds=("hosted_fish_id", "local_voice"),
            notes=f"official hosted API, model header {self.model!r}",
        )

    def _headers(self, ctype: str | None = None) -> dict[str, str]:
        h = {"Authorization": f"Bearer {self.api_key}", "model": self.model, "User-Agent": UA}
        if ctype:
            h["Content-Type"] = ctype
        return h

    def health(self) -> tuple[bool, str]:
        req = urllib.request.Request(self.base_url + "/wallet/self/api-credit", headers=self._headers())
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return 200 <= r.status < 300, f"HTTP {r.status}"
        except urllib.error.HTTPError as e:
            return False, f"HTTP {e.code} (check API key)"
        except Exception as e:  # noqa: BLE001
            return False, redact(f"unreachable: {e}", self.api_key)

    def get_voice(self, reference_id: str) -> dict[str, Any]:
        req = urllib.request.Request(f"{self.base_url}/model/{reference_id}", headers=self._headers())
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise TTSError(f"hosted voice {reference_id}: HTTP {e.code}") from e

    def synthesize(self, text, *, voice=None, cancel=None) -> Iterator[AudioChunk]:
        ctype, pack = _packer()
        body = build_request(
            text,
            voice=voice,
            fmt="pcm",
            chunk_length=self.chunk_length,
            temperature=self.temperature,
            top_p=self.top_p,
            latency=self.latency,
            hosted=True,
            sample_rate=self.sample_rate,
        )
        req = urllib.request.Request(self.base_url + "/v1/tts", data=pack(body), headers=self._headers(ctype), method="POST")
        yield from _stream_audio(
            req, timeout=self.timeout, cancel=cancel, raw_pcm_rate=self.sample_rate, secrets=(self.api_key,)
        )


def time_to_first_audio(backend: TTSBackend, text: str, voice: Any | None = None) -> tuple[float | None, float, int]:
    """(ms to first audio, total ms, samples) for one synthesis."""
    t0 = time.perf_counter()
    first = None
    n = 0
    for ch in backend.synthesize(text, voice=voice):
        if first is None:
            first = (time.perf_counter() - t0) * 1000
        n += ch.pcm.size
    return first, (time.perf_counter() - t0) * 1000, n
