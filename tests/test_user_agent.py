"""User-Agent strings must follow the package version, not a stale literal."""

from __future__ import annotations

import tomllib
from pathlib import Path

from aivoice import __version__
from aivoice.download import _download_once
from aivoice.providers.voice_models import VoiceModelsProvider, _post_form

_CATALOG_UA = f"aivoice/{__version__} (+https://github.com/r3dg0d/aivoice; respectful catalog client)"
_DOWNLOAD_UA = f"aivoice/{__version__} (+https://github.com/r3dg0d/aivoice)"


class _Resp:
    def __init__(self, body: bytes, *, status: int = 200, headers: dict | None = None):
        self.status = status
        self.headers = headers or {}
        self._body = body
        self._sent = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, n: int = -1) -> bytes:
        if self._sent:
            return b""
        self._sent = True
        if n < 0 or n >= len(self._body):
            return self._body
        chunk, self._body = self._body[:n], self._body[n:]
        self._sent = False
        return chunk


def _ua(req) -> str:
    headers = {k.lower(): v for k, v in req.header_items()}
    return headers["user-agent"]


def test_pyproject_version_matches_package():
    root = Path(__file__).resolve().parents[1]
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["project"]["version"] == __version__


def test_download_user_agent_tracks_package_version(monkeypatch, tmp_path: Path):
    captured: dict = {}

    def _urlopen(req, timeout=120):
        captured["ua"] = _ua(req)
        captured["url"] = req.full_url
        return _Resp(b"ok", headers={"Content-Length": "2"})

    monkeypatch.setattr("aivoice.download.urllib.request.urlopen", _urlopen)
    _download_once(
        "https://example.invalid/model.bin",
        tmp_path / "model.bin.partial",
        timeout=5,
        progress=False,
    )
    assert captured["url"] == "https://example.invalid/model.bin"
    assert captured["ua"] == _DOWNLOAD_UA
    assert __version__ in captured["ua"]
    assert "aivoice/0.2" not in captured["ua"]


def test_voice_models_search_user_agent_tracks_package_version(monkeypatch):
    captured: dict = {}

    def _urlopen(req, timeout=30):
        captured["ua"] = _ua(req)
        captured["url"] = req.full_url
        return _Resp(b"{}")

    monkeypatch.setattr("aivoice.providers.voice_models.urllib.request.urlopen", _urlopen)
    monkeypatch.setattr("aivoice.providers.voice_models._throttle", lambda: None)
    status, raw, _headers = _post_form("https://voice-models.com/fetch_data.php", {"page": "1", "search": "x"})
    assert status == 200
    assert raw == "{}"
    assert captured["url"] == "https://voice-models.com/fetch_data.php"
    assert captured["ua"] == _CATALOG_UA
    assert "aivoice/0.2" not in captured["ua"]


def test_voice_models_page_user_agent_tracks_package_version(monkeypatch):
    captured: dict = {}
    html = b"<html><head><title>Sample Voice</title></head><body></body></html>"

    def _urlopen(req, timeout=30):
        captured["ua"] = _ua(req)
        captured["url"] = req.full_url
        return _Resp(html)

    monkeypatch.setattr("aivoice.providers.voice_models.urllib.request.urlopen", _urlopen)
    monkeypatch.setattr("aivoice.providers.voice_models._throttle", lambda: None)
    ref = VoiceModelsProvider().get_model("sample-voice")
    assert ref.id == "sample-voice"
    assert captured["url"] == "https://voice-models.com/model/sample-voice"
    assert captured["ua"] == _CATALOG_UA
    assert "aivoice/0.2" not in captured["ua"]
