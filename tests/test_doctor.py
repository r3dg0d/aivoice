from aivoice import __version__
from aivoice.doctor import format_doctor, run_doctor


def test_doctor_offline():
    checks = run_doctor(check_network=False)
    assert any(c.name == "Voice library" for c in checks)
    text = format_doctor(checks)
    assert "aivoice doctor" in text


def test_doctor_user_agent_tracks_package_version(monkeypatch):
    """HEAD probe must advertise the package version, not a stale literal."""
    captured: dict = {}

    class _Resp:
        status = 204

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def _urlopen(req, timeout=10):
        captured["headers"] = dict(req.header_items())
        captured["url"] = req.full_url
        return _Resp()

    monkeypatch.setattr("aivoice.doctor.urllib.request.urlopen", _urlopen)
    checks = run_doctor(check_network=True)
    ua = captured["headers"].get("User-agent")
    assert ua == f"aivoice/{__version__} doctor"
    assert "aivoice/0.2" not in ua
    prov = next(c for c in checks if c.name == "provider:voice-models")
    assert prov.ok
    assert captured["url"] == "https://voice-models.com/"
