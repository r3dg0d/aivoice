import hashlib
import http.server
import threading

import pytest

from aivoice import model_fetch as mf


class _H(http.server.BaseHTTPRequestHandler):
    files: dict = {}

    def log_message(self, *a):
        pass

    def do_GET(self):
        body, ctype = self.files.get(self.path, (None, None))
        if body is None:
            self.send_error(404)
            return
        rng = self.headers.get("Range")
        if rng:
            start = int(rng.split("=")[1].split("-")[0])
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(body) - 1}/{len(body)}")
            body = body[start:]
        else:
            self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def server(monkeypatch):
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    monkeypatch.setattr(mf, "ALLOWED", mf.ALLOWED + (base,))
    _H.files = {}
    yield base, _H.files
    srv.shutdown()


def spec(base, name, body):
    return {"url": f"{base}/{name}", "size": len(body), "sha256": hashlib.sha256(body).hexdigest()}


def test_fetch_verifies_and_resumes(server, tmp_path):
    base, files = server
    body = bytes(range(256)) * 300
    files["/m.pt"] = (body, "application/octet-stream")
    (tmp_path / "m.pt.part").write_bytes(body[:7000])
    mf.fetch(spec(base, "m.pt", body), tmp_path / "m.pt")
    assert (tmp_path / "m.pt").read_bytes() == body


def test_fetch_rejects_html_and_bad_digest(server, tmp_path):
    base, files = server
    files["/q.pt"] = (b"<!DOCTYPE html><html>Google Drive - Quota exceeded</html>", "text/html")
    with pytest.raises(mf.FetchError, match="HTML"):
        mf.fetch({"url": f"{base}/q.pt", "size": None, "sha256": None}, tmp_path / "q.pt")
    files["/x.pt"] = (b"y" * 64, "application/octet-stream")
    with pytest.raises(mf.FetchError, match="SHA-256"):
        mf.fetch({"url": f"{base}/x.pt", "size": 64, "sha256": "0" * 64}, tmp_path / "x.pt")
    assert not (tmp_path / "x.pt").exists() and not (tmp_path / "x.pt.part").exists()


def test_fetch_refuses_unlisted_hosts(tmp_path):
    with pytest.raises(mf.FetchError, match="allowlisted"):
        mf.fetch({"url": "https://example.org/w.pt", "size": 1, "sha256": None}, tmp_path / "w.pt")


def test_meanvc2_not_ready_from_marker_alone(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    from aivoice import models

    d = models.models_dir() / "meanvc2"
    d.mkdir(parents=True)
    (d / "READY").write_text("x")
    assert not models.is_ready("meanvc2")
