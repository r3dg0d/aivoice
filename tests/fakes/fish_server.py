"""A stand-in for fish-speech's tools/api_server.py, for tests only.

Implements the documented surface (``/v1/health``, ``/v1/tts`` with msgpack or
JSON bodies and streaming WAV, ``/v1/tts/defaults``, ``/v1/references/*``) and
records every request. It produces tones, not speech: it tests aivoice's
client, never Drama 3 itself.
"""

from __future__ import annotations

import base64
import json
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np


def wav_header(rate: int, channels: int = 1, bits: int = 16, data_size: int = 0) -> bytes:
    byte_rate = rate * channels * bits // 8
    return (
        b"RIFF" + struct.pack("<I", 36 + data_size) + b"WAVE"
        + b"fmt " + struct.pack("<IHHIIHH", 16, 1, channels, rate, byte_rate, channels * bits // 8, bits)
        + b"data" + struct.pack("<I", data_size)
    )


def _unpack(ctype: str, body: bytes) -> dict:
    if "msgpack" in ctype:
        try:
            import msgpack  # type: ignore

            return msgpack.unpackb(body, raw=False)
        except ImportError:
            import ormsgpack  # type: ignore

            return ormsgpack.unpackb(body)
    return json.loads(body)


class FakeFishServer:
    def __init__(self, *, rate: int = 44100, ttfa_ms: float = 30, chunk_ms: int = 40, model: str = "drama-3",
                 require_key: str | None = None, fail_tts: bool = False) -> None:
        self.rate = rate
        self.ttfa_ms = ttfa_ms
        self.chunk_ms = chunk_ms
        self.model = model
        self.require_key = require_key
        self.fail_tts = fail_tts
        self.requests: list[dict] = []
        self.headers: list[dict] = []
        self.references: dict[str, dict] = {}
        srv = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _auth(self) -> bool:
                if srv.require_key and self.headers.get("Authorization") != f"Bearer {srv.require_key}":
                    self._json(401, {"detail": "unauthorized"})
                    return False
                return True

            def _json(self, code, obj):
                raw = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                srv.headers.append(dict(self.headers))
                if not self._auth():
                    return
                if self.path == "/v1/health":
                    self._json(200, {"status": "ok"})
                elif self.path == "/v1/tts/defaults":
                    d = {"temperature": 0.7, "top_p": 0.7}
                    if srv.model == "drama-3":
                        d["single_chunk"] = False
                    self._json(200, d)
                elif self.path == "/v1/references/list":
                    self._json(200, {"success": True, "reference_ids": sorted(srv.references), "message": "Success"})
                elif self.path.startswith("/model/"):
                    self._json(200, {"_id": self.path.split("/")[-1], "title": "Hosted", "state": "trained"})
                elif self.path == "/wallet/self/api-credit":
                    self._json(200, {"credit": "1.0"})
                else:
                    self._json(404, {"detail": "not found"})

            def _read(self) -> bytes:
                n = int(self.headers.get("Content-Length") or 0)
                return self.rfile.read(n)

            def do_DELETE(self):
                body = self._read().decode("utf-8", "replace")
                rid = body.split('name="reference_id"\r\n\r\n', 1)[-1].split("\r\n", 1)[0]
                srv.references.pop(rid, None)
                self._json(200, {"success": True, "message": "deleted", "reference_id": rid})

            def do_POST(self):
                srv.headers.append(dict(self.headers))
                if not self._auth():
                    return
                body = self._read()
                if self.path == "/v1/references/add":
                    text = body.decode("latin-1")
                    rid = text.split('name="id"\r\n\r\n', 1)[1].split("\r\n", 1)[0]
                    ref_text = text.split('name="text"\r\n\r\n', 1)[1].split("\r\n", 1)[0]
                    srv.references[rid] = {"text": ref_text, "bytes": len(body)}
                    self._json(200, {"success": True, "message": "added", "reference_id": rid})
                    return
                if self.path != "/v1/tts":
                    self._json(404, {"detail": "not found"})
                    return
                req = _unpack(self.headers.get("Content-Type", ""), body)
                for r in req.get("references") or []:
                    if isinstance(r.get("audio"), str):
                        r["audio"] = base64.b64decode(r["audio"])
                srv.requests.append(req)
                if srv.fail_tts:
                    self._json(500, {"detail": "boom"})
                    return
                if req.get("reference_id") and req["reference_id"] not in srv.references and "model" not in self.headers:
                    self._json(400, {"detail": f"reference {req['reference_id']} not found"})
                    return
                words = max(1, len(str(req.get("text", "")).split()))
                n = int(srv.rate * words / 3.0)
                t = np.arange(n) / srv.rate
                pcm = (0.25 * np.sin(2 * np.pi * 200 * t) * 32767).astype("<i2").tobytes()
                streaming = req.get("streaming", True)
                fmt = req.get("format", "wav")
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav" if fmt == "wav" else "application/octet-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()

                def chunk(b: bytes):
                    self.wfile.write(f"{len(b):x}\r\n".encode() + b + b"\r\n")
                    self.wfile.flush()

                time.sleep(srv.ttfa_ms / 1000)
                if fmt == "wav":
                    chunk(wav_header(srv.rate, data_size=0 if streaming else len(pcm)))
                step = int(srv.rate * srv.chunk_ms / 1000) * 2
                for i in range(0, len(pcm), step if streaming else len(pcm)):
                    chunk(pcm[i : i + step] if streaming else pcm)
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.httpd.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self._t = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()
        return False
