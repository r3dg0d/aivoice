"""Client side of the isolated engine worker (see worker_main.py)."""

from __future__ import annotations

import base64
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from ..paths import runtime_dir
from .base import EngineError, EngineUnavailable

WORKER_SCRIPT = Path(__file__).with_name("worker_main.py")


def engine_python(engine: str) -> str:
    """Interpreter for an engine: AIVOICE_<ENGINE>_PYTHON, settings engines.<engine>.python, else this one."""
    env = os.environ.get(f"AIVOICE_{engine.upper().replace('-', '_')}_PYTHON")
    if env:
        return env
    try:
        from .. import settings

        p = settings.get(f"engines.{engine}.python")
        if p:
            return str(p)
    except Exception:  # noqa: BLE001
        pass
    return sys.executable


class WorkerClient:
    def __init__(self, handler: str, *, python: str | None = None, env: dict[str, str] | None = None,
                 timeout_s: float = 600.0) -> None:
        self.handler = handler
        self.python = python or sys.executable
        self.env = env
        self.timeout_s = timeout_s
        self._proc: subprocess.Popen | None = None
        self._id = 0
        self._lock = threading.Lock()
        self._log_path: Path | None = None
        self._lines: queue.Queue = queue.Queue()
        self._reader: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self) -> None:
        if self.running:
            return
        runtime_dir().mkdir(parents=True, exist_ok=True)
        self._log_path = runtime_dir() / f"worker-{self.handler}.log"
        log = open(self._log_path, "ab")
        env = {**os.environ, **(self.env or {})}
        try:
            self._proc = subprocess.Popen(
                [self.python, "-u", str(WORKER_SCRIPT), self.handler],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=log,
                env=env,
            )
        except OSError as e:
            raise EngineUnavailable(f"cannot start worker with {self.python}: {e}") from e
        finally:
            log.close()
        # A reader thread, not select(): several JSON lines can sit in the pipe's
        # Python buffer at once, which select() would never report as readable.
        self._lines = queue.Queue()
        proc = self._proc

        def pump() -> None:
            assert proc.stdout
            for line in proc.stdout:
                self._lines.put(line)
            self._lines.put(None)

        self._reader = threading.Thread(target=pump, name=f"aivoice-worker-{self.handler}", daemon=True)
        self._reader.start()
        hello = self._read(timeout=60)
        if not hello.get("ok"):
            raise EngineUnavailable(f"worker failed to start: {hello.get('error')}")

    def _read(self, timeout: float) -> dict[str, Any]:
        end = time.monotonic() + timeout
        while True:
            left = end - time.monotonic()
            if left <= 0:
                self.stop()
                raise EngineError(f"{self.handler} worker timed out after {timeout:.0f}s")
            try:
                line = self._lines.get(timeout=min(left, 1.0))
            except queue.Empty:
                continue
            if line is None:
                code = self._proc.poll() if self._proc else None
                raise EngineError(f"{self.handler} worker exited (code {code}); see {self._log_path}")
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue

    def request(self, op: str, **kw: Any) -> Iterator[dict[str, Any]]:
        """Yield chunk messages, then the final response (raises on error)."""
        with self._lock:
            if not self.running:
                self.start()
            self._id += 1
            rid = self._id
            assert self._proc and self._proc.stdin
            self._proc.stdin.write((json.dumps({"id": rid, "op": op, **kw}) + "\n").encode())
            self._proc.stdin.flush()
            while True:
                msg = self._read(self.timeout_s)
                if msg.get("id") != rid:
                    continue
                if "chunk" in msg:
                    yield msg
                    continue
                if not msg.get("ok"):
                    raise EngineError(f"{self.handler}: {msg.get('error')}")
                yield msg
                return

    def call(self, op: str, **kw: Any) -> dict[str, Any]:
        last: dict[str, Any] = {}
        for msg in self.request(op, **kw):
            last = msg
        return last

    def synthesize(self, **kw: Any) -> dict[str, Any]:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=runtime_dir()) as f:
            out = f.name
        return self.call("synthesize", out_path=out, **kw)

    def stream(self, **kw: Any) -> Iterator[tuple[np.ndarray, int]]:
        for msg in self.request("synthesize", stream=True, **kw):
            if "chunk" in msg:
                yield np.frombuffer(base64.b64decode(msg["chunk"]), dtype="<f4").copy(), int(msg["sample_rate"])

    def stop(self) -> None:
        p, self._proc = self._proc, None
        if p is None:
            return
        try:
            if p.poll() is None and p.stdin:
                p.stdin.write(b'{"id": -9, "op": "shutdown"}\n')
                p.stdin.flush()
                p.wait(timeout=5)
        except Exception:  # noqa: BLE001
            p.kill()
