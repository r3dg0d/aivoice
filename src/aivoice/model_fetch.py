"""Verified, resumable model downloads (HTTP Range) with HTML-page and format detection.

Used by ``aivoice models install``: a file is written under ``<dest>.part`` and only renamed into
place after its size and SHA-256 match. Upstream hosts that publish no digest (Google Drive) get
their digest recorded on the first verified download and checked from then on.
"""

from __future__ import annotations

import hashlib
import os
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

ALLOWED = ("https://huggingface.co/", "https://github.com/", "https://drive.usercontent.google.com/")
DRIVE = "https://drive.usercontent.google.com/download?id={}&export=download&confirm=t"

Progress = Callable[[str, int, int], None]


class FetchError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def looks_like_html(head: bytes) -> bool:
    h = head.lstrip()[:64].lower()
    return h.startswith((b"<!doctype html", b"<html", b"<head")) or b"<html" in h


def fetch(spec: dict[str, Any], dest: Path, progress: Progress | None = None, *, user_agent: str = "aivoice",
          timeout: float = 60.0) -> Path:
    """Download ``spec['url']`` to ``dest``; ``spec`` has ``size`` and ``sha256`` (either may be None)."""
    url, size, digest = spec["url"], spec.get("size"), spec.get("sha256")
    if not url.startswith(ALLOWED):
        raise FetchError(f"refusing download from non-allowlisted host: {url}")
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and (size is None or dest.stat().st_size == size) and (digest is None or sha256(dest) == digest):
        return dest
    part = dest.with_name(dest.name + ".part")
    offset = part.stat().st_size if part.is_file() else 0
    if size is not None and offset > size:
        part.unlink()
        offset = 0
    if size is None or offset < size:
        headers = {"User-Agent": user_agent}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as r:
                if r.headers.get("Content-Type", "").startswith("text/html"):
                    raise FetchError(f"{dest.name}: {url} returned an HTML page (quota/permission/removed?)")
                resumed = offset > 0 and r.status == 206
                done = offset if resumed else 0
                total = size or (int(r.headers.get("Content-Length") or 0) + done)
                with part.open("ab" if resumed else "wb") as out:
                    first = not resumed
                    while chunk := r.read(1024 * 1024):
                        if first and looks_like_html(chunk[:512]):
                            raise FetchError(f"{dest.name}: download is an HTML page, not a model ({url})")
                        first = False
                        out.write(chunk)
                        done += len(chunk)
                        if progress:
                            progress(dest.name, done, total)
        except urllib.error.HTTPError as e:
            raise FetchError(f"{dest.name}: HTTP {e.code} from {url}") from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise FetchError(f"{dest.name}: download interrupted ({e}); re-run to resume") from e
    got = part.stat().st_size
    if size is not None and got != size:
        raise FetchError(f"{dest.name}: incomplete ({got} of {size} bytes); re-run to resume")
    with part.open("rb") as f:
        if looks_like_html(f.read(512)):
            part.unlink()
            raise FetchError(f"{dest.name}: file is an HTML page")
    if digest is not None and (actual := sha256(part)) != digest:
        part.unlink()
        raise FetchError(f"{dest.name}: SHA-256 mismatch (expected {digest}, got {actual}); partial file discarded")
    os.chmod(part, 0o600)
    part.replace(dest)
    return dest


def check_format(path: Path, fmt: str) -> str:
    """Structural check without executing anything: torch zip archives (incl. TorchScript), safetensors."""
    path = Path(path)
    with path.open("rb") as f:
        head = f.read(16)
    if looks_like_html(head):
        raise FetchError(f"{path.name}: is an HTML page")
    if fmt in ("torch-zip", "torchscript"):
        if not zipfile.is_zipfile(path):
            raise FetchError(f"{path.name}: not a torch zip archive")
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            if fmt == "torchscript" and not any("/code/" in n for n in names):
                raise FetchError(f"{path.name}: zip has no code/ directory; not a TorchScript archive")
            if not any(n.endswith("data.pkl") for n in names):
                raise FetchError(f"{path.name}: zip has no data.pkl; not a torch archive")
            bad = z.testzip()
            if bad:
                raise FetchError(f"{path.name}: corrupt member {bad}")
        return f"{fmt} ({len(names)} members, CRC ok)"
    if fmt == "safetensors":
        import json

        n = int.from_bytes(head[:8], "little")
        if not 0 < n < 100_000_000:
            raise FetchError(f"{path.name}: bad safetensors header")
        with path.open("rb") as f:
            f.seek(8)
            keys = json.loads(f.read(n))
        return f"safetensors ({len(keys) - ('__metadata__' in keys)} tensors)"
    raise FetchError(f"unknown format {fmt}")


__all__ = ["DRIVE", "FetchError", "check_format", "fetch", "looks_like_html", "sha256"]
