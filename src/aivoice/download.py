"""Resumable, checksum-aware downloads + safe archive extraction."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import unquote, urlparse

from .paths import downloads_dir, ensure_dirs

UA = "aivoice/0.2 (+https://github.com/r3dg0d/aivoice)"
MAX_RETRIES = 3


class DownloadError(RuntimeError):
    pass


class UnsafeArchiveError(RuntimeError):
    pass


def cache_key(provider: str, model_id: str, version: str | None = None) -> str:
    raw = f"{provider}:{model_id}:{version or 'latest'}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def filename_from_url(url: str) -> str:
    path = unquote(urlparse(url).path)
    name = Path(path).name or "download.bin"
    name = name.split("?")[0]
    if not name or name in {".", ".."}:
        name = "download.bin"
    return name


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_cached(
    url: str,
    *,
    provider: str,
    model_id: str,
    version: str | None = None,
    expected_sha256: str | None = None,
    timeout: float = 120.0,
    progress: bool = True,
) -> Path:
    """Download URL into XDG cache (resumable Range). Returns final path."""
    if not url.lower().startswith(("http://", "https://")):
        raise DownloadError(f"refusing non-http(s) URL: {url!r}")
    # Google Drive / Mega often need interactive auth — caller should handle.
    ensure_dirs()
    key = cache_key(provider, model_id, version)
    dest_dir = downloads_dir() / provider / key
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = filename_from_url(url)
    final = dest_dir / name
    meta = dest_dir / "source.json"
    if final.is_file() and final.stat().st_size > 0:
        if expected_sha256 and file_sha256(final).lower() != expected_sha256.lower():
            final.unlink(missing_ok=True)
        else:
            return final

    free = shutil.disk_usage(dest_dir).free
    if free < 50 * 1024 * 1024:
        raise DownloadError(f"low disk space under {dest_dir} ({free} bytes free)")

    partial = dest_dir / (name + ".partial")
    last_err: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            _download_once(url, partial, timeout=timeout, progress=progress)
            if expected_sha256:
                got = file_sha256(partial)
                if got.lower() != expected_sha256.lower():
                    raise DownloadError(f"checksum mismatch: got {got}, expected {expected_sha256}")
            os.replace(partial, final)
            meta.write_text(
                json.dumps(
                    {
                        "url": url,
                        "provider": provider,
                        "model_id": model_id,
                        "version": version,
                        "sha256": file_sha256(final),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            return final
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(min(2**attempt, 8))
    raise DownloadError(f"download failed after {MAX_RETRIES} attempts: {last_err}")


def _download_once(url: str, partial: Path, *, timeout: float, progress: bool) -> None:
    existing = partial.stat().st_size if partial.is_file() else 0
    headers = {"User-Agent": UA, "Accept": "*/*"}
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"
    req = urllib.request.Request(url, headers=headers)
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        if e.code == 416 and existing > 0:
            # already complete?
            return
        raise DownloadError(f"HTTP {e.code} for {url}") from e

    with resp:
        # If server ignored Range, restart
        status = getattr(resp, "status", 200)
        mode = "ab" if status == 206 else "wb"
        if mode == "wb" and existing:
            existing = 0
        total_hdr = resp.headers.get("Content-Length")
        total = int(total_hdr) + existing if total_hdr and status == 206 else (int(total_hdr) if total_hdr else None)
        got = existing
        with partial.open(mode) as out:
            while True:
                chunk = resp.read(1024 * 256)
                if not chunk:
                    break
                out.write(chunk)
                got += len(chunk)
                if progress and total:
                    pct = min(100, int(100 * got / total))
                    bar = "#" * (pct // 5) + "-" * (20 - pct // 5)
                    print(f"\rDownloading... [{bar}] {pct}%", end="", flush=True)
                elif progress:
                    print(f"\rDownloading... {got // (1024 * 1024)} MiB", end="", flush=True)
        if progress:
            print()


def safe_extract(archive: Path, dest: Path, *, max_files: int = 10_000, max_total_bytes: int = 8 * 1024**3) -> Path:
    """Extract zip/tar safely (no path traversal, no symlink escape, size caps)."""
    dest.mkdir(parents=True, exist_ok=True)
    dest = dest.resolve()
    if zipfile.is_zipfile(archive):
        return _extract_zip(archive, dest, max_files=max_files, max_total_bytes=max_total_bytes)
    if tarfile.is_tarfile(archive):
        return _extract_tar(archive, dest, max_files=max_files, max_total_bytes=max_total_bytes)
    # bare .pth/.index — just copy
    if archive.suffix.lower() in {".pth", ".pt", ".index", ".wav", ".flac", ".mp3", ".ogg"}:
        target = dest / archive.name
        shutil.copy2(archive, target)
        return dest
    raise DownloadError(f"unsupported archive type: {archive}")


def _safe_join(dest: Path, member: str) -> Path:
    # reject absolute / .. / null
    if member.startswith("/") or member.startswith("\\") or ".." in Path(member).parts or "\x00" in member:
        raise UnsafeArchiveError(f"unsafe archive path: {member!r}")
    out = (dest / member).resolve()
    if not str(out).startswith(str(dest) + os.sep) and out != dest:
        raise UnsafeArchiveError(f"path escapes destination: {member!r}")
    return out


def _extract_zip(archive: Path, dest: Path, *, max_files: int, max_total_bytes: int) -> Path:
    total = 0
    with zipfile.ZipFile(archive) as zf:
        infos = zf.infolist()
        if len(infos) > max_files:
            raise UnsafeArchiveError(f"too many files in zip ({len(infos)})")
        for info in infos:
            if info.is_dir():
                _safe_join(dest, info.filename).mkdir(parents=True, exist_ok=True)
                continue
            # zip bombs: refuse extreme compression ratios on large claimed sizes
            if info.file_size > max_total_bytes:
                raise UnsafeArchiveError(f"zip member too large: {info.filename}")
            total += info.file_size
            if total > max_total_bytes:
                raise UnsafeArchiveError("zip total uncompressed size exceeds limit")
            target = _safe_join(dest, info.filename)
            target.parent.mkdir(parents=True, exist_ok=True)
            # skip symlinks (zip zipinfo external_attr)
            with zf.open(info) as src, target.open("wb") as out:
                shutil.copyfileobj(src, out, length=1024 * 1024)
    return dest


def _extract_tar(archive: Path, dest: Path, *, max_files: int, max_total_bytes: int) -> Path:
    total = 0
    with tarfile.open(archive) as tf:
        members = tf.getmembers()
        if len(members) > max_files:
            raise UnsafeArchiveError(f"too many files in tar ({len(members)})")
        for m in members:
            if m.issym() or m.islnk():
                raise UnsafeArchiveError(f"refusing symlink/hardlink in tar: {m.name}")
            if m.isdir():
                _safe_join(dest, m.name).mkdir(parents=True, exist_ok=True)
                continue
            if m.size > max_total_bytes:
                raise UnsafeArchiveError(f"tar member too large: {m.name}")
            total += m.size
            if total > max_total_bytes:
                raise UnsafeArchiveError("tar total size exceeds limit")
            target = _safe_join(dest, m.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            src = tf.extractfile(m)
            if src is None:
                continue
            with src, target.open("wb") as out:
                shutil.copyfileobj(src, out, length=1024 * 1024)
    return dest
