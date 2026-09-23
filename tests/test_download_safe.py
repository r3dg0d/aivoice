"""Safe archive extraction tests."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from aivoice.download import UnsafeArchiveError, filename_from_url, safe_extract


def test_filename_from_url():
    assert filename_from_url("https://example.com/a/b/model.zip?download=true") == "model.zip"


def test_zip_ok(tmp_path: Path):
    z = tmp_path / "ok.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("model.pth", b"abc")
        zf.writestr("ref.wav", b"RIFF")
    out = tmp_path / "out"
    safe_extract(z, out)
    assert (out / "model.pth").read_bytes() == b"abc"


def test_zip_traversal_rejected(tmp_path: Path):
    z = tmp_path / "bad.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("../evil.pth", b"nope")
    with pytest.raises(UnsafeArchiveError):
        safe_extract(z, tmp_path / "out")
