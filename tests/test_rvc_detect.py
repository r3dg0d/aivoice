"""RVC package detection without executing pickle."""

from __future__ import annotations

import zipfile
from pathlib import Path

from aivoice.rvc.detect import inspect_path


def test_zip_listing_v2(tmp_path: Path):
    z = tmp_path / "voice_rvc_v2.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("G_model.pth", b"x")
        zf.writestr("model.index", b"y")
        zf.writestr("preview.wav", b"RIFF")
    info = inspect_path(z)
    assert info.checkpoint
    assert info.index
    assert info.reference_audio
    assert info.generation and "v2" in info.generation.lower()


def test_dir_missing_audio(tmp_path: Path):
    d = tmp_path / "pkg"
    d.mkdir()
    (d / "model.pth").write_bytes(b"not-a-real-torch-file")
    info = inspect_path(d)
    assert info.checkpoint
    assert not info.has_usable_reference
