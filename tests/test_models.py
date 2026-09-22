import pytest

from aivoice.models import CATALOG, install_model, list_models


def test_catalog():
    assert "meanvc2" in CATALOG
    gap = CATALOG["meanvc2"]["code_license_gap"]
    assert "MISSING" in gap or "missing" in gap.lower()


def test_list_models(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    rows = list_models()
    assert any(r.name == "meanvc2" for r in rows)


def test_install_requires_yes(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    with pytest.raises(RuntimeError, match="--yes"):
        install_model("meanvc2", yes=False)
