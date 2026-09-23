"""Offline tests for voice-models.com table parser (fixture, no network)."""

from __future__ import annotations

import json
from pathlib import Path

from aivoice.providers.voice_models import parse_table_html, pagination_has_more

FIX = Path(__file__).parent / "fixtures" / "voice_models" / "search_gura_page1.json"


def test_parse_fixture_rows():
    data = json.loads(FIX.read_text(encoding="utf-8"))
    rows = parse_table_html(data["table"])
    assert len(rows) >= 1
    assert rows[0].provider == "voice-models"
    assert rows[0].id
    assert rows[0].name
    # download URL should be present on fixture rows
    assert rows[0].download_url is None or rows[0].download_url.startswith("http")


def test_pagination_detect():
    html = "fetchData(1, 'x'); fetchData(2, 'x');"
    assert pagination_has_more(html, 1) is True
    assert pagination_has_more(html, 2) is False
