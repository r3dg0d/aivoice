"""voice-models.com provider — uses the site's own fetch_data.php AJAX.

Research notes (2026-09-23):
- No documented public REST/OpenAPI/GraphQL API (probes return 404).
- robots.txt: Allow: / with sitemap.
- Frontend search POSTs to /fetch_data.php with {page, search} and receives
  JSON {table: html, pagination: html}. Download links typically point at
  Hugging Face / Google Drive / etc. — the site indexes, it does not host models.
- We call only that same endpoint the website uses, with a clear User-Agent,
  modest rate limiting, and no CAPTCHA/auth bypass.
- Automated download is supported when a direct http(s) URL is present
  (prefer Hugging Face). Google Drive / Mega may require manual download +
  `aivoice voices import-rvc`.
"""

from __future__ import annotations

import html as htmlmod
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from .base import ProviderCapabilities, SearchPage, VoiceModelRef

BASE = "https://voice-models.com"
FETCH = f"{BASE}/fetch_data.php"
UA = "aivoice/0.2 (+https://github.com/r3dg0d/aivoice; respectful catalog client)"
MIN_INTERVAL_S = 1.0

_last_request = 0.0


def _throttle() -> None:
    global _last_request
    now = time.monotonic()
    wait = MIN_INTERVAL_S - (now - _last_request)
    if wait > 0:
        time.sleep(wait)
    _last_request = time.monotonic()


def _post_form(url: str, data: dict[str, str], *, timeout: float = 30.0) -> tuple[int, str, dict[str, str]]:
    _throttle()
    body = urllib.parse.urlencode(data).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "User-Agent": UA,
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "X-Requested-With": "XMLHttpRequest",
            "Origin": BASE,
            "Referer": f"{BASE}/",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            headers = {k.lower(): v for k, v in resp.headers.items()}
            return int(resp.status), raw, headers
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        return int(e.code), raw, {}


def parse_table_html(table_html: str, *, provider: str = "voice-models") -> list[VoiceModelRef]:
    """Parse fetch_data.php table HTML into VoiceModelRef list (stdlib only)."""
    rows = re.findall(r"<tr>(.*?)</tr>", table_html, flags=re.S | re.I)
    out: list[VoiceModelRef] = []
    for row in rows:
        m = re.search(
            r"href=['\"](/model/([^'\"]+))['\"][^>]*>(.*?)</a>",
            row,
            flags=re.S | re.I,
        )
        if not m:
            continue
        path, slug, name_html = m.group(1), m.group(2), m.group(3)
        name = htmlmod.unescape(re.sub(r"<[^>]+>", "", name_html))
        name = re.sub(r"\s+", " ", name).strip()
        size_m = re.search(r"badge[^>]*>([^<]+)</span>", row, flags=re.I)
        size_label = size_m.group(1).strip() if size_m else None
        mid_m = re.search(r"data-model-id=['\"](\d+)['\"]", row, flags=re.I)
        numeric_id = mid_m.group(1) if mid_m else None
        # Prefer direct HF / http(s) download (not easyaivoice wrapper)
        hrefs = re.findall(r"href=['\"](https?://[^'\"]+)['\"]", row)
        download_url = None
        for h in hrefs:
            if "easyaivoice.com" in h:
                continue
            if any(
                host in h
                for host in (
                    "huggingface.co",
                    "hf.co",
                    "drive.google.com",
                    "mega.nz",
                    "mediafire.com",
                    "pixeldrain.com",
                    "github.com",
                )
            ):
                download_url = htmlmod.unescape(h)
                break
        if download_url is None:
            # unwrap easyaivoice run?url=
            for h in hrefs:
                if "easyaivoice.com/run" in h and "url=" in h:
                    q = urllib.parse.parse_qs(urllib.parse.urlparse(h).query)
                    if "url" in q and q["url"]:
                        download_url = urllib.parse.unquote(q["url"][0])
                        break
        arch = None
        for pat in (r"RVC\s*v?3", r"RVC\s*v?2", r"RVC\s*v?1", r"RVC"):
            am = re.search(pat, name, flags=re.I)
            if am:
                arch = am.group(0)
                # normalize
                low = arch.lower().replace(" ", "")
                if "v3" in low:
                    arch = "RVC v3"
                elif "v2" in low:
                    arch = "RVC v2"
                elif "v1" in low:
                    arch = "RVC v1"
                else:
                    arch = "RVC"
                break
        out.append(
            VoiceModelRef(
                provider=provider,
                id=slug,
                name=name,
                page_url=f"{BASE}{path}",
                download_url=download_url,
                architecture=arch,
                size_label=size_label,
                extra={"numeric_id": numeric_id} if numeric_id else {},
            )
        )
    return out


def pagination_has_more(pagination_html: str, page: int) -> bool:
    # Look for fetchData(N, ...) with N > page
    nums = [int(x) for x in re.findall(r"fetchData\((\d+)", pagination_html)]
    return any(n > page for n in nums)


class VoiceModelsProvider:
    name = "voice-models"

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            search=True,
            download=True,  # when a direct URL is present
            versions=False,
            details=True,
            offline=False,
            notes=(
                "Uses voice-models.com fetch_data.php (same as the website UI). "
                "No official public API documented. Downloads follow third-party "
                "links (HF preferred). Rate-limited (~1 req/s). Local voices work offline."
            ),
        )

    def search(self, query: str, *, page: int = 1) -> SearchPage:
        status, raw, _ = _post_form(FETCH, {"page": str(page), "search": query.strip()})
        if status != 200:
            raise RuntimeError(f"voice-models.com search HTTP {status}")
        try:
            data: dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError as e:
            raise RuntimeError("voice-models.com returned non-JSON search payload") from e
        table = data.get("table") or ""
        pagination = data.get("pagination") or ""
        results = parse_table_html(table)
        return SearchPage(
            query=query,
            page=page,
            results=results,
            has_more=pagination_has_more(pagination, page),
            raw_note="parsed from fetch_data.php HTML table; ratings/authors omitted when absent",
        )

    def get_model(self, model_id: str) -> VoiceModelRef:
        # Re-search by id slug is unreliable; fetch model page HTML lightly.
        _throttle()
        url = f"{BASE}/model/{urllib.parse.quote(model_id)}"
        req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "text/html"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            html = resp.read().decode("utf-8", errors="replace")
        title_m = re.search(r"<title>([^<]+)", html, flags=re.I)
        name = htmlmod.unescape(title_m.group(1)).strip() if title_m else model_id
        name = re.sub(r"\s*[|\-].*$", "", name).strip() or model_id
        hrefs = re.findall(r'href="(https?://[^"]+)"', html)
        download_url = None
        for h in hrefs:
            if "huggingface.co" in h and ("resolve" in h or h.endswith(".zip") or ".pth" in h):
                download_url = htmlmod.unescape(h)
                break
        if download_url is None:
            for h in hrefs:
                if any(x in h for x in ("drive.google.com", "mega.nz", "huggingface.co")):
                    download_url = htmlmod.unescape(h)
                    break
        arch = None
        for label, pat in (("RVC v2", r"RVC\s*v?2"), ("RVC v1", r"RVC\s*v?1"), ("RVC v3", r"RVC\s*v?3")):
            if re.search(pat, html[:5000], flags=re.I):
                arch = label
                break
        return VoiceModelRef(
            provider=self.name,
            id=model_id,
            name=name,
            page_url=url,
            download_url=download_url,
            architecture=arch,
        )

    def download_url_for(self, model: VoiceModelRef) -> str | None:
        return model.download_url
