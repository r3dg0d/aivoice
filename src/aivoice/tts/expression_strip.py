"""Strip expression markup for engines that would read it aloud."""

from __future__ import annotations

import re

_BRACKET = re.compile(r"\[[^\[\]]{0,80}\]")
_PAIRED = re.compile(r"</?\s*[a-zA-Z_]{2,20}\s*>")
_PAREN_CUE = re.compile(r"\((?:[a-z][a-z ,-]{1,30})\)")
_SPEAKER = re.compile(r"<\|speaker:\d+\|>")


def strip_tags(text: str, *, parens: bool = False) -> str:
    out = _SPEAKER.sub("", text)
    out = _BRACKET.sub("", out)
    out = _PAIRED.sub("", out)
    if parens:
        out = _PAREN_CUE.sub("", out)
    return re.sub(r"\s{2,}", " ", out).strip()
