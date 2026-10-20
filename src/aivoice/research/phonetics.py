"""Text normalisation, language detection, code-switch segmentation and an IPA view.

Adapted from X-Voice (arXiv:2605.05611): X-Voice feeds IPA and per-segment
language identifiers to its model. aivoice cannot make another model multilingual
by changing its input, so this layer is used (a) to normalise text the same way
for every engine, (b) to split code-switched text into language runs and format
them with X-Voice's own ``[main|lang]`` tags, and (c) to provide IPA for duration
planning and pronunciation checks. IPA comes from espeak-ng.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import unicodedata
from dataclasses import dataclass
from pathlib import Path

ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
        "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
EN_ABBR = {
    "dr.": "doctor", "mr.": "mister", "mrs.": "missus", "ms.": "miz", "st.": "saint", "vs.": "versus",
    "etc.": "et cetera", "e.g.": "for example", "i.e.": "that is", "approx.": "approximately", "no.": "number",
}


def _en_int(n: int) -> str:
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else "-" + ONES[n % 10])
    if n < 1000:
        return ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + _en_int(n % 100))
    for div, word in ((10**9, "billion"), (10**6, "million"), (1000, "thousand")):
        if n >= div:
            return _en_int(n // div) + " " + word + ("" if n % div == 0 else " " + _en_int(n % div))
    return str(n)


def _en_number(tok: str) -> str:
    tok = tok.replace(",", "")
    if "." in tok:
        a, b = tok.split(".", 1)
        return _en_int(int(a or 0)) + " point " + " ".join(ONES[int(d)] for d in b if d.isdigit())
    return _en_int(int(tok))


def normalize_text(text: str, language: str = "en", *, lexicon: dict[str, str] | None = None) -> str:
    """Unicode NFKC, quotes/dashes, whitespace; English numbers, %, $, abbreviations; user lexicon."""
    t = unicodedata.normalize("NFKC", text)
    t = t.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    t = re.sub(r"[–—]", ", ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if lexicon:
        for word, say in sorted(lexicon.items(), key=lambda kv: -len(kv[0])):
            t = re.sub(rf"(?<!\w){re.escape(word)}(?!\w)", say, t, flags=re.I)
    if language.startswith("en"):
        for abbr, full in EN_ABBR.items():
            t = re.sub(rf"(?<!\w){re.escape(abbr)}", full, t, flags=re.I)
        t = re.sub(r"\$(\d[\d,]*)\.(\d{2})\b", lambda m: _en_number(m.group(1)) + " dollars and " + _en_int(int(m.group(2))) + " cents", t)
        t = re.sub(r"\$(\d[\d,]*(?:\.\d+)?)", lambda m: _en_number(m.group(1)) + " dollars", t)
        t = re.sub(r"(\d[\d,]*(?:\.\d+)?)\s?%", lambda m: _en_number(m.group(1)) + " percent", t)
        t = re.sub(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b|\b\d+(?:\.\d+)?\b", lambda m: _en_number(m.group(0)), t)
        if not t.isupper():  # an all-caps sentence is shouting, not a string of acronyms
            t = re.sub(r"\b([A-Z]{2,5})\b", lambda m: " ".join(m.group(1)) if m.group(1) not in ("OK", "I") else m.group(1), t)
    return t


SCRIPTS = [
    ("ko", r"[가-힯ᄀ-ᇿ]"),
    ("ja", r"[぀-ヿ]"),
    ("zh", r"[一-鿿]"),
    ("ru", r"[Ѐ-ӿ]"),
    ("ar", r"[؀-ۿ]"),
    ("he", r"[֐-׿]"),
    ("hi", r"[ऀ-ॿ]"),
    ("th", r"[฀-๿]"),
    ("el", r"[Ͱ-Ͽ]"),
]
STOPWORDS = {
    "en": {"the", "and", "is", "you", "to", "of", "it", "that", "this", "with", "are", "was", "what", "i"},
    "es": {"el", "la", "de", "que", "y", "en", "los", "es", "por", "una", "con", "para", "pero", "muy"},
    "fr": {"le", "la", "les", "de", "et", "est", "que", "une", "pas", "pour", "dans", "vous", "je", "avec"},
    "de": {"der", "die", "und", "ist", "nicht", "das", "ich", "ein", "eine", "mit", "zu", "sie", "es", "auf"},
    "it": {"il", "di", "che", "e", "la", "per", "non", "una", "sono", "con", "come", "anche", "ma", "gli"},
    "pt": {"o", "de", "que", "e", "do", "da", "em", "um", "para", "não", "uma", "os", "com", "mais"},
}


def detect_language(text: str, default: str = "en") -> str:
    """Script first, then stopword overlap for Latin-script languages. A heuristic, labelled as such."""
    counts = {lang: len(re.findall(p, text)) for lang, p in SCRIPTS}
    if counts.get("ja") and counts.get("zh"):
        counts["ja"] += counts["zh"]
    best = max(counts.items(), key=lambda kv: kv[1])
    if best[1] > 0:
        return best[0]
    words = set(re.findall(r"[^\W\d_]+", text.lower()))
    scores = {lang: len(words & sw) for lang, sw in STOPWORDS.items()}
    lang, score = max(scores.items(), key=lambda kv: kv[1])
    return lang if score > 0 else default


@dataclass
class Run:
    language: str
    text: str


def segment_code_switch(text: str, default: str = "en") -> list[Run]:
    """Split into runs by script (Latin vs CJK/Cyrillic/...). Latin runs share one detected language."""
    pieces = re.findall(r"[぀-ヿ一-鿿가-힯Ѐ-ӿ؀-ۿ֐-׿ऀ-ॿ฀-๿Ͱ-Ͽ][^A-Za-z]*|[^぀-ヿ一-鿿가-힯Ѐ-ӿ؀-ۿ֐-׿ऀ-ॿ฀-๿Ͱ-Ͽ]+", text)
    latin_lang = detect_language(" ".join(p for p in pieces if re.search(r"[A-Za-z]", p)), default)
    runs: list[Run] = []
    for p in pieces:
        if not p.strip():
            if runs:
                runs[-1].text += p
            continue
        lang = detect_language(p, latin_lang) if not re.search(r"[A-Za-z]", p) else latin_lang
        if runs and runs[-1].language == lang:
            runs[-1].text += p
        else:
            runs.append(Run(lang, p))
    for r in runs:
        r.text = r.text.strip()
    return [r for r in runs if r.text]


def xvoice_tagged(text: str, default: str = "en", voice: str = "main") -> str:
    """X-Voice's documented code-switch tag format: one '[voice|lang]text' line per run."""
    return "\n".join(f"[{voice}|{r.language}]{r.text}" for r in segment_code_switch(text, default))


ESPEAK_VOICE = {"zh": "cmn", "ja": "ja", "ko": "ko", "en": "en-us"}


def to_ipa(text: str, language: str = "en") -> str | None:
    exe = shutil.which("espeak-ng")
    if not exe:
        return None
    try:
        r = subprocess.run([exe, "-q", "--ipa", "-v", ESPEAK_VOICE.get(language, language), text],
                           capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return re.sub(r"\s+", " ", r.stdout).strip() if r.returncode == 0 else None


def load_lexicon(path: str | Path | None) -> dict[str, str]:
    if not path:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in data.items()):
        raise ValueError("lexicon must be a JSON object of word -> spoken form")
    return data
