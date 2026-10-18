"""ExpressionCompiler: turn (text, emotion, delivery) into markup the active TTS actually understands.

Dialects, from Fish Audio's published syntax (Oct 2026):

``drama3``
    Free-form direction in square brackets before the words it shapes
    (``[barely holding it together] ...``), point tags for sounds and pauses
    (``[laughs]``, ``[sighs]``, ``[pause]``, ``[breath]``), and paired scoped
    tags around words: ``<whisper>``, ``<emphasis>``, ``<soft>``, ``<fast>``,
    ``<slow>``, ``<stress>``, ``<forceful>``. Fish notes wordings close to the
    documented ones work best, so the compiler emits those.
``s2``
    S2-Pro: square-bracket cues such as ``[excited]``, ``[whisper]``,
    ``[angry]``, ``[laughing]``, ``[pause]``. No paired tags.
``s1``
    OpenAudio S1: cues in parentheses, e.g. ``(excited)``.
``none``
    Engines that would read markup aloud (espeak, Piper): everything is stripped.

The compiler validates any markup coming from the LLM against the dialect and
drops what the backend would not understand, so unsupported tags never reach it.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from .tts.expression_strip import strip_tags

DRAMA3_SCOPED = ("whisper", "emphasis", "soft", "fast", "slow", "stress", "forceful")
DRAMA3_POINT = ("laughs", "laugh", "sighs", "sigh", "pause", "breath", "shaky breath", "chuckles", "gasps", "clears throat")
# Only the S2 cues Fish documents by example; others may work but are unverified.
S2_TAGS = ("excited", "whisper", "angry", "laughing", "pause")
S1_TAGS = ("excited", "angry", "sad", "surprised", "whispering", "shouting", "laughing", "sighing", "sarcastic", "confused")

EMOTIONS = (
    "neutral", "happy", "excited", "sad", "angry", "confused", "sarcastic", "laughing",
    "sighing", "whisper", "shouting", "breathy", "calm", "surprised",
)

# emotion -> (drama3 markup, s2 tag, s1 tag); "{text}" marks scoped wrapping.
EMOTION_MAP: dict[str, tuple[str | None, str | None, str | None]] = {
    "neutral": (None, None, None),
    "happy": ("[warm, smiling]", None, None),
    "excited": ("[excited]", "excited", "excited"),
    "sad": ("[sad, quiet]", None, "sad"),
    "angry": ("[angry]", "angry", "angry"),
    "confused": ("[confused]", None, "confused"),
    "sarcastic": ("[sarcastic]", None, "sarcastic"),
    "laughing": ("[laughs]", "laughing", "laughing"),
    "sighing": ("[sighs]", None, "sighing"),
    "whisper": ("<whisper>{text}</whisper>", "whisper", "whispering"),
    "shouting": ("<forceful>{text}</forceful>", None, "shouting"),
    "breathy": ("[breathy]", None, None),
    "calm": ("[calm]", None, None),
    "surprised": ("[surprised]", None, "surprised"),
}

_BRACKET = re.compile(r"\[([^\[\]]{1,80})\]")
_PAIRED = re.compile(r"<\s*(/?)\s*([a-zA-Z_]+)\s*>")
_PAREN = re.compile(r"\(([a-z][a-z ,-]{1,30})\)")


@dataclass
class CompiledText:
    text: str
    tags: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    dialect: str = "none"


class ExpressionCompiler:
    def __init__(
        self,
        dialect: str = "drama3",
        *,
        intensity: float = 0.5,
        frequency: float = 0.5,
        expression_only: bool = False,
        max_direction_words: int = 8,
        allow_free_direction: bool = True,
    ) -> None:
        if dialect not in ("drama3", "s2", "s1", "none"):
            raise ValueError(f"unknown dialect {dialect!r}")
        self.dialect = dialect
        self.intensity = max(0.0, min(1.0, intensity))
        self.frequency = max(0.0, min(1.0, frequency))
        self.expression_only = expression_only
        self.max_direction_words = max_direction_words
        self.allow_free_direction = allow_free_direction

    # --- validation ------------------------------------------------------

    def _bracket_ok(self, inner: str) -> bool:
        inner = inner.strip().lower()
        if not inner:
            return False
        if self.dialect == "drama3":
            if inner in DRAMA3_POINT:
                return True
            if not self.allow_free_direction:
                return False
            words = inner.replace(",", " ").split()
            return 0 < len(words) <= self.max_direction_words and not any(c in inner for c in "<>{}|\\")
        if self.dialect == "s2":
            return inner in S2_TAGS
        return False

    def validate(self, text: str) -> CompiledText:
        """Keep only markup the dialect supports; drop the rest (reported in ``removed``)."""
        if self.dialect == "none":
            stripped = strip_tags(text, parens=False)
            removed = _BRACKET.findall(text) + [m.group(0) for m in _PAIRED.finditer(text)]
            return CompiledText(stripped, [], removed, "none")
        kept: list[str] = []
        removed: list[str] = []

        def bracket(m: re.Match) -> str:
            if self._bracket_ok(m.group(1)):
                kept.append(f"[{m.group(1).strip()}]")
                return f"[{m.group(1).strip()}]"
            removed.append(m.group(0))
            return ""

        out = _BRACKET.sub(bracket, text)
        if self.dialect == "drama3":
            out = self._balance_paired(out, kept, removed)
        else:
            for m in _PAIRED.finditer(out):
                removed.append(m.group(0))
            out = _PAIRED.sub("", out)
        if self.dialect == "s1":
            # convert any surviving bracket cues to parentheses where S1 knows them
            def to_paren(m: re.Match) -> str:
                t = m.group(1).strip().lower()
                return f"({t})" if t in S1_TAGS else ""

            out = _BRACKET.sub(to_paren, out)
        out = re.sub(r"\s{2,}", " ", out).strip()
        return CompiledText(out, kept, removed, self.dialect)

    def _balance_paired(self, text: str, kept: list[str], removed: list[str]) -> str:
        stack: list[str] = []
        pieces: list[str] = []
        pos = 0
        for m in _PAIRED.finditer(text):
            pieces.append(text[pos : m.start()])
            pos = m.end()
            closing, tag = bool(m.group(1)), m.group(2).lower()
            if tag not in DRAMA3_SCOPED:
                removed.append(m.group(0))
                continue
            if not closing:
                if stack:  # no nesting: close the open one first
                    pieces.append(f"</{stack.pop()}>")
                stack.append(tag)
                pieces.append(f"<{tag}>")
                kept.append(f"<{tag}>")
            elif stack and stack[-1] == tag:
                stack.pop()
                pieces.append(f"</{tag}>")
            else:
                removed.append(m.group(0))
        pieces.append(text[pos:])
        while stack:
            pieces.append(f"</{stack.pop()}>")
        return "".join(pieces)

    # --- compilation -----------------------------------------------------

    def max_tags(self, text: str) -> int:
        if self.intensity <= 0 or self.frequency <= 0:
            return 0
        words = max(1, len(text.split()))
        return max(1, math.ceil(self.frequency * words / 10))

    def compile(self, text: str, *, emotion: str | None = None, delivery: str | None = None) -> CompiledText:
        """Apply emotion/delivery to *text* (which may already contain LLM markup)."""
        base = self.validate(text)
        if self.dialect == "none":
            return base
        budget = self.max_tags(base.text) - len(base.tags)
        out = base.text
        tags = list(base.tags)
        emo = (emotion or "neutral").lower().strip()
        if emo not in EMOTION_MAP:
            emo = "neutral"
        # low intensity only keeps strong, explicit events (laughs, whispers)
        if self.intensity < 0.34 and emo not in ("laughing", "whisper", "sighing"):
            emo = "neutral"
        if budget > 0 and emo != "neutral" and not base.tags:
            d3, s2, s1 = EMOTION_MAP[emo]
            if self.dialect == "drama3" and d3:
                if "{text}" in d3:
                    out = d3.replace("{text}", out)
                    tags.append(d3.split(">")[0] + ">")
                else:
                    out = f"{d3} {out}"
                    tags.append(d3)
                budget -= 1
            elif self.dialect == "s2" and s2:
                out = f"[{s2}] {out}"
                tags.append(f"[{s2}]")
                budget -= 1
            elif self.dialect == "s1" and s1:
                out = f"({s1}) {out}"
                tags.append(f"({s1})")
                budget -= 1
        if (
            budget > 0
            and delivery
            and self.dialect == "drama3"
            and self.allow_free_direction
            and self.intensity >= 0.5
            and not any(t.startswith("[") and t not in DRAMA3_POINT for t in tags)
        ):
            d = re.sub(r"[^a-zA-Z ,'-]", "", delivery).strip().lower()
            if d and len(d.split()) <= self.max_direction_words and self._bracket_ok(d):
                out = f"[{d}] {out}"
                tags.append(f"[{d}]")
        return CompiledText(out, tags, base.removed, self.dialect)


def dialect_for(backend_caps) -> str:
    return getattr(backend_caps, "dialect", "none") or "none"
