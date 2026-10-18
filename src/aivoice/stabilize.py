"""Transcript commitment: decide which recognised words are safe to speak.

Streaming ASR revises itself: "I scream" becomes "ice cream" a pass later.
Speaking a partial that is then revised cannot be undone, so text is only
*committed* once it is stable:

* **agreement** - a word prefix must be identical (after normalising case and
  punctuation) in the last ``agreement`` hypotheses (LocalAgreement-n);
* **hold-back** - the last word of a hypothesis is never committed from a
  partial unless punctuation closes it (it is the word most likely to change);
* **chunking** - a commit needs ``min_commit_words`` stable words or a clause
  boundary, so TTS gets phrases with natural prosody, not single words;
* **confidence** - when the backend reports per-word confidence, a word under
  ``min_confidence`` blocks the commit at that point.

At the end of the utterance :meth:`finalize` returns whatever the final
transcript adds beyond what was committed, aligned so committed words are never
repeated even when the final transcript differs from the partials.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

_PUNCT_END = re.compile(r"[.!?;:,…]$")
_NORM = re.compile(r"[^\w']+", re.UNICODE)


def norm(word: str) -> str:
    return _NORM.sub("", word.lower())


def words_of(text: str) -> list[str]:
    return text.split()


@dataclass
class Commit:
    text: str
    start_word: int
    end_word: int
    final: bool = False
    revision: int = 0


@dataclass
class StabilizerStats:
    partials: int = 0
    commits: int = 0
    committed_words: int = 0
    divergences: int = 0  # final transcript disagreed with already-committed text
    history: list[str] = field(default_factory=list)


class TranscriptStabilizer:
    def __init__(
        self,
        *,
        agreement: int = 2,
        min_commit_words: int = 3,
        commit_on_punctuation: bool = True,
        min_confidence: float | None = None,
        speculative: bool = True,
        max_history: int = 8,
    ) -> None:
        self.agreement = max(1, agreement)
        self.min_commit_words = max(1, min_commit_words)
        self.commit_on_punctuation = commit_on_punctuation
        self.min_confidence = min_confidence
        self.speculative = speculative
        self.max_history = max_history
        self.reset()

    def reset(self) -> None:
        self._hyps: list[list[str]] = []
        self._committed: list[str] = []  # surface words already handed to TTS
        self.stats = StabilizerStats()
        self.latest = ""

    @property
    def committed_text(self) -> str:
        return " ".join(self._committed)

    @property
    def pending_text(self) -> str:
        """Latest hypothesis beyond the committed prefix (display only)."""
        words = words_of(self.latest)
        return " ".join(words[len(self._committed):])

    def _stable_prefix_len(self) -> int:
        if len(self._hyps) < self.agreement:
            return 0
        recent = self._hyps[-self.agreement:]
        n = min(len(h) for h in recent)
        i = 0
        while i < n and len({norm(h[i]) for h in recent}) == 1:
            i += 1
        return i

    def update(self, text: str, *, confidences: list[float] | None = None, revision: int = 0) -> list[Commit]:
        """Feed a partial hypothesis; returns newly committed phrases (speculative mode)."""
        self.stats.partials += 1
        self.latest = text
        words = words_of(text)
        self._hyps.append(words)
        if len(self._hyps) > self.max_history:
            self._hyps.pop(0)
        if not self.speculative:
            return []
        # Committed words must still be what the hypothesis says; if ASR revised
        # inside the committed span we stop committing until the final.
        for i, w in enumerate(self._committed):
            if i >= len(words) or norm(words[i]) != norm(w):
                return []
        stable = self._stable_prefix_len()
        latest_words = words
        # hold back the trailing word unless punctuation closes it
        if stable == len(latest_words) and stable and not _PUNCT_END.search(latest_words[stable - 1]):
            stable -= 1
        if confidences and self.min_confidence is not None:
            for i in range(len(self._committed), min(stable, len(confidences))):
                if confidences[i] < self.min_confidence:
                    stable = i
                    break
        start = len(self._committed)
        if stable <= start:
            return []
        cut = None
        if self.commit_on_punctuation:
            for i in range(stable, start, -1):
                if _PUNCT_END.search(latest_words[i - 1]):
                    cut = i
                    break
        if cut is None and stable - start >= self.min_commit_words:
            cut = stable
        if cut is None:
            return []
        phrase = latest_words[start:cut]
        self._committed.extend(phrase)
        self.stats.commits += 1
        self.stats.committed_words += len(phrase)
        c = Commit(" ".join(phrase), start, cut, final=False, revision=revision)
        self.stats.history.append(c.text)
        return [c]

    def finalize(self, final_text: str) -> Commit | None:
        """Remainder of the final transcript after the committed words (never repeats them)."""
        final = words_of(final_text)
        done = self._committed
        if not done:
            rest = final
            start = 0
        else:
            fn = [norm(w) for w in final]
            dn = [norm(w) for w in done]
            if fn[: len(dn)] == dn:
                start = len(dn)
            else:
                self.stats.divergences += 1
                sm = difflib.SequenceMatcher(a=dn, b=fn, autojunk=False)
                blocks = [b for b in sm.get_matching_blocks() if b.size]
                # continue after the last final-transcript word that matched committed text
                start = (blocks[-1].b + blocks[-1].size) if blocks else len(final)
                start = max(start, 0)
            rest = final[start:]
        self._committed.extend(rest)
        result = None
        if rest:
            self.stats.commits += 1
            self.stats.committed_words += len(rest)
            result = Commit(" ".join(rest), start, start + len(rest), final=True)
            self.stats.history.append(result.text)
        self._hyps.clear()
        self.latest = final_text
        return result
