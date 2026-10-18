"""SpeechIntelligenceEngine: optional rewriting of recognised speech before synthesis.

The pipeline must work without it, so every path has a no-LLM answer:

* ``verbatim`` never calls a model;
* ``cleanup`` and ``hybrid`` start from a deterministic filler/repetition filter
  and only ask the model when one is loaded and the latency budget allows;
* any model failure - timeout, malformed JSON, a rewrite that drops or invents
  numbers or names - falls back to the (cleaned) original text.

The model answers with a JSON object (the internal contract below). That object
is never sent to the TTS: :class:`~aivoice.expression.ExpressionCompiler`
turns ``emotion``/``delivery`` into the backend's own markup.

``{"original": str, "transformed": str, "emotion": str, "delivery": str, "confidence": float}``
"""

from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from .expression import EMOTIONS

MODES = ("verbatim", "cleanup", "natural", "expressive", "character", "custom", "hybrid")

OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "transformed": {"type": "string"},
        "emotion": {"type": "string", "enum": list(EMOTIONS)},
        "delivery": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["transformed", "emotion", "delivery", "confidence"],
}

_MODE_INSTRUCTIONS = {
    "cleanup": "Remove filler words, false starts and stutters; fix grammar. Do not change meaning, tone or wording otherwise.",
    "natural": "Make it flow like natural conversational speech. Keep the meaning and every fact.",
    "expressive": "Keep the words (light cleanup only). Choose the emotion and a short delivery direction that fit.",
    "character": "Rephrase as the character described below would say it. Keep every fact, name and number.",
    "custom": "Follow the custom instruction below.",
    "hybrid": "Light cleanup only, then pick emotion and delivery. Be brief.",
}

SYSTEM_PREFIX = (
    "You rewrite one spoken sentence for a text-to-speech voice. "
    "Reply with JSON only: {\"transformed\": str, \"emotion\": one of "
    + ", ".join(EMOTIONS)
    + ", \"delivery\": short phrase (max 6 words), \"confidence\": 0..1}. "
    "Never add facts, names, numbers, or statements the speaker did not say."
)

FILLERS = re.compile(r"\b(?:um+|uh+|erm+|er|ah+|hmm+|mm+|you know|i mean)\b[,.]?\s*", re.I)
STUTTER = re.compile(r"\b(\w+)(?:[\s,]+\1\b)+", re.I)
NUMBER = re.compile(r"\d+(?:[.,:]\d+)*")


@dataclass
class TransformResult:
    original: str
    transformed: str
    emotion: str = "neutral"
    delivery: str = ""
    confidence: float = 1.0
    mode: str = "verbatim"
    used_llm: bool = False
    fallback_reason: str | None = None
    latency_ms: float = 0.0
    raw: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LLMBackend(Protocol):
    name: str

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int,
        temperature: float,
        schema: dict[str, Any] | None,
        cancel: threading.Event,
    ) -> str: ...


def rule_cleanup(text: str) -> str:
    out = FILLERS.sub("", text)
    out = STUTTER.sub(r"\1", out)
    out = re.sub(r"\s{2,}", " ", out).strip(" ,")
    if out and text[:1].isupper() and not out[:1].isupper():
        out = out[0].upper() + out[1:]
    return out or text.strip()


def _proper_nouns(text: str) -> set[str]:
    words = re.findall(r"[A-Za-z][\w'-]*", text)
    out = set()
    for i, w in enumerate(words):
        if w[0].isupper() and i > 0 and w.lower() not in ("i", "i'm", "i've", "i'll", "i'd"):
            prev = text[: text.find(w)].rstrip()
            if not prev.endswith((".", "!", "?")):
                out.add(w.lower())
    return out


def fact_check(original: str, transformed: str, *, allow_new_names: bool = False) -> str | None:
    """None when the rewrite keeps the facts; else why it must be rejected."""
    orig_nums = NUMBER.findall(original)
    new_nums = NUMBER.findall(transformed)
    for n in orig_nums:
        if n not in transformed:
            return f"dropped number {n!r}"
    for n in new_nums:
        if n not in original:
            return f"invented number {n!r}"
    names = _proper_nouns(original)
    low = transformed.lower()
    for name in names:
        if name not in low:
            return f"dropped name {name!r}"
    if not allow_new_names:
        added = _proper_nouns(transformed) - names - {w.lower() for w in original.split()}
        if added:
            return f"introduced names {sorted(added)}"
    return None


def parse_response(raw: str) -> dict[str, Any]:
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s)
    m = re.search(r"\{.*\}", s, re.S)
    if not m:
        raise ValueError("no JSON object in model output")
    data = json.loads(m.group(0))
    if not isinstance(data, dict):
        raise ValueError("model output is not a JSON object")
    t = data.get("transformed")
    if not isinstance(t, str) or not t.strip():
        raise ValueError("missing 'transformed'")
    emo = str(data.get("emotion", "neutral")).lower().strip()
    data["emotion"] = emo if emo in EMOTIONS else "neutral"
    data["delivery"] = str(data.get("delivery", ""))[:60]
    try:
        data["confidence"] = max(0.0, min(1.0, float(data.get("confidence", 0.5))))
    except (TypeError, ValueError):
        data["confidence"] = 0.5
    return data


class LlamaCppBackend:
    """llama-cpp-python with a persistent context; the constant system prompt prefix is reused by llama.cpp."""

    def __init__(
        self,
        model_path: str,
        *,
        n_ctx: int = 1024,
        n_threads: int | None = None,
        n_gpu_layers: int = 0,
        no_think: bool = False,
    ) -> None:
        try:
            from llama_cpp import Llama  # type: ignore
        except ImportError as e:
            raise RuntimeError("llama-cpp-python is not installed: pip install 'aivoice[llm]'") from e
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        self.name = f"llama.cpp:{model_path.rsplit('/', 1)[-1]}"
        self.no_think = no_think
        self._lock = threading.Lock()

    def complete(self, system, user, *, max_tokens, temperature, schema, cancel) -> str:
        from llama_cpp import StoppingCriteriaList  # type: ignore

        if self.no_think:
            user = user + " /no_think"
        stop = StoppingCriteriaList([lambda tokens, logits: cancel.is_set()])
        kw: dict[str, Any] = dict(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=temperature,
            stopping_criteria=stop,
        )
        if schema:
            kw["response_format"] = {"type": "json_object", "schema": schema}
        with self._lock:
            out = self.llm.create_chat_completion(**kw)
        return out["choices"][0]["message"]["content"] or ""

    def close(self) -> None:
        try:
            self.llm.close()
        except Exception:  # noqa: BLE001
            pass


@dataclass
class EngineStats:
    calls: int = 0
    timeouts: int = 0
    rejected: int = 0
    errors: int = 0
    latencies_ms: list[float] = field(default_factory=list)


class SpeechIntelligenceEngine:
    def __init__(
        self,
        backend: LLMBackend | None = None,
        *,
        mode: str = "verbatim",
        timeout_ms: int = 350,
        max_tokens: int = 96,
        creativity: float = 0.3,
        intensity: float = 0.5,
        character: str = "",
        custom_prompt: str = "",
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.backend = backend
        self.mode = mode
        self.timeout_ms = timeout_ms
        self.max_tokens = max_tokens
        self.creativity = creativity
        self.intensity = intensity
        self.character = character
        self.custom_prompt = custom_prompt
        self.stats = EngineStats()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="aivoice-llm")
        self._inflight: tuple[Future, threading.Event] | None = None

    @property
    def enabled(self) -> bool:
        return self.mode != "verbatim"

    def system_prompt(self) -> str:
        parts = [SYSTEM_PREFIX, _MODE_INSTRUCTIONS.get(self.mode, "")]
        if self.mode == "character" and self.character:
            parts.append(f"Character: {self.character.strip()[:600]}")
        if self.mode == "custom" and self.custom_prompt:
            parts.append(f"Instruction: {self.custom_prompt.strip()[:600]}")
        parts.append(f"Transformation intensity: {self.intensity:.1f} (0 = minimal, 1 = strong).")
        return " ".join(p for p in parts if p)

    def transform(self, text: str) -> TransformResult:
        t0 = time.perf_counter()
        original = text.strip()
        if not original or self.mode == "verbatim":
            return TransformResult(original, original, mode="verbatim")
        cleaned = rule_cleanup(original) if self.mode in ("cleanup", "hybrid", "expressive") else original
        if self.backend is None:
            return TransformResult(
                original, cleaned, mode=self.mode, fallback_reason="no LLM loaded (rule-based cleanup only)",
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        if self._inflight and not self._inflight[0].done():
            # the last call is still being cancelled; do not queue behind it
            return TransformResult(original, cleaned, mode=self.mode, fallback_reason="LLM busy",
                                   latency_ms=(time.perf_counter() - t0) * 1000)
        self.stats.calls += 1
        cancel = threading.Event()
        temperature = 0.1 + 0.8 * self.creativity
        fut = self._pool.submit(
            self.backend.complete,
            self.system_prompt(),
            cleaned if self.mode in ("cleanup", "hybrid") else original,
            max_tokens=self.max_tokens,
            temperature=temperature,
            schema=OUTPUT_SCHEMA,
            cancel=cancel,
        )
        self._inflight = (fut, cancel)
        remaining = self.timeout_ms / 1000 - (time.perf_counter() - t0)
        try:
            raw = fut.result(timeout=max(0.0, remaining))
        except FutureTimeout:
            cancel.set()
            self.stats.timeouts += 1
            return TransformResult(original, cleaned, mode=self.mode, fallback_reason=f"LLM exceeded {self.timeout_ms} ms",
                                   latency_ms=(time.perf_counter() - t0) * 1000)
        except Exception as e:  # noqa: BLE001
            self.stats.errors += 1
            return TransformResult(original, cleaned, mode=self.mode, fallback_reason=f"LLM error: {e}",
                                   latency_ms=(time.perf_counter() - t0) * 1000)
        ms = (time.perf_counter() - t0) * 1000
        self.stats.latencies_ms.append(ms)
        try:
            data = parse_response(raw)
        except (ValueError, json.JSONDecodeError) as e:
            self.stats.rejected += 1
            return TransformResult(original, cleaned, mode=self.mode, fallback_reason=f"malformed output: {e}", latency_ms=ms, raw=raw)
        transformed = data["transformed"].strip()
        why = fact_check(original, transformed, allow_new_names=False)
        if why is None and self.mode in ("cleanup", "hybrid", "expressive") and len(transformed) > 1.6 * len(original) + 20:
            why = "rewrite much longer than the original"
        if why:
            self.stats.rejected += 1
            return TransformResult(original, cleaned, emotion=data["emotion"], delivery=data["delivery"],
                                   confidence=data["confidence"], mode=self.mode, fallback_reason=f"rejected: {why}",
                                   latency_ms=ms, raw=raw)
        if self.mode == "expressive":
            transformed = cleaned  # expression-only: wording stays the speaker's
        return TransformResult(original, transformed, emotion=data["emotion"], delivery=data["delivery"],
                               confidence=data["confidence"], mode=self.mode, used_llm=True, latency_ms=ms, raw=raw)

    def close(self) -> None:
        if self._inflight:
            self._inflight[1].set()
        self._pool.shutdown(wait=False)
        close = getattr(self.backend, "close", None)
        if close:
            close()


class ScriptedLLM:
    """Test double: returns canned output after an optional delay."""

    def __init__(self, reply: str | None = None, *, delay_ms: float = 0.0, fn=None) -> None:
        self.reply = reply
        self.delay_ms = delay_ms
        self.fn = fn
        self.name = "scripted"
        self.calls: list[str] = []

    def complete(self, system, user, *, max_tokens, temperature, schema, cancel) -> str:
        self.calls.append(user)
        end = time.monotonic() + self.delay_ms / 1000
        while time.monotonic() < end:
            if cancel.is_set():
                return ""
            time.sleep(0.005)
        if self.fn:
            return self.fn(user)
        return self.reply or json.dumps({"transformed": user, "emotion": "neutral", "delivery": "", "confidence": 0.9})


def resolve_model_path(name: str | None, manager: Any | None = None) -> str | None:
    """A GGUF file path from a direct path or a registered model."""
    from pathlib import Path

    if not name:
        return None
    p = Path(name).expanduser()
    if p.is_file():
        return str(p)
    if manager is not None:
        entry = manager.get_entry(name)
        if entry:
            if entry.get("file"):
                return entry["file"]
            ggufs = sorted(Path(entry["path"]).rglob("*.gguf"))
            if ggufs:
                return str(ggufs[0])
    return None
