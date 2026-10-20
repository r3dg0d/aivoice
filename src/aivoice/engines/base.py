"""Engine interfaces: text-to-speech and voice conversion are different tasks.

* :class:`TTSEngine` turns *text* into speech in a target voice.
* :class:`VoiceConversionEngine` turns *source speech* into the target voice.

An engine states what it can do in :class:`EngineCapabilities`. Every request is
checked against those capabilities by :func:`validate_options` before it runs:
an option the engine cannot honour is an error (or, with ``allow_unsupported``,
a recorded warning and the option is dropped). Nothing is silently ignored.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Protocol, runtime_checkable

import numpy as np

TASK_TTS = "tts"
TASK_VC = "vc"


class EngineError(RuntimeError):
    pass


class EngineUnavailable(EngineError):
    """Runtime, weights or hardware missing. The message says what to install."""


class UnsupportedOption(EngineError):
    pass


@dataclass(frozen=True)
class EngineCapabilities:
    name: str
    display_name: str
    tasks: tuple[str, ...]
    zero_shot: bool = False  # clones an unseen voice from reference audio, no training
    transcript_free_prompt: bool = False  # reference audio needs no transcript (native)
    accepts_prompt_transcript: bool = False  # can use a transcript when given
    requires_prompt_transcript: bool = False
    cross_lingual: bool = False
    languages: tuple[str, ...] = ()  # ISO codes; empty = unspecified
    style_prompt: bool = False  # natural-language delivery instruction
    style_reference: bool = False  # a second audio reference for style (separate from identity)
    emotion_tags: bool = False
    speed_control: bool = False
    duration_control: bool = False  # exact target duration conditioning
    streaming: bool = False
    native_sample_rate: int | None = None
    precisions: tuple[str, ...] = ()
    devices: tuple[str, ...] = ("cpu",)
    est_vram_mib: int | None = None
    local: bool = True
    remote: bool = False
    adaptation: str | None = None  # e.g. "lora" when the engine supports speaker fine-tuning
    max_reference_s: float | None = None
    license: str = ""
    weights: str = ""
    paper: str = ""
    install: str = ""
    notes: str = ""
    facts: dict[str, str] = field(default_factory=dict, hash=False, compare=False)

    def supports(self, task: str) -> bool:
        return task in self.tasks

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SpeakerRef:
    """What an engine needs to condition on one voice."""

    name: str
    audio: Path | None = None  # identity reference (prepared copy when available)
    transcript: str | None = None
    language: str | None = None
    engine_ids: dict[str, str] = field(default_factory=dict)  # e.g. {"fish-server": "reference_id"}
    profile_id: str | None = None


@dataclass
class SynthesisOptions:
    language: str | None = None
    style_prompt: str | None = None
    style_reference: Path | None = None
    emotion: str | None = None
    speed: float | None = None  # 1.0 = natural
    target_duration_s: float | None = None
    pitch_shift: float | None = None
    identity_strength: float | None = None
    seed: int | None = None
    steps: int | None = None
    guidance: float | None = None
    use_prompt_transcript: bool = True  # pass the profile transcript when the engine can use it
    allow_unsupported: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversionOptions:
    mode: str = "balanced"
    device: str = "cuda"
    allow_unsupported: bool = False


@dataclass
class AudioResult:
    audio: np.ndarray
    sample_rate: int
    engine: str
    native_sample_rate: int | None = None  # rate the model produced before any resampling
    first_audio_ms: float | None = None
    total_ms: float | None = None
    load_ms: float | None = None
    warnings: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return self.audio.size / self.sample_rate if self.sample_rate else 0.0

    @property
    def rtf(self) -> float | None:
        if self.total_ms is None or not self.duration_s:
            return None
        return (self.total_ms / 1000) / self.duration_s


@runtime_checkable
class VoiceEngine(Protocol):
    def capabilities(self) -> EngineCapabilities: ...

    def available(self) -> str | None:
        """None when usable now, else why not."""

    def load(self) -> None: ...

    def unload(self) -> None: ...


@runtime_checkable
class TTSEngine(VoiceEngine, Protocol):
    def synthesize(self, text: str, speaker: SpeakerRef | None, options: SynthesisOptions) -> AudioResult: ...


@runtime_checkable
class StreamingTTSEngine(TTSEngine, Protocol):
    def synthesize_stream(
        self, text: str, speaker: SpeakerRef | None, options: SynthesisOptions
    ) -> Iterator[tuple[np.ndarray, int]]: ...


@runtime_checkable
class VoiceConversionEngine(VoiceEngine, Protocol):
    def convert(self, source: Path, target: SpeakerRef, options: ConversionOptions) -> AudioResult: ...


@dataclass
class OptionIssue:
    option: str
    message: str
    fatal: bool


def validate_options(
    caps: EngineCapabilities, speaker: SpeakerRef | None, opts: SynthesisOptions
) -> tuple[SynthesisOptions, list[str]]:
    """Check a request against capabilities. Raises UnsupportedOption, or returns (cleaned options, warnings)."""
    issues: list[OptionIssue] = []

    def bad(option: str, msg: str) -> None:
        issues.append(OptionIssue(option, msg, fatal=not opts.allow_unsupported))

    if not caps.supports(TASK_TTS):
        raise UnsupportedOption(f"{caps.name} is not a text-to-speech engine (tasks: {', '.join(caps.tasks)})")
    if speaker is not None and speaker.audio is not None and not caps.zero_shot:
        bad("speaker", f"{caps.name} cannot clone a voice from reference audio")
    if speaker is not None and caps.requires_prompt_transcript and not speaker.transcript:
        issues.append(
            OptionIssue(
                "transcript",
                f"{caps.name} needs the exact transcript of the reference audio "
                "(set it with `aivoice clone create --transcript ...`)",
                fatal=True,
            )
        )
    if opts.style_prompt and not caps.style_prompt:
        bad("style_prompt", f"{caps.name} has no natural-language style control")
    if opts.style_reference and not caps.style_reference:
        bad(
            "style_reference",
            f"{caps.name} cannot take a separate style reference; identity and style come from one prompt. "
            "aivoice does not blend embeddings from unrelated encoders to fake this.",
        )
    if opts.emotion and not (caps.emotion_tags or caps.style_prompt):
        bad("emotion", f"{caps.name} has no emotion control")
    if opts.speed not in (None, 1.0) and not (caps.speed_control or caps.duration_control):
        bad("speed", f"{caps.name} has no speaking-rate control")
    if opts.target_duration_s is not None and not caps.duration_control:
        bad("target_duration_s", f"{caps.name} has no duration conditioning")
    if opts.pitch_shift not in (None, 0.0):
        bad("pitch_shift", "no engine in aivoice implements pitch shifting yet")
    if opts.identity_strength is not None:
        bad("identity_strength", f"{caps.name} exposes no identity-strength parameter")
    lang = (opts.language or "").lower()
    if lang and caps.languages and lang not in caps.languages:
        bad("language", f"{caps.name} does not list language {lang!r} ({', '.join(caps.languages)})")
    fatal = [i for i in issues if i.fatal]
    if fatal:
        raise UnsupportedOption("; ".join(i.message for i in fatal))
    cleaned = SynthesisOptions(**{**opts.__dict__})
    warnings = []
    for i in issues:
        warnings.append(f"ignored {i.option}: {i.message}")
        if hasattr(cleaned, i.option):
            setattr(cleaned, i.option, None)
    return cleaned, warnings


class Timer:
    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self.first: float | None = None

    def mark_first(self) -> None:
        if self.first is None:
            self.first = (time.perf_counter() - self.t0) * 1000

    @property
    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.t0) * 1000
