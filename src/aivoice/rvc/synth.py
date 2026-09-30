"""Synthesize a MeanVC2 reference clip from an RVC model (TTS -> RVC).

MeanVC2 is zero-shot and needs a recording of the target voice. Many RVC
packages ship no preview audio. When that happens the honest way to get one is to
let the RVC model itself speak: a text-to-speech engine reads a neutral passage,
and the RVC model converts that speech into its voice. The result is a genuine
sample of what the RVC voice sounds like - but it is *synthesized*, and every
voice built from it says so (``reference_kind = "synthesized"``).

Two pluggable steps, both optional dependencies:

* TTS: ``AIVOICE_TTS_CMD`` (a command template), Piper (``AIVOICE_PIPER_MODEL``),
  or espeak-ng.
* RVC inference: ``AIVOICE_RVC_CMD`` (a command template) or the ``rvc-python``
  package.

Command templates are split with :mod:`shlex` and run as an argument list, never
through a shell. Placeholders: TTS ``{text_file}`` ``{text}`` ``{out}``; RVC
``{input}`` ``{model}`` ``{index}`` ``{output}`` ``{version}``.

Security: an RVC ``.pth`` is a Python pickle. Running inference loads it, which
executes whatever the file contains. Only synthesize from models you trust.
"""

from __future__ import annotations

import importlib.util
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .detect import RvcPackageInfo

# Harvard sentences (a standard phonetically balanced set, ~20 s spoken).
DEFAULT_TEXT = (
    "The birch canoe slid on the smooth planks. "
    "Glue the sheet to the dark blue background. "
    "It's easy to tell the depth of a well. "
    "These days a chicken leg is a rare dish. "
    "Rice is often served in round bowls. "
    "The juice of lemons makes fine punch."
)

TTS_CMD_ENV = "AIVOICE_TTS_CMD"
PIPER_MODEL_ENV = "AIVOICE_PIPER_MODEL"
RVC_CMD_ENV = "AIVOICE_RVC_CMD"
NO_SYNTH_ENV = "AIVOICE_NO_SYNTH"

TTS_TIMEOUT_S = 180
RVC_TIMEOUT_S = 900
MIN_WAV_BYTES = 10_000

PICKLE_WARNING = (
    "RVC checkpoints are Python pickles: synthesizing loads the model file and runs "
    "any code inside it. Only do this for models you trust."
)


class SynthError(RuntimeError):
    """No reference could be synthesized (missing engines, or an engine failed)."""


def synthesis_disabled() -> bool:
    return os.environ.get(NO_SYNTH_ENV, "").strip().lower() in ("1", "true", "yes")


def render_command(template: str, values: dict[str, str]) -> list[str]:
    """Split *template* into an argv and substitute ``{name}`` placeholders per argument.

    Substitution happens after splitting, so a value containing spaces or shell
    metacharacters stays a single argument.
    """
    argv = shlex.split(template)
    if not argv:
        raise SynthError("empty command template")
    out = []
    for arg in argv:
        for key, val in values.items():
            arg = arg.replace("{" + key + "}", val)
        out.append(arg)
    return out


def _run(argv: list[str], *, stdin: str | None = None, timeout: int) -> None:
    try:
        r = subprocess.run(argv, input=stdin, capture_output=True, text=True, timeout=timeout, check=False)
    except FileNotFoundError as e:
        raise SynthError(f"{argv[0]}: not found") from e
    except subprocess.TimeoutExpired as e:
        raise SynthError(f"{argv[0]} timed out after {timeout}s") from e
    if r.returncode != 0:
        tail = (r.stderr or r.stdout or "").strip().splitlines()[-3:]
        raise SynthError(f"{argv[0]} failed (exit {r.returncode}): {' | '.join(tail)}")


def _check_wav(path: Path, what: str) -> None:
    if not path.is_file() or path.stat().st_size < MIN_WAV_BYTES:
        raise SynthError(f"{what} produced no usable audio ({path.name})")


# --- TTS engines ---------------------------------------------------------------


class TtsEngine(Protocol):
    name: str

    def unavailable_reason(self) -> str | None: ...

    def synthesize(self, text: str, out: Path) -> None: ...


@dataclass
class CommandTts:
    name: str = "command"
    template: str = ""

    def unavailable_reason(self) -> str | None:
        return None if self.template.strip() else f"{TTS_CMD_ENV} is not set"

    def synthesize(self, text: str, out: Path) -> None:
        text_file = out.with_suffix(".txt")
        text_file.write_text(text, encoding="utf-8")
        try:
            argv = render_command(self.template, {"text_file": str(text_file), "text": text, "out": str(out)})
            _run(argv, timeout=TTS_TIMEOUT_S)
        finally:
            text_file.unlink(missing_ok=True)
        _check_wav(out, "TTS command")


@dataclass
class PiperTts:
    name: str = "piper"
    model: str = ""

    def unavailable_reason(self) -> str | None:
        if not shutil.which("piper"):
            return "piper is not installed"
        if not self.model or not Path(self.model).is_file():
            return f"{PIPER_MODEL_ENV} must point to a Piper .onnx voice"
        return None

    def synthesize(self, text: str, out: Path) -> None:
        _run(["piper", "--model", self.model, "--output_file", str(out)], stdin=text, timeout=TTS_TIMEOUT_S)
        _check_wav(out, "piper")


@dataclass
class EspeakTts:
    name: str = "espeak-ng"

    def unavailable_reason(self) -> str | None:
        return None if shutil.which("espeak-ng") else "espeak-ng is not installed"

    def synthesize(self, text: str, out: Path) -> None:
        _run(["espeak-ng", "-v", "en-us", "-s", "150", "--stdin", "-w", str(out)], stdin=text, timeout=TTS_TIMEOUT_S)
        _check_wav(out, "espeak-ng")


def default_tts_engines() -> list[TtsEngine]:
    return [
        CommandTts(template=os.environ.get(TTS_CMD_ENV, "")),
        PiperTts(model=os.environ.get(PIPER_MODEL_ENV, "")),
        EspeakTts(),
    ]


# --- RVC inference engines -----------------------------------------------------


class RvcEngine(Protocol):
    name: str

    def unavailable_reason(self) -> str | None: ...

    def convert(self, src: Path, checkpoint: Path, index: Path | None, out: Path, version: str) -> None: ...


@dataclass
class CommandRvc:
    name: str = "command"
    template: str = ""

    def unavailable_reason(self) -> str | None:
        return None if self.template.strip() else f"{RVC_CMD_ENV} is not set"

    def convert(self, src: Path, checkpoint: Path, index: Path | None, out: Path, version: str) -> None:
        argv = render_command(
            self.template,
            {
                "input": str(src),
                "model": str(checkpoint),
                "index": str(index) if index else "",
                "output": str(out),
                "version": version,
            },
        )
        _run(argv, timeout=RVC_TIMEOUT_S)
        _check_wav(out, "RVC command")


@dataclass
class RvcPythonEngine:
    """The ``rvc-python`` package (``pip install rvc-python``)."""

    name: str = "rvc-python"
    device: str | None = None

    def unavailable_reason(self) -> str | None:
        if importlib.util.find_spec("rvc_python") is None:
            return "the rvc-python package is not installed (pip install rvc-python)"
        return None

    def _device(self) -> str:
        if self.device:
            return self.device
        try:
            import torch  # type: ignore[import-not-found]

            return "cuda:0" if torch.cuda.is_available() else "cpu:0"
        except ImportError:
            return "cpu:0"

    def convert(self, src: Path, checkpoint: Path, index: Path | None, out: Path, version: str) -> None:
        try:
            from rvc_python.infer import RVCInference  # type: ignore[import-not-found]
        except ImportError as e:
            raise SynthError(f"rvc-python could not be imported: {e}") from e
        try:
            rvc = RVCInference(device=self._device())
            rvc.load_model(str(checkpoint), version=version, index_path=str(index) if index else "")
            rvc.infer_file(str(src), str(out))
        except Exception as e:
            raise SynthError(f"rvc-python failed: {e}") from e
        _check_wav(out, "rvc-python")


def default_rvc_engines() -> list[RvcEngine]:
    return [CommandRvc(template=os.environ.get(RVC_CMD_ENV, "")), RvcPythonEngine()]


# --- pipeline ------------------------------------------------------------------


@dataclass
class SynthResult:
    path: Path
    tts_engine: str
    rvc_engine: str
    text: str
    notes: list[str] = field(default_factory=list)


def _first_available(engines: list, kind: str):
    reasons = []
    for eng in engines:
        why = eng.unavailable_reason()
        if why is None:
            return eng, reasons
        reasons.append(f"{eng.name}: {why}")
    raise SynthError(f"no {kind} engine available ({'; '.join(reasons)})")


def rvc_version(info: RvcPackageInfo) -> str:
    return "v1" if (info.generation or "").endswith("v1") else "v2"


def synthesize_reference(
    info: RvcPackageInfo,
    out_dir: Path,
    *,
    text: str | None = None,
    tts_engines: list[TtsEngine] | None = None,
    rvc_engines: list[RvcEngine] | None = None,
) -> SynthResult:
    """Write ``synthesized_reference.wav`` into *out_dir* using TTS then the RVC model."""
    if not info.checkpoint or not Path(info.checkpoint).is_file():
        raise SynthError("the package has no RVC checkpoint to synthesize with")
    text = (text or DEFAULT_TEXT).strip()
    if not text:
        raise SynthError("empty synthesis text")

    tts, _ = _first_available(tts_engines if tts_engines is not None else default_tts_engines(), "text-to-speech")
    rvc, _ = _first_available(rvc_engines if rvc_engines is not None else default_rvc_engines(), "RVC inference")

    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / "_tts_base.wav"
    final = out_dir / "synthesized_reference.wav"
    final.unlink(missing_ok=True)
    try:
        tts.synthesize(text, base)
        index = Path(info.index) if info.index and Path(info.index).is_file() else None
        rvc.convert(base, Path(info.checkpoint), index, final, rvc_version(info))
    finally:
        base.unlink(missing_ok=True)
    notes = [f"reference synthesized: {tts.name} TTS -> {rvc.name} (RVC {rvc_version(info)})"]
    if index is None:
        notes.append("no .index file; timbre retrieval was skipped")
    return SynthResult(path=final, tts_engine=tts.name, rvc_engine=rvc.name, text=text, notes=notes)


def engine_report() -> list[str]:
    """Human-readable availability lines for ``aivoice doctor``."""
    lines = []
    for kind, engines in (("TTS", default_tts_engines()), ("RVC", default_rvc_engines())):
        for eng in engines:
            why = eng.unavailable_reason()
            lines.append(f"{kind} {eng.name}: {'available' if why is None else 'missing - ' + why}")
    return lines


__all__ = [
    "DEFAULT_TEXT",
    "PICKLE_WARNING",
    "SynthError",
    "SynthResult",
    "engine_report",
    "render_command",
    "synthesis_disabled",
    "synthesize_reference",
]
