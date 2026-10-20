"""Research TTS engines: VoxCPM2, X-Voice (Stage 2), VoiceStar.

Each runs in its own Python environment (``AIVOICE_<ENGINE>_PYTHON`` or
``settings engines.<engine>.python``; default: aivoice's own interpreter).
Weights come from the Hugging Face cache through ``aivoice models download``.
The APIs used here were read from the released packages (voxcpm 2.0.3,
x-voice 0.1.1, voicestar 0.1.0), not inferred from the papers.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .base import (
    TASK_TTS,
    AudioResult,
    EngineCapabilities,
    EngineError,
    EngineUnavailable,
    SpeakerRef,
    SynthesisOptions,
    Timer,
    validate_options,
)
from .worker import WorkerClient, engine_python

VOXCPM2_LANGS = (
    "ar", "my", "zh", "da", "nl", "en", "fi", "fr", "de", "el", "he", "hi", "id", "it", "ja", "km",
    "ko", "lo", "ms", "no", "pl", "pt", "ru", "es", "sw", "sv", "tl", "th", "tr", "vi",
)


def _settings_get(key: str, default: Any = None) -> Any:
    from .. import settings

    v = settings.get(key)
    return default if v is None else v


def _snapshot(repo_id: str, manager: Any | None, key: str | None = None) -> Path | None:
    configured = _settings_get(f"engines.{key}.model_path") if key else None
    if configured:
        return Path(configured)
    try:
        if manager is None:
            from ..hub.manager import HuggingFaceModelManager

            manager = HuggingFaceModelManager()
        entry = manager.get_entry(repo_id)
    except Exception:  # noqa: BLE001
        entry = None
    return Path(entry["path"]) if entry and entry.get("path") else None


def _python_has(python: str, module: str) -> bool:
    import sys

    if python == sys.executable:
        import importlib.util

        return importlib.util.find_spec(module) is not None
    try:
        r = subprocess.run([python, "-c", f"import importlib.util,sys;sys.exit(0 if importlib.util.find_spec('{module}') else 1)"],
                           capture_output=True, timeout=30)
        return r.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _read_wav(path: str) -> tuple[np.ndarray, int]:
    from ..audio_io import read_wav

    return read_wav(Path(path))


class _WorkerEngine:
    handler = ""
    package = ""
    repo_id = ""
    engine_key = ""

    def __init__(self, *, model_path: str | Path | None = None, device: str | None = None,
                 manager: Any | None = None, python: str | None = None) -> None:
        self.model_path = Path(model_path) if model_path else None
        self.device = device
        self.manager = manager
        self.python = python or engine_python(self.engine_key)
        self.worker: WorkerClient | None = None
        self.sample_rate: int | None = None
        self.load_ms: float | None = None

    def _model_path(self) -> Path | None:
        return self.model_path or _snapshot(self.repo_id, self.manager, self.engine_key)

    def available(self) -> str | None:
        if not _python_has(self.python, self.package):
            return f"`{self.package}` is not installed for {self.python} (pip install {self.package} in a venv, then set AIVOICE_{self.engine_key.upper()}_PYTHON)"
        if self._needs_snapshot() and self._model_path() is None:
            return f"weights not installed: aivoice models download {self.repo_id} --yes"
        return None

    def _needs_snapshot(self) -> bool:
        return True

    def _load_request(self) -> dict[str, Any]:
        return {"model_path": str(self._model_path()), "device": self.device}

    def load(self) -> None:
        if self.worker is not None and self.worker.running:
            return
        why = self.available()
        if why:
            raise EngineUnavailable(f"{self.engine_key}: {why}")
        self.worker = WorkerClient(self.handler, python=self.python)
        resp = self.worker.call("load", **self._load_request())
        self.sample_rate = int(resp["sample_rate"])
        self.load_ms = resp.get("load_ms")

    def unload(self) -> None:
        if self.worker:
            self.worker.stop()
        self.worker = None


class VoxCPM2Engine(_WorkerEngine):
    handler = "voxcpm"
    package = "voxcpm"
    repo_id = "openbmb/VoxCPM2"
    engine_key = "voxcpm2"

    def __init__(self, *, optimize: bool = True, lora_path: str | None = None, **kw: Any) -> None:
        super().__init__(**kw)
        self.optimize = optimize
        self.lora_path = lora_path

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            name="voxcpm2",
            display_name="VoxCPM2 (OpenBMB)",
            tasks=(TASK_TTS,),
            zero_shot=True,
            transcript_free_prompt=True,
            accepts_prompt_transcript=True,
            cross_lingual=True,
            languages=VOXCPM2_LANGS,
            style_prompt=True,
            streaming=True,
            native_sample_rate=48000,
            devices=("cuda", "cpu"),
            est_vram_mib=8 * 1024,
            adaptation="lora",
            license="Apache-2.0",
            weights="openbmb/VoxCPM2",
            paper="arXiv:2606.06928 (VoxCPM2); VoxCPM tech report arXiv:2509.24650",
            install="pip install voxcpm (own venv recommended)",
            notes="Identity from reference audio (no transcript needed); style from a '(description)' instruction.",
            facts={"~8 GB VRAM": "official README", "48 kHz output": "official README; read at load from tts_model.sample_rate"},
        )

    def _load_request(self) -> dict[str, Any]:
        return {**super()._load_request(), "optimize": self.optimize, "lora_path": self.lora_path}

    def _request(self, text: str, speaker: SpeakerRef | None, o: SynthesisOptions) -> dict[str, Any]:
        style = ", ".join(x for x in (o.emotion, o.style_prompt) if x) or None
        req: dict[str, Any] = {"text": text, "style_prompt": style, "seed": o.seed, "steps": o.steps, "guidance": o.guidance}
        if speaker is not None and speaker.audio:
            req["reference_wav"] = str(speaker.audio)
            if o.use_prompt_transcript and speaker.transcript:
                # "ultimate cloning" (official): same clip as prompt + reference
                req["prompt_wav"] = str(speaker.audio)
                req["prompt_text"] = speaker.transcript
        return req

    def synthesize(self, text: str, speaker: SpeakerRef | None, options: SynthesisOptions) -> AudioResult:
        o, warnings = validate_options(self.capabilities(), speaker, options)
        self.load()
        t = Timer()
        resp = self.worker.synthesize(**self._request(text, speaker, o))  # type: ignore[union-attr]
        audio, sr = _read_wav(resp["path"])
        os.unlink(resp["path"])
        return AudioResult(audio, sr, "voxcpm2", native_sample_rate=int(resp["sample_rate"]),
                           first_audio_ms=t.elapsed_ms, total_ms=t.elapsed_ms, load_ms=self.load_ms,
                           warnings=warnings, params={k: v for k, v in self._request(text, speaker, o).items() if k != "text"})

    def synthesize_stream(self, text: str, speaker: SpeakerRef | None, options: SynthesisOptions) -> Iterator[tuple[np.ndarray, int]]:
        o, _ = validate_options(self.capabilities(), speaker, options)
        self.load()
        yield from self.worker.stream(**self._request(text, speaker, o))  # type: ignore[union-attr]


class VoiceStarEngine(_WorkerEngine):
    handler = "voicestar"
    package = "voicestar"
    repo_id = "pyp1/VoiceStar"
    engine_key = "voicestar"

    def __init__(self, *, model_name: str = "VoiceStar_840M_30s", **kw: Any) -> None:
        super().__init__(**kw)
        self.model_name = model_name

    def _needs_snapshot(self) -> bool:
        return False  # the package resolves pyp1/VoiceStar through the HF cache itself

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            name="voicestar",
            display_name="VoiceStar",
            tasks=(TASK_TTS,),
            zero_shot=True,
            accepts_prompt_transcript=True,
            requires_prompt_transcript=True,
            languages=("en",),
            speed_control=True,
            duration_control=True,
            native_sample_rate=16000,
            devices=("cuda", "cpu"),
            est_vram_mib=5 * 1024,
            license="code MIT, weights CC-BY-4.0",
            weights="pyp1/VoiceStar",
            paper="VoiceStar, Findings of ACL 2026 (2026.findings-acl.570)",
            install="pip install voicestar (own venv recommended)",
            notes="Native target-duration conditioning (PM-RoPE); long-form up to 30/40 s per model.",
        )

    def _load_request(self) -> dict[str, Any]:
        return {"model_name": self.model_name, "device": self.device}

    def plan_duration(self, text: str, speaker: SpeakerRef, o: SynthesisOptions) -> tuple[float, str]:
        if o.target_duration_s:
            return float(o.target_duration_s), "requested"
        from ..research.duration import plan_duration

        plan = plan_duration(text, speaker.audio, transcript=speaker.transcript, language=o.language or "en",
                             speed=o.speed or 1.0)
        return plan.seconds, plan.method

    def synthesize(self, text: str, speaker: SpeakerRef | None, options: SynthesisOptions) -> AudioResult:
        if speaker is None or speaker.audio is None:
            raise EngineError("VoiceStar clones from reference audio; pass a voice")
        o, warnings = validate_options(self.capabilities(), speaker, options)
        dur, method = self.plan_duration(text, speaker, o)
        self.load()
        t = Timer()
        resp = self.worker.synthesize(text=text, reference_wav=str(speaker.audio),  # type: ignore[union-attr]
                                      prompt_text=speaker.transcript, target_duration=dur, seed=o.seed)
        audio, sr = _read_wav(resp["path"])
        os.unlink(resp["path"])
        return AudioResult(audio, sr, "voicestar", native_sample_rate=16000, first_audio_ms=t.elapsed_ms,
                           total_ms=t.elapsed_ms, load_ms=self.load_ms, warnings=warnings,
                           params={"target_duration_s": round(dur, 3), "duration_method": method, "seed": o.seed})


class XVoiceEngine:
    """X-Voice Stage 2 (transcript-free) through the package's own CLI, one process per request."""

    repo_id = "XRXRX/X-Voice"

    def __init__(self, *, model_dir: str | Path | None = None, device: str | None = None,
                 manager: Any | None = None, python: str | None = None) -> None:
        self.model_dir = Path(model_dir) if model_dir else None
        self.device = device
        self.manager = manager
        self.python = python or engine_python("xvoice")

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            name="xvoice",
            display_name="X-Voice Stage 2",
            tasks=(TASK_TTS,),
            zero_shot=True,
            transcript_free_prompt=True,
            cross_lingual=True,
            speed_control=True,
            duration_control=True,
            native_sample_rate=24000,
            devices=("cuda", "cpu"),
            est_vram_mib=3 * 1024,
            license="code MIT; weights per model card",
            weights="XRXRX/X-Voice",
            paper="arXiv:2605.05611",
            install="pip install x-voice (own venv recommended; heavy dependencies)",
            notes="Drop-text cloning with the SRP speaking-rate predictor; IPA frontend; reloads per call (not real time).",
        )

    def _files(self) -> dict[str, Path]:
        root = self.model_dir or _snapshot(self.repo_id, self.manager, "xvoice")
        if root is None:
            raise EngineUnavailable(f"weights not installed: aivoice models download {self.repo_id} --yes")
        def pick(*pats: str) -> Path | None:
            for pat in pats:
                hits = sorted(root.rglob(pat))
                if hits:
                    return hits[-1]
            return None
        files = {
            "ckpt": pick("*stage2*.safetensors", "*Stage2*.safetensors", "*stage2*.pt"),
            "srp": pick("*srp*.safetensors", "*SRP*.safetensors", "*rate*.safetensors"),
            "vocab": pick("vocab.txt"),
        }
        missing = [k for k, v in files.items() if v is None]
        if missing:
            raise EngineUnavailable(f"X-Voice snapshot at {root} lacks {missing}; set engines.xvoice files explicitly")
        return files  # type: ignore[return-value]

    def available(self) -> str | None:
        if not _python_has(self.python, "x_voice"):
            return f"`x-voice` is not installed for {self.python} (set AIVOICE_XVOICE_PYTHON to its venv)"
        try:
            self._files()
        except EngineUnavailable as e:
            return str(e)
        return None

    def load(self) -> None:
        why = self.available()
        if why:
            raise EngineUnavailable(f"xvoice: {why}")

    def unload(self) -> None:
        pass

    def command(self, text: str, speaker: SpeakerRef, o: SynthesisOptions, out_dir: Path) -> list[str]:
        f = self._files()
        argv = [self.python, "-m", "x_voice.infer.infer_cli_stage2",
                "-p", str(f["ckpt"]), "--srp_ckpt_file", str(f["srp"]), "-v", str(f["vocab"]),
                "-r", str(speaker.audio), "-t", text, "-o", str(out_dir), "-w", "out.wav",
                "--auto_detect_lang", "--normalize_text", "--layered"]
        if o.language:
            argv += ["--gen_lang", o.language]
        if o.speed:
            argv += ["--speed", str(o.speed)]
        if o.target_duration_s:
            argv += ["--fix_duration", str(o.target_duration_s)]
        if o.steps:
            argv += ["--nfe_step", str(o.steps)]
        if o.guidance:
            argv += ["--cfg_strength", str(o.guidance)]
        if self.device:
            argv += ["--device", self.device]
        return argv

    def synthesize(self, text: str, speaker: SpeakerRef | None, options: SynthesisOptions) -> AudioResult:
        if speaker is None or speaker.audio is None:
            raise EngineError("X-Voice clones from reference audio; pass a voice")
        o, warnings = validate_options(self.capabilities(), speaker, options)
        self.load()
        t = Timer()
        with tempfile.TemporaryDirectory(prefix="aivoice-xvoice-") as td:
            r = subprocess.run(self.command(text, speaker, o, Path(td)), capture_output=True, text=True, timeout=1800)
            if r.returncode != 0:
                raise EngineError(f"X-Voice failed (exit {r.returncode}): {(r.stderr or r.stdout)[-1500:]}")
            audio, sr = _read_wav(str(Path(td) / "out.wav"))
        return AudioResult(audio, sr, "xvoice", native_sample_rate=24000, first_audio_ms=t.elapsed_ms,
                           total_ms=t.elapsed_ms, warnings=warnings + ["includes model load (one process per request)"],
                           params={"speed": o.speed, "fix_duration": o.target_duration_s})
