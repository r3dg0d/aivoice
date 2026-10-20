"""Engines aivoice already had, behind the new interfaces: MeanVC2 (VC), Fish (TTS), espeak-ng (TTS)."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from ..audio_io import read_wav
from .base import (
    TASK_TTS,
    TASK_VC,
    AudioResult,
    ConversionOptions,
    EngineCapabilities,
    EngineError,
    EngineUnavailable,
    SpeakerRef,
    SynthesisOptions,
    Timer,
    validate_options,
)


class MeanVC2Engine:
    """Zero-shot streaming voice conversion (the existing MeanVC2 integration)."""

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            name="meanvc2",
            display_name="MeanVC2",
            tasks=(TASK_VC,),
            zero_shot=True,
            transcript_free_prompt=True,
            cross_lingual=True,
            streaming=True,
            native_sample_rate=16000,
            devices=("cuda", "cpu"),
            license="code: README claims Apache-2.0 (LICENSE file missing upstream, see NOTICE); weights Apache-2.0 claimed",
            weights="ASLP-lab/MeanVC2",
            paper="arXiv:2606.09050",
            install="aivoice models install meanvc2 --yes",
            notes="Voice conversion: needs source speech, not text. Live path: `aivoice live` / `virtualmic`.",
        )

    def available(self) -> str | None:
        from ..models import is_ready
        from ..paths import vendor_dir

        if not is_ready("meanvc2"):
            return "not installed: aivoice models install meanvc2 --yes"
        if not (vendor_dir() / "src" / "infer" / "infer_e2e.py").is_file() and not (vendor_dir() / "runtime" / "run_rt.py").is_file():
            return f"MeanVC2 vendor checkout incomplete at {vendor_dir()}"
        return None

    def load(self) -> None:
        why = self.available()
        if why:
            raise EngineUnavailable(f"meanvc2: {why}")

    def unload(self) -> None:
        pass

    def convert(self, source: Path, target: SpeakerRef, options: ConversionOptions) -> AudioResult:
        if target.audio is None:
            raise EngineError("MeanVC2 needs target reference audio")
        self.load()
        from ..backend.meanvc2 import MeanVC2Backend
        from ..presets import get_mode

        be = MeanVC2Backend(get_mode(options.mode), device=options.device)
        be.set_reference(Path(target.audio))
        t = Timer()
        with tempfile.TemporaryDirectory(prefix="aivoice-vc-") as td:
            out = Path(td) / "out.wav"
            be.convert_file(Path(source), out)
            audio, sr = read_wav(out)
        return AudioResult(audio, sr, "meanvc2", native_sample_rate=16000, total_ms=t.elapsed_ms,
                           params={"mode": options.mode, "note": "offline file conversion (includes process start)"})


class EspeakEngine:
    """espeak-ng: formant TTS, no cloning. Used as a neutral source voice and a fallback."""

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            name="espeak",
            display_name="espeak-ng",
            tasks=(TASK_TTS,),
            zero_shot=False,
            speed_control=True,
            native_sample_rate=22050,
            license="GPL-3.0 (program, run as a separate process)",
            install="system package espeak-ng",
            notes="Not a cloning engine. Voices are synthetic (e.g. en-us+m3), so samples imitate no one.",
        )

    def available(self) -> str | None:
        return None if shutil.which("espeak-ng") else "espeak-ng is not installed"

    def load(self) -> None:
        if self.available():
            raise EngineUnavailable("espeak-ng is not installed")

    def unload(self) -> None:
        pass

    def synthesize(self, text: str, speaker: SpeakerRef | None, options: SynthesisOptions) -> AudioResult:
        o, warnings = validate_options(self.capabilities(), speaker, options)  # refuses reference audio: not a cloner
        self.load()
        voice = str(o.extra.get("voice") or ("en-us" if not o.language else o.language))
        wpm = int(round(175 * (o.speed or 1.0)))
        pitch = o.extra.get("pitch")
        t = Timer()
        with tempfile.TemporaryDirectory(prefix="aivoice-espeak-") as td:
            out = Path(td) / "out.wav"
            argv = ["espeak-ng", "-v", voice, "-s", str(wpm), "-w", str(out)]
            if pitch is not None:
                argv += ["-p", str(int(pitch))]
            r = subprocess.run(argv + [text], capture_output=True, text=True, timeout=120)
            if r.returncode != 0:
                raise EngineError(f"espeak-ng failed: {r.stderr.strip()[:300]}")
            audio, sr = read_wav(out)
        return AudioResult(audio, sr, "espeak", native_sample_rate=sr, first_audio_ms=t.elapsed_ms,
                           total_ms=t.elapsed_ms, warnings=warnings, params={"voice": voice, "wpm": wpm, "pitch": pitch})


class _ProfileShim:
    """Duck-types the TTS voice profile that tts.fish.build_request expects."""

    def __init__(self, speaker: SpeakerRef, kind: str) -> None:
        self.name = speaker.name
        self.kind = kind
        self.reference_text = speaker.transcript
        self.reference_id = speaker.engine_ids.get("fish")
        self._audio = speaker.audio

    def reference_bytes(self) -> bytes:
        return Path(self._audio).read_bytes()  # type: ignore[arg-type]


class FishEngine:
    """fish-speech server (Drama 3 / S2-Pro) or the hosted Fish API, as a TTS engine."""

    def __init__(self, mode: str = "server") -> None:
        self.mode = mode
        self._backend: Any = None

    def _cfg(self) -> dict[str, Any]:
        from .. import settings

        return settings.load()["tts"]

    def _dialect(self) -> str:
        if self.mode == "api":
            return "drama3" if self._cfg().get("api_model") == "drama-3" else "s2"
        return "drama3" if self._cfg().get("server_model") == "drama-3" else "s2"

    def capabilities(self) -> EngineCapabilities:
        drama = self._dialect() == "drama3"
        return EngineCapabilities(
            name="fish-api" if self.mode == "api" else "fish-server",
            display_name="Fish Audio API" if self.mode == "api" else "fish-speech server (Drama 3 / S2-Pro)",
            tasks=(TASK_TTS,),
            zero_shot=True,
            accepts_prompt_transcript=True,
            requires_prompt_transcript=True,
            cross_lingual=True,
            style_prompt=drama,  # Drama 3 accepts free-form bracket direction
            emotion_tags=True,
            streaming=True,
            native_sample_rate=44100,
            devices=("server",),
            local=self.mode == "server",
            remote=self.mode == "api",
            license="Drama 3: Fish Audio Community License v1.0; S2-Pro: see model card",
            weights="fishaudio/drama-3 | fishaudio/s2-pro",
            paper="Fish Audio Drama 3 technical report, arXiv:2610.18010",
            install="run fish-speech api_server (docs/drama3.md)",
            notes="Drama 3 needs >= 48 GB GPU memory; S2-Pro is the local option on a 24 GB card.",
        )

    def available(self) -> str | None:
        try:
            self._get()
        except Exception as e:  # noqa: BLE001
            return str(e)
        if self.mode == "server":
            ok, detail = self._backend.health()
            return None if ok else f"server not answering: {detail}"
        return None

    def _get(self):
        if self._backend is None:
            from ..tts.fish import FishAPIBackend, FishServerBackend

            c = self._cfg()
            if self.mode == "api":
                self._backend = FishAPIBackend(model=c.get("api_model", "drama-3"), latency=c.get("latency", "balanced"))
            else:
                self._backend = FishServerBackend(c.get("server_url", "http://127.0.0.1:8080"), model=c.get("server_model", "auto"))
        return self._backend

    def load(self) -> None:
        why = self.available()
        if why:
            raise EngineUnavailable(f"{self.capabilities().name}: {why}")

    def unload(self) -> None:
        self._backend = None

    def _text(self, text: str, o: SynthesisOptions) -> str:
        from ..expression import ExpressionCompiler

        comp = ExpressionCompiler(self._dialect(), intensity=0.6)
        out = comp.compile(text, emotion=o.emotion).text
        if o.style_prompt and self._dialect() == "drama3":
            out = f"[{o.style_prompt.strip().rstrip('.')}] {out}"
        return out

    def synthesize_stream(self, text: str, speaker: SpeakerRef | None, options: SynthesisOptions) -> Iterator[tuple[np.ndarray, int]]:
        o, _ = validate_options(self.capabilities(), speaker, options)
        voice = None
        if speaker is not None and speaker.audio is not None:
            voice = _ProfileShim(speaker, "local_voice")
        for ch in self._get().synthesize(self._text(text, o), voice=voice):
            yield ch.pcm, ch.sample_rate

    def synthesize(self, text: str, speaker: SpeakerRef | None, options: SynthesisOptions) -> AudioResult:
        o, warnings = validate_options(self.capabilities(), speaker, options)
        t = Timer()
        parts, sr = [], 44100
        for pcm, sr in self.synthesize_stream(text, speaker, options):
            t.mark_first()
            parts.append(pcm)
        audio = np.concatenate(parts) if parts else np.zeros(0, np.float32)
        return AudioResult(audio, sr, self.capabilities().name, native_sample_rate=sr, first_audio_ms=t.first,
                           total_ms=t.elapsed_ms, warnings=warnings, params={"text_sent": self._text(text, o)})
