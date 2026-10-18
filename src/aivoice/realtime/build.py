"""Assemble a RealtimeEngine from settings (and report what was chosen, and why)."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .. import settings as st
from ..asr import available_backends, backend_config, create_backend
from ..asr.selector import ASRSelector, Selection
from ..audio.capture import ArrayCapture, make_capture
from ..audio.playback import ClockDriver, Player, make_driver
from ..expression import ExpressionCompiler
from ..paths import runtime_dir
from ..vad import Endpointer, make_vad
from .engine import EngineOptions, RealtimeEngine


class BuildError(RuntimeError):
    pass


@dataclass
class BuildReport:
    asr: str = ""
    asr_selection: Selection | None = None
    vad: str = ""
    tts: str = ""
    tts_notes: str = ""
    llm: str = "off"
    voice: str | None = None
    output: str = ""
    input: str = ""
    warnings: list[str] = field(default_factory=list)
    cleanup: list[Callable[[], None]] = field(default_factory=list)

    def lines(self) -> list[str]:
        out = [
            f"ASR     {self.asr}" + (f"  ({self.asr_selection.reason})" if self.asr_selection else ""),
            f"VAD     {self.vad}",
            f"TTS     {self.tts}  {self.tts_notes}".rstrip(),
            f"LLM     {self.llm}",
            f"Voice   {self.voice or '(backend default)'}",
            f"Input   {self.input}",
            f"Output  {self.output}",
        ]
        out += [f"warning: {w}" for w in self.warnings]
        return out


def status_path() -> Path:
    return runtime_dir() / "engine.json"


def pid_path() -> Path:
    return runtime_dir() / "engine.pid"


def choose_asr(cfg: dict[str, Any]) -> tuple[str, Selection | None]:
    asr = cfg["asr"]
    name = asr.get("backend", "auto")
    if name != "auto":
        return name, None
    avail = {k: backend_config(k, asr) for k, why in available_backends().items() if why is None}
    if not avail:
        raise BuildError(
            "no ASR runtime installed: pip install 'aivoice[moonshine]' (CPU) or 'nemo_toolkit[asr]' (GPU)"
        )
    sel = ASRSelector().select(asr.get("policy", "balanced"), avail, latency_budget_ms=float(asr.get("latency_budget_ms", 500)))
    return sel.backend, sel


def build_llm(cfg: dict[str, Any], manager: Any | None, report: BuildReport):
    from ..llm import LlamaCppBackend, SpeechIntelligenceEngine, resolve_model_path

    llm = cfg["llm"]
    mode = llm.get("mode", "verbatim")
    if mode == "verbatim":
        return None
    backend = None
    if llm.get("enabled") and llm.get("model"):
        path = resolve_model_path(llm.get("model"), manager)
        if path is None:
            report.warnings.append(f"LLM model {llm.get('model')!r} not found locally; using rule-based cleanup only")
        else:
            try:
                backend = LlamaCppBackend(
                    path,
                    n_threads=llm.get("n_threads"),
                    n_gpu_layers=int(llm.get("n_gpu_layers") or 0),
                    no_think="qwen3" in path.lower(),
                )
            except Exception as e:  # noqa: BLE001
                report.warnings.append(f"LLM failed to load ({e}); using rule-based cleanup only")
    elif llm.get("enabled"):
        report.warnings.append("llm.enabled is set but no llm.model is configured (`aivoice llm model ...`)")
    from ..tts_voices import load_profile

    character = llm.get("character") or ""
    active = cfg.get("voice", {}).get("active")
    if not character and active:
        try:
            character = load_profile(active).character
        except FileNotFoundError:
            pass
    eng = SpeechIntelligenceEngine(
        backend,
        mode=mode,
        timeout_ms=int(llm.get("timeout_ms", 350)),
        max_tokens=int(llm.get("max_tokens", 96)),
        creativity=float(llm.get("creativity", 0.3)),
        intensity=float(llm.get("intensity", 0.5)),
        character=character,
        custom_prompt=llm.get("prompt") or "",
    )
    report.llm = f"{mode} via {backend.name}" if backend else f"{mode} (rule-based, no model)"
    return eng


def build_engine(
    cfg: dict[str, Any] | None = None,
    *,
    input_file: Path | None = None,
    output: str = "virtual",  # virtual | default | null
    manager: Any | None = None,
    asr_override: Any | None = None,
    tts_override: Any | None = None,
    write_status: bool = True,
    realtime_input: bool = True,
) -> tuple[RealtimeEngine, BuildReport]:
    cfg = cfg or st.load()
    report = BuildReport()
    if manager is None and importlib.util.find_spec("huggingface_hub") is not None:
        from ..hub.manager import HuggingFaceModelManager

        manager = HuggingFaceModelManager()

    # ASR
    if asr_override is not None:
        asr = asr_override
        report.asr = asr.get_capabilities().name
    else:
        name, sel = choose_asr(cfg)
        report.asr_selection = sel
        asr = create_backend(name, settings={**cfg["asr"]}, registry=manager)
        caps = asr.get_capabilities()
        report.asr = f"{name} ({'native streaming' if caps.native_streaming else 'chunked re-decoding, not native streaming'})"

    # VAD
    v = cfg["vad"]
    vad = make_vad(v.get("engine", "auto"))
    report.vad = vad.name
    if vad.name == "energy" and v.get("engine", "auto") == "auto":
        report.warnings.append("Silero VAD unavailable (pip install silero-vad onnxruntime); using energy VAD")
    ep = Endpointer(
        vad,
        threshold=float(v.get("threshold", 0.5)),
        min_silence_ms=int(v.get("min_silence_ms", 300)),
        speech_pad_ms=int(v.get("speech_pad_ms", 120)),
    )

    # TTS
    if tts_override is not None:
        tts = tts_override
    else:
        from ..tts import create_tts

        tts = create_tts(cfg["tts"])
    caps = tts.capabilities()
    report.tts = caps.name
    report.tts_notes = ("streaming" if caps.streaming else "not streaming") + (f", remote: {caps.destination}" if caps.remote else ", local")

    # Voice
    voice = None
    active = cfg.get("voice", {}).get("active")
    if active:
        from ..tts_voices import load_profile

        try:
            voice = load_profile(active)
        except FileNotFoundError:
            report.warnings.append(f"active voice {active!r} not found; using the backend default voice")
        if voice is not None and caps.voice_kinds and voice.kind not in caps.voice_kinds:
            report.warnings.append(
                f"voice {voice.name!r} is a {voice.kind}; {caps.name} accepts {', '.join(caps.voice_kinds)}. "
                "Using the backend default voice."
            )
            voice = None
        report.voice = voice.name if voice else None

    llm = build_llm(cfg, manager, report)
    ex = cfg["expression"]
    compiler = ExpressionCompiler(
        caps.dialect,
        intensity=float(ex.get("intensity", 0.5)),
        frequency=float(ex.get("frequency", 0.5)),
        expression_only=bool(ex.get("expression_only", False)),
    )

    # Input
    a = cfg["audio"]
    block_ms = int(a.get("block_ms", 20))
    if input_file is not None:
        capture = ArrayCapture.from_file(Path(input_file), block_ms=block_ms, realtime=realtime_input)
        report.input = f"file {input_file} ({'real-time paced' if realtime_input else 'as fast as possible'})"
    else:
        from ..virtualmic import feedback_risk

        risk = feedback_risk(a.get("input_device"), a.get("virtual_mic_name", "aivoice"))
        if risk:
            raise BuildError(risk)
        capture = make_capture(device=a.get("input_device"), sample_rate=16000, block_ms=block_ms)
        report.input = f"microphone {a.get('input_device') or '(default)'}"

    # Output
    if output == "null":
        def driver_factory(p: Player):
            return ClockDriver(p, keep=False)

        report.output = "null (discarded, real-time clock)"
    elif output == "virtual":
        from ..virtualmic import cleanup_virtual_mic, ensure_virtual_mic

        name = a.get("virtual_mic_name", "aivoice")
        vm = ensure_virtual_mic(name)
        if not vm.sink:
            raise BuildError(vm.hint)
        if not vm.reused:
            report.cleanup.append(lambda: cleanup_virtual_mic(name))
        sink = vm.sink

        def driver_factory(p: Player):
            return make_driver(p, kind=a.get("output_driver", "auto"), sink=sink)

        report.output = f"virtual mic {vm.source} (sink {sink}){' [reused]' if vm.reused else ''}"
    else:
        dev = a.get("output_device")

        def driver_factory(p: Player):
            return make_driver(p, kind=a.get("output_driver", "auto"), sink=dev if isinstance(dev, str) else None, device=dev)

        report.output = f"device {dev or '(default)'}"

    s = cfg["stabilizer"]
    opts = EngineOptions(
        speculative=bool(s.get("speculative", True)),
        agreement=int(s.get("agreement", 2)),
        min_commit_words=int(s.get("min_commit_words", 3)),
        commit_on_punctuation=bool(s.get("commit_on_punctuation", True)),
        barge_in=a.get("barge_in", "queue"),
        duck_db=float(a.get("duck_db", -12)),
        max_backlog_ms=int(a.get("max_backlog_ms", 4000)),
        input_mode=a.get("input_mode", "vad"),
        echo_gate=bool(a.get("echo_gate", True)),
        monitor=bool(a.get("monitor", False)),
        status_path=status_path() if write_status else None,
    )
    eng = RealtimeEngine(
        capture=capture,
        asr=asr,
        endpointer=ep,
        tts=tts,
        driver_factory=driver_factory,
        llm=llm,
        compiler=compiler,
        voice=voice,
        options=opts,
        prebuffer_ms=int(a.get("prebuffer_ms", 60)),
        gain_db=float(a.get("output_gain_db", 0.0)),
    )
    return eng, report
