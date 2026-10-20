"""The research-fusion synthesis pipeline behind ``aivoice clone synthesize``.

Stages (each recorded with its time and what it did):

1. text: Unicode/number/abbreviation normalisation and an optional user lexicon
   (phonetics); language detection when none is given; for X-Voice, code-switched
   text is tagged with its own ``[main|lang]`` format;
2. routing: compatible, installed engine chosen by measured results (router);
3. conditioning: the profile's *prepared* reference (silence-trimmed, best window,
   loudness-normalised, identity-guarded) plus its transcript when the engine can use it;
4. duration: silence-aware duration plan for engines with native duration control;
5. long-form: segmentation and consistent-identity stitching for long text;
6. enhancement (off by default): only steps that measurably help, identity-guarded;
7. evaluation: speaker similarity to the reference and acoustic metrics, stored with the output.

Every technique here has an off switch so its contribution can be ablated
(docs/research/experiments).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from ..engines.base import TASK_TTS, SpeakerRef, SynthesisOptions
from .router import Decision, route


@dataclass
class FusionConfig:
    normalize_text: bool = True
    lexicon: dict[str, str] | None = None
    use_prepared_reference: bool = True
    plan_duration: bool = True
    longform: bool = True
    max_chars: int = 220
    enhance: str = "off"
    evaluate: bool = True
    tag_code_switch: bool = True


@dataclass
class FusionResult:
    audio: np.ndarray
    sample_rate: int
    engine: str
    decision: dict[str, Any]
    stages: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)

    def sidecar(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("audio")
        d["synthetic"] = True
        return d


def synthesize(
    text: str,
    profile: Any | None,
    options: SynthesisOptions | None = None,
    *,
    engine: str = "auto",
    priority: str = "balanced",
    config: FusionConfig | None = None,
    engine_obj: Any | None = None,
) -> FusionResult:
    from ..engines.registry import get_engine
    from ..research import phonetics

    cfg = config or FusionConfig()
    o = options or SynthesisOptions()
    stages: list[dict[str, Any]] = []
    warnings: list[str] = []

    def stage(name: str, t0: float, **detail: Any) -> None:
        stages.append({"stage": name, "ms": round((time.perf_counter() - t0) * 1000, 2), **detail})

    speaker: SpeakerRef | None = None
    if profile is not None:
        if not profile.consent_recorded:
            raise PermissionError(f"profile {profile.voice_name!r} has no consent attestation: aivoice clone attest {profile.id} --i-have-rights")
        speaker = profile.speaker_ref()
        if not cfg.use_prepared_reference and profile.original_audio:
            speaker.audio = Path(profile.original_audio)

    t0 = time.perf_counter()
    lang = o.language or phonetics.detect_language(text)
    norm = phonetics.normalize_text(text, lang, lexicon=cfg.lexicon) if cfg.normalize_text else text
    stage("text", t0, language=lang, language_detected=o.language is None, normalized=norm != text)
    o = replace(o, language=o.language or None)

    t0 = time.perf_counter()
    if engine_obj is not None:
        dec = Decision(engine_obj.capabilities().name, priority, False, ["engine given explicitly"])
        eng = engine_obj
    elif engine == "auto":
        dec = route(TASK_TTS, priority=priority, speaker=speaker, options=o)
        if dec.engine is None:
            raise RuntimeError("no installed engine can do this: " + "; ".join(f"{k}: {v}" for k, v in dec.rejected.items()))
        eng = get_engine(dec.engine)
    else:
        dec = Decision(engine, priority, False, ["engine chosen by the user"])
        eng = get_engine(engine)
    caps = eng.capabilities()
    stage("route", t0, engine=caps.name, measured=dec.measured)

    if caps.name == "xvoice" and cfg.tag_code_switch and len(phonetics.segment_code_switch(norm, lang)) > 1:
        norm = phonetics.xvoice_tagged(norm, lang)
        stages[-1]["code_switch_tagged"] = True

    if speaker is not None and speaker.transcript and not caps.accepts_prompt_transcript and caps.transcript_free_prompt:
        warnings.append(f"{caps.name} clones transcript-free; the profile transcript is not used")

    t0 = time.perf_counter()
    long = cfg.longform and len(norm) > cfg.max_chars and caps.name != "xvoice"
    if long:
        from ..research.longform import synthesize_longform
        from ..research.speaker_identity import get_encoder

        enc = get_encoder() if cfg.evaluate else None
        ref = None
        if enc is not None and speaker is not None and speaker.audio:
            from ..audio_io import read_wav

            ref = read_wav(Path(speaker.audio))
        lf = synthesize_longform(eng, norm, speaker, o, max_chars=cfg.max_chars, encoder=enc, reference_audio=ref,
                                 plan_durations=cfg.plan_duration)
        audio, sr, params = lf.audio, lf.sample_rate, {"segments": lf.segments, "consistency_min": lf.consistency_min,
                                                        "outliers": lf.outliers}
        stage("synthesize-longform", t0, segments=len(lf.segments))
    else:
        if cfg.plan_duration and caps.duration_control and speaker is not None and speaker.audio and o.target_duration_s is None:
            from ..research.duration import plan_duration

            plan = plan_duration(norm, speaker.audio, transcript=speaker.transcript, language=lang, speed=o.speed or 1.0)
            o = replace(o, target_duration_s=plan.seconds)
            stages.append({"stage": "duration-plan", "seconds": plan.seconds, "method": plan.method,
                           "rate_syl_per_s": round(plan.rate_syl_per_s, 3)})
        r = eng.synthesize(norm, speaker, o)
        audio, sr, params = r.audio, r.sample_rate, dict(r.params)
        warnings += r.warnings
        stage("synthesize", t0, first_audio_ms=r.first_audio_ms, rtf=r.rtf, native_sample_rate=r.native_sample_rate)

    if cfg.enhance != "off":
        from ..audio.enhancement import enhance
        from ..research.speaker_identity import get_encoder

        t0 = time.perf_counter()
        audio, rep = enhance(audio, sr, mode=cfg.enhance, encoder=get_encoder())
        stage("enhance", t0, applied=rep.applied, skipped=rep.skipped, rejected=rep.rejected,
              identity_similarity=rep.identity_similarity)

    metrics: dict[str, Any] = {}
    if cfg.evaluate:
        from ..evaluation.metrics import acoustic_metrics, speaker_similarity

        t0 = time.perf_counter()
        metrics["acoustic"] = acoustic_metrics(audio, sr)
        if speaker is not None and speaker.audio:
            from ..audio_io import read_wav

            ref, rsr = read_wav(Path(speaker.audio))
            try:
                metrics["similarity_to_reference"] = speaker_similarity(audio, sr, ref, rsr)
            except ValueError as e:
                metrics["similarity_to_reference"] = {"error": str(e)}
        stage("evaluate", t0)
    return FusionResult(audio, sr, caps.name, dec.to_dict(), stages, warnings, metrics,
                        {**params, "text": norm, "options": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(o).items()}})


def save_result(res: FusionResult, out: str | Path) -> tuple[Path, Path]:
    from ..evaluation.metrics import write_labeled_wav

    out = Path(out)
    write_labeled_wav(out, res.audio, res.sample_rate,
                      comment=f"Synthetic speech generated by aivoice ({res.engine}). Not a recording.")
    side = out.with_suffix(out.suffix + ".json")
    side.write_text(json.dumps(res.sidecar(), indent=2, default=str) + "\n", encoding="utf-8")
    return out, side
