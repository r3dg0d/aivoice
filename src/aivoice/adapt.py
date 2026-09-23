"""Honest RVC → MeanVC2 voice migration (reference-audio adaptation, not weight conversion)."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from .download import safe_extract
from .rvc.detect import RvcPackageInfo, inspect_path
from .voices import VoiceEntry, create_from_reference, find_by_source, slugify


@dataclass
class AdaptResult:
    voice: VoiceEntry | None
    rvc: RvcPackageInfo
    message: str
    used_reference: Path | None = None
    engine: str = "meanvc2"


PERSON_NOTICE = (
    "Use voice models only where you have the necessary rights/permission, "
    "and don't use generated audio deceptively or fraudulently."
)


def adapt_rvc_to_meanvc2(
    package: Path,
    *,
    display_name: str | None = None,
    reference_override: Path | None = None,
    prefer_rvc_fallback: bool = True,
    source_provider: str | None = None,
    source_model_id: str | None = None,
    source_url: str | None = None,
    work_dir: Path | None = None,
) -> AdaptResult:
    """Build a MeanVC2 voice profile from an RVC package when legitimate reference audio exists.

    This does **not** convert RVC `.pth` weights into MeanVC2 checkpoints.
    MeanVC2 is zero-shot and conditions on target reference audio (WavLM+ECAPA → UTTE).
    """
    package = Path(package)
    extract_root = package
    if package.is_file() and package.suffix.lower() in {".zip", ".pth", ".pt"}:
        if work_dir is None:
            work_dir = package.parent / f".aivoice_extract_{slugify(package.stem)}"
        if package.suffix.lower() == ".zip":
            extract_root = safe_extract(package, work_dir)
        else:
            work_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(package, work_dir / package.name)
            extract_root = work_dir

    rvc = inspect_path(extract_root)
    name = display_name or rvc.speaker or package.stem
    name = name.strip() or "imported-voice"

    ref: Path | None = Path(reference_override) if reference_override else None
    if ref is None and rvc.reference_audio:
        ref = Path(rvc.reference_audio[0])

    if ref is not None and ref.is_file():
        voice = create_from_reference(
            name,
            ref,
            engine="meanvc2",
            source_provider=source_provider or "local",
            source_model_id=source_model_id,
            source_url=source_url,
            original_architecture=rvc.generation,
            notes=(
                "MeanVC2 profile built from package reference/preview audio "
                "(zero-shot conditioning). Not a direct RVC weight conversion."
            ),
            extra={"rvc": rvc.to_dict(), "adaptation": "reference-audio"},
        )
        # stash original checkpoint paths in voice dir for optional RVC fallback later
        if rvc.checkpoint:
            try:
                shutil.copy2(rvc.checkpoint, voice.dir() / "source_rvc.pth")
            except OSError:
                pass
        if rvc.index:
            try:
                shutil.copy2(rvc.index, voice.dir() / "source_rvc.index")
            except OSError:
                pass
        return AdaptResult(
            voice=voice,
            rvc=rvc,
            message="MeanVC2 profile created from reference audio",
            used_reference=ref,
            engine="meanvc2",
        )

    if prefer_rvc_fallback and rvc.checkpoint:
        # Record an RVC-backed voice entry (runtime RVC engine may be limited)
        voice = create_from_reference(
            name,
            # create a tiny placeholder wav note via empty? — require user reference
            # Without audio we cannot create MeanVC2 profile; store metadata-only incomplete voice
            _write_placeholder_note(extract_root, name),
            engine="rvc",
            source_provider=source_provider or "local",
            source_model_id=source_model_id,
            source_url=source_url,
            original_architecture=rvc.generation,
            notes=(
                "No usable target reference audio found. Voice recorded with engine=rvc. "
                "Provide --reference for MeanVC2 zero-shot adaptation."
            ),
            extra={"rvc": rvc.to_dict(), "adaptation": "rvc-fallback-incomplete"},
        )
        voice.status = "incomplete"
        from .voices import _write

        _write(voice)
        return AdaptResult(
            voice=voice,
            rvc=rvc,
            message=(
                "This RVC package does not contain enough target-speaker audio "
                "to build a reliable MeanVC2 profile. Options: (1) provide a clean "
                "reference WAV (2) locate provider preview audio (3) use RVC backend when available."
            ),
            used_reference=None,
            engine="rvc",
        )

    return AdaptResult(
        voice=None,
        rvc=rvc,
        message=(
            "This RVC package does not contain enough target-speaker audio "
            "to build a reliable MeanVC2 profile."
        ),
        engine="none",
    )


def _write_placeholder_note(root: Path, name: str) -> Path:
    """Create a silent minimal wav so voice dirs stay consistent — marked incomplete."""
    import wave

    path = Path(root) / "_aivoice_placeholder.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 1600)  # 0.1s silence
    return path


def cached_voice_for_search(provider: str, model_id: str) -> VoiceEntry | None:
    return find_by_source(provider, model_id)
