"""Speaker profiles v2 for zero-shot cloning (``aivoice clone ...``).

Stored under ``$XDG_DATA_HOME/aivoice/speakers/<id>/``::

    profile.json      profile_version 2, consent, hashes, analysis, provenance
    original.<ext>    untouched copy (only if retention was allowed)
    prepared.wav      conservative preparation used for conditioning
    prepared.json     what preparation did, before/after reports
    ge2e.npy          speaker-verification embedding (evaluation only)

Version-1 data (the MeanVC2 ``voices`` library and experimental ``voice``
profiles) is never modified: :func:`migrate_v1` copies it into v2 profiles and
records where it came from. Deleting a v2 profile removes its directory,
including cached embeddings and prepared audio.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .paths import data_home
from .voices import slugify

PROFILE_VERSION = 2
CONSENT_STATEMENT = (
    "I own this voice or have the explicit permission of the person whose voice it is to "
    "create a synthetic copy of it, and I will disclose synthetic audio where required."
)


class ProfileError(ValueError):
    pass


def speakers_dir() -> Path:
    return data_home() / "speakers"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


@dataclass
class SpeakerProfile:
    id: str
    voice_name: str
    profile_version: int = PROFILE_VERSION
    reference_kind: str = "recorded"  # recorded | synthesized | imported
    original_audio: str | None = None
    prepared_audio: str | None = None
    transcript: str | None = None
    language: str | None = None
    consent: dict[str, Any] = field(default_factory=dict)
    publishable: bool = False  # the speaker also agreed to publication of synthetic samples
    audio_hash: str | None = None
    prepared_hash: str | None = None
    analysis: dict[str, Any] = field(default_factory=dict)
    reference_report: dict[str, Any] = field(default_factory=dict)
    preparation: dict[str, Any] = field(default_factory=dict)
    identity: dict[str, Any] = field(default_factory=dict)
    conditioning: dict[str, dict[str, Any]] = field(default_factory=dict)
    engine_ids: dict[str, str] = field(default_factory=dict)
    migrated_from: dict[str, Any] | None = None
    created: str = ""
    notes: str = ""

    def dir(self) -> Path:
        return speakers_dir() / self.id

    @property
    def consent_recorded(self) -> bool:
        return bool(self.consent.get("attested"))

    def conditioning_audio(self) -> Path | None:
        for p in (self.prepared_audio, self.original_audio):
            if p and Path(p).is_file():
                return Path(p)
        return None

    def speaker_ref(self):
        from .engines.base import SpeakerRef

        return SpeakerRef(name=self.voice_name, audio=self.conditioning_audio(), transcript=self.transcript,
                          language=self.language, engine_ids=dict(self.engine_ids), profile_id=self.id)

    def embedding(self) -> np.ndarray | None:
        p = self.dir() / "ge2e.npy"
        return np.load(p) if p.is_file() else None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SpeakerProfile":
        d = migrate_dict(d)
        fields = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in fields})

    def save(self) -> None:
        self.dir().mkdir(parents=True, exist_ok=True)
        tmp = self.dir() / "profile.json.tmp"
        tmp.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        tmp.replace(self.dir() / "profile.json")


def migrate_dict(d: dict[str, Any]) -> dict[str, Any]:
    """Upgrade an older on-disk profile dict to the current version (pure function)."""
    v = int(d.get("profile_version") or d.get("meanvc2_profile_version") or 1)
    if v >= PROFILE_VERSION:
        return d
    # v1 (voices library entry) -> v2
    out = {
        "id": d.get("id"),
        "voice_name": d.get("voice_name") or d.get("display_name") or d.get("name") or d.get("id"),
        "profile_version": PROFILE_VERSION,
        "reference_kind": (d.get("extra") or {}).get("reference_kind", "recorded"),
        "original_audio": d.get("reference") or d.get("reference_audio"),
        "prepared_audio": None,
        "transcript": d.get("reference_text"),
        "consent": d.get("consent") or {},
        "audio_hash": d.get("checksum_reference"),
        "migrated_from": {"profile_version": v, "engine": d.get("engine"), "kind": d.get("kind"), "id": d.get("id")},
        "created": d.get("created") or _now(),
        "notes": d.get("notes", ""),
    }
    return out


def _new_id(name: str) -> str:
    base = slugify(name)
    vid, i = base, 2
    while (speakers_dir() / vid).exists():
        vid, i = f"{base}-{i}", i + 1
    return vid


def create_profile(
    name: str,
    reference: Path,
    *,
    i_have_rights: bool,
    transcript: str | None = None,
    language: str | None = None,
    keep_original: bool = True,
    publishable: bool = False,
    prepare: bool = True,
    denoise: bool | str = "auto",
    max_seconds: float | None = 20.0,
    reference_kind: str = "recorded",
    encoder: Any | None = None,
) -> SpeakerProfile:
    if not i_have_rights:
        raise ProfileError("Creating a voice clone requires --i-have-rights:\n  " + CONSENT_STATEMENT)
    reference = Path(reference)
    if not reference.is_file():
        raise ProfileError(f"no such reference: {reference}")
    from .audio_io import read_wav, write_wav
    from .research.reference_quality import analyze_reference, prepare_reference
    from .research.speaker_identity import analyze_speaker

    vid = _new_id(name)
    d = speakers_dir() / vid
    d.mkdir(parents=True)
    try:
        prof = SpeakerProfile(id=vid, voice_name=name.strip(), reference_kind=reference_kind, transcript=transcript,
                              language=language, publishable=publishable, created=_now(),
                              consent={"attested": True, "statement": CONSENT_STATEMENT, "at": _now(),
                                       "publication_agreed": publishable})
        prof.audio_hash = _sha256(reference)
        if keep_original:
            dst = d / f"original{reference.suffix.lower() or '.wav'}"
            shutil.copy2(reference, dst)
            prof.original_audio = str(dst)
        src = Path(prof.original_audio) if prof.original_audio else reference
        if prepare:
            rep = prepare_reference(src, d / "prepared.wav", max_seconds=max_seconds, denoise=denoise, encoder=encoder)
            prof.prepared_audio = rep.prepared
            prof.preparation = {"steps": rep.steps, "reverted": rep.reverted, "identity_similarity": rep.identity_similarity}
            prof.reference_report = rep.report_before.to_dict()
        else:
            x, sr = read_wav(src)
            write_wav(d / "prepared.wav", x, sr)
            prof.prepared_audio = str(d / "prepared.wav")
            prof.reference_report = analyze_reference(src, encoder=encoder).to_dict()
        prof.prepared_hash = _sha256(Path(prof.prepared_audio))
        x, sr = read_wav(Path(prof.prepared_audio))
        an = analyze_speaker(x, sr, transcript=transcript, encoder=encoder)
        prof.analysis = an.to_dict()
        if an.embedding is not None:
            np.save(d / "ge2e.npy", np.asarray(an.embedding, np.float32))
            prof.identity = {"encoder": an.embedding_model, "file": "ge2e.npy",
                             "use": "evaluation only (speaker verification); not a synthesis conditioning input"}
        prof.save()
        return prof
    except Exception:
        shutil.rmtree(d, ignore_errors=True)
        raise


def list_profiles() -> list[SpeakerProfile]:
    root = speakers_dir()
    if not root.is_dir():
        return []
    out = []
    for p in sorted(root.iterdir()):
        f = p / "profile.json"
        if f.is_file():
            try:
                out.append(SpeakerProfile.from_dict(json.loads(f.read_text(encoding="utf-8"))))
            except (json.JSONDecodeError, TypeError):
                continue
    return out


def load_profile(name_or_id: str) -> SpeakerProfile:
    f = speakers_dir() / name_or_id / "profile.json"
    if f.is_file():
        return SpeakerProfile.from_dict(json.loads(f.read_text(encoding="utf-8")))
    for p in list_profiles():
        if p.voice_name.lower() == name_or_id.lower() or p.id == slugify(name_or_id):
            return p
    raise FileNotFoundError(f"voice clone profile not found: {name_or_id} (aivoice clone list)")


def delete_profile(name_or_id: str) -> SpeakerProfile:
    p = load_profile(name_or_id)
    shutil.rmtree(p.dir())
    return p


def record_conditioning(p: SpeakerProfile, engine: str, info: dict[str, Any]) -> None:
    p.conditioning[engine] = {**info, "source": "prepared.wav" if p.prepared_audio else "original", "updated": _now()}
    p.save()


def migrate_v1(*, include_tts_voices: bool = True) -> list[tuple[str, str]]:
    """Copy v1 profiles into v2 (originals untouched). Returns (source, new id) pairs."""
    done: list[tuple[str, str]] = []
    existing = {(p.migrated_from or {}).get("id") for p in list_profiles()}
    from .voices import list_voices

    for v in list_voices():
        if v.id in existing or not v.reference or not Path(v.reference).is_file():
            continue
        d = migrate_dict({**v.to_dict(), "meanvc2_profile_version": 1})
        d["id"] = _new_id(v.display_name)
        d["consent"] = {"attested": False, "note": "migrated from the MeanVC2 voice library; re-attest before cloning"}
        dst = speakers_dir() / d["id"]
        dst.mkdir(parents=True)
        shutil.copy2(v.reference, dst / "original.wav")
        d["original_audio"] = str(dst / "original.wav")
        SpeakerProfile.from_dict(d).save()
        done.append((f"voices:{v.id}", d["id"]))
    if include_tts_voices:
        from .tts_voices import list_profiles as list_tts

        for t in list_tts():
            if t.id in existing or t.kind != "local_voice" or not t.reference_audio or not Path(t.reference_audio).is_file():
                continue
            d = migrate_dict({"id": t.id, "name": t.name, "reference_audio": t.reference_audio,
                              "reference_text": t.reference_text, "consent": t.consent, "kind": t.kind, "profile_version": 1})
            d["id"] = _new_id(t.name)
            dst = speakers_dir() / d["id"]
            dst.mkdir(parents=True)
            ext = Path(t.reference_audio).suffix or ".wav"
            shutil.copy2(t.reference_audio, dst / f"original{ext}")
            d["original_audio"] = str(dst / f"original{ext}")
            SpeakerProfile.from_dict(d).save()
            done.append((f"voice:{t.id}", d["id"]))
    return done


def attest(p: SpeakerProfile, *, i_have_rights: bool) -> SpeakerProfile:
    if not i_have_rights:
        raise ProfileError(CONSENT_STATEMENT)
    p.consent = {"attested": True, "statement": CONSENT_STATEMENT, "at": _now()}
    p.save()
    return p
