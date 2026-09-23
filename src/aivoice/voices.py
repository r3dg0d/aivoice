"""Installed voice library (MeanVC2 profiles + optional RVC-backed entries)."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .audio_prep import prepare_reference, score_reference
from .paths import ensure_dirs, voices_dir

_SAFE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9 _.-]{0,80}$")
PROFILE_VERSION = 1


def slugify(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "-", name.strip()).strip("-").lower()
    return (s or "voice")[:64]


@dataclass
class VoiceEntry:
    id: str
    display_name: str
    engine: str  # meanvc2 | rvc
    status: str  # ready | incomplete | error
    source_provider: str | None = None
    source_model_id: str | None = None
    source_url: str | None = None
    original_architecture: str | None = None
    author: str | None = None
    license: str | None = None
    created: str = ""
    reference: str | None = None
    checksum_reference: str | None = None
    meanvc2_profile_version: int = PROFILE_VERSION
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def dir(self) -> Path:
        return voices_dir() / self.id

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VoiceEntry:
        return cls(
            id=data["id"],
            display_name=data.get("display_name") or data["id"],
            engine=data.get("engine", "meanvc2"),
            status=data.get("status", "ready"),
            source_provider=data.get("source_provider"),
            source_model_id=data.get("source_model_id"),
            source_url=data.get("source_url"),
            original_architecture=data.get("original_architecture"),
            author=data.get("author"),
            license=data.get("license"),
            created=data.get("created", ""),
            reference=data.get("reference"),
            checksum_reference=data.get("checksum_reference"),
            meanvc2_profile_version=int(data.get("meanvc2_profile_version") or PROFILE_VERSION),
            notes=data.get("notes", ""),
            extra=data.get("extra") or {},
        )


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def list_voices() -> list[VoiceEntry]:
    ensure_dirs()
    out: list[VoiceEntry] = []
    for p in sorted(voices_dir().iterdir()):
        meta = p / "metadata.json"
        if p.is_dir() and meta.is_file():
            try:
                out.append(VoiceEntry.from_dict(json.loads(meta.read_text(encoding="utf-8"))))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return out


def load_voice(name_or_id: str) -> VoiceEntry:
    ensure_dirs()
    # exact id
    meta = voices_dir() / name_or_id / "metadata.json"
    if meta.is_file():
        return VoiceEntry.from_dict(json.loads(meta.read_text(encoding="utf-8")))
    # display name match
    for v in list_voices():
        if v.display_name.lower() == name_or_id.lower() or v.id == slugify(name_or_id):
            return v
    raise FileNotFoundError(f"voice not found: {name_or_id}")


def find_by_source(provider: str, model_id: str) -> VoiceEntry | None:
    for v in list_voices():
        if v.source_provider == provider and v.source_model_id == model_id:
            return v
    return None


def remove_voice(name_or_id: str) -> None:
    v = load_voice(name_or_id)
    shutil.rmtree(v.dir())


def rename_voice(name_or_id: str, new_display: str) -> VoiceEntry:
    v = load_voice(name_or_id)
    v.display_name = new_display.strip()
    _write(v)
    return v


def verify_voice(name_or_id: str) -> list[str]:
    v = load_voice(name_or_id)
    issues: list[str] = []
    if not v.dir().is_dir():
        return ["voice directory missing"]
    if v.engine == "meanvc2":
        if not v.reference or not Path(v.reference).is_file():
            issues.append("missing reference audio")
        elif v.checksum_reference:
            got = _sha256(Path(v.reference))
            if got != v.checksum_reference:
                issues.append("reference checksum mismatch")
    if v.status != "ready":
        issues.append(f"status={v.status}")
    return issues


def create_from_reference(
    display_name: str,
    reference: Path,
    *,
    voice_id: str | None = None,
    engine: str = "meanvc2",
    source_provider: str | None = None,
    source_model_id: str | None = None,
    source_url: str | None = None,
    original_architecture: str | None = None,
    author: str | None = None,
    license: str | None = None,
    notes: str = "",
    extra: dict[str, Any] | None = None,
) -> VoiceEntry:
    ensure_dirs()
    display_name = display_name.strip()
    if not display_name:
        raise ValueError("display name required")
    vid = voice_id or slugify(display_name)
    dest = voices_dir() / vid
    if dest.exists():
        raise FileExistsError(f"voice already exists: {vid}")
    dest.mkdir(parents=True, exist_ok=True)

    ref_dest = dest / "reference.wav"
    prep = prepare_reference(Path(reference), ref_dest)
    quality = score_reference(ref_dest)

    # profile.toml (human-readable) + metadata.json
    entry = VoiceEntry(
        id=vid,
        display_name=display_name,
        engine=engine,
        status="ready" if prep.output.is_file() else "incomplete",
        source_provider=source_provider,
        source_model_id=source_model_id,
        source_url=source_url,
        original_architecture=original_architecture,
        author=author,
        license=license,
        created=_now(),
        reference=str(ref_dest),
        checksum_reference=_sha256(ref_dest),
        notes=notes
        or "; ".join(prep.notes + quality),
        extra={
            **(extra or {}),
            "prep_notes": prep.notes,
            "quality_notes": quality,
            "duration_s": prep.duration_s,
            "sample_rate": prep.sample_rate,
        },
    )
    _write(entry)
    (dest / "source.json").write_text(
        json.dumps(
            {
                "provider": source_provider,
                "model_id": source_model_id,
                "url": source_url,
                "architecture": original_architecture,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (dest / "profile.toml").write_text(
        (
            f'id = "{entry.id}"\n'
            f'display_name = "{entry.display_name}"\n'
            f'engine = "{entry.engine}"\n'
            f'status = "{entry.status}"\n'
            f'reference = "reference.wav"\n'
            f"meanvc2_profile_version = {PROFILE_VERSION}\n"
        ),
        encoding="utf-8",
    )
    return entry


def _write(entry: VoiceEntry) -> None:
    ensure_dirs()
    d = entry.dir()
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps(entry.to_dict(), indent=2) + "\n", encoding="utf-8")
