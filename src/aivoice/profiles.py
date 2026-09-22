"""Voice profiles under XDG data dir."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .paths import ensure_dirs, profiles_dir

_SAFE = re.compile(r"^[a-zA-Z0-9_-]+$")


@dataclass
class VoiceProfile:
    name: str
    reference: str  # path to reference wav
    mode: str = "balanced"
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def path(self) -> Path:
        return profiles_dir() / self.name

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VoiceProfile:
        return cls(
            name=data["name"],
            reference=data["reference"],
            mode=data.get("mode", "balanced"),
            notes=data.get("notes", ""),
            extra=data.get("extra") or {},
        )


def _validate_name(name: str) -> None:
    if not _SAFE.match(name):
        raise ValueError("profile name must be alphanumeric / _ / -")


def create_profile(
    name: str,
    reference: Path,
    *,
    mode: str = "balanced",
    notes: str = "",
    copy_reference: bool = True,
) -> VoiceProfile:
    ensure_dirs()
    _validate_name(name)
    dest = profiles_dir() / name
    if dest.exists():
        raise FileExistsError(f"profile already exists: {name}")
    if not reference.is_file():
        raise FileNotFoundError(f"reference wav not found: {reference}")
    dest.mkdir(parents=True, exist_ok=True)
    ref_path = reference
    if copy_reference:
        ref_dest = dest / ("reference" + reference.suffix.lower())
        shutil.copy2(reference, ref_dest)
        ref_path = ref_dest
    profile = VoiceProfile(name=name, reference=str(ref_path), mode=mode, notes=notes)
    (dest / "profile.json").write_text(json.dumps(profile.to_dict(), indent=2) + "\n", encoding="utf-8")
    return profile


def list_profiles() -> list[VoiceProfile]:
    ensure_dirs()
    out: list[VoiceProfile] = []
    for p in sorted(profiles_dir().iterdir()):
        meta = p / "profile.json"
        if p.is_dir() and meta.is_file():
            try:
                out.append(VoiceProfile.from_dict(json.loads(meta.read_text(encoding="utf-8"))))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    return out


def load_profile(name: str) -> VoiceProfile:
    meta = profiles_dir() / name / "profile.json"
    if not meta.is_file():
        raise FileNotFoundError(f"profile not found: {name}")
    return VoiceProfile.from_dict(json.loads(meta.read_text(encoding="utf-8")))
