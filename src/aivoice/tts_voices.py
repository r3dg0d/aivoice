"""Voice profiles for experimental (ASR -> TTS) mode.

Three kinds, deliberately kept apart:

``local_voice``
    Reference audio + its exact transcript stored on this machine. Sent with
    each request to a fish-speech compatible server (local or one you run
    remotely). Works with Drama 3 and S2-Pro.

``local_voice_id``
    A ``reference_id`` saved on a fish-speech server (``POST /v1/references/add``).
    aivoice keeps the id, the server URL, and a copy of the audio so the id can
    be re-created on another server.

``hosted_fish_id``
    A voice model id on fish.audio, usable only through the official hosted API
    (``reference_id`` in ``POST /v1/tts``). Hosted ids are *not* interchangeable
    with local ids: the official API offers no way to export a hosted voice's
    conditioning data into a local Drama 3 server, and aivoice does not try.

Every profile records an authorization attestation: the user states they have
the right to reproduce that voice. aivoice will not create one without it.
"""

from __future__ import annotations

import json
import re
import shutil
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .paths import tts_voices_dir
from .voices import slugify

KINDS = ("local_voice", "local_voice_id", "hosted_fish_id")
_REF_ID = re.compile(r"^[A-Za-z0-9 _-]{1,255}$")  # fish-speech AddReferenceRequest.id
_HOSTED_ID = re.compile(r"^[A-Za-z0-9]{8,64}$")

AUTHORIZATION_STATEMENT = (
    "I own this voice or have the explicit permission of the person whose voice it is to "
    "reproduce it, and I will disclose synthetic audio where required."
)


class VoiceProfileError(ValueError):
    pass


@dataclass
class VoiceProfile:
    id: str
    name: str
    kind: str
    reference_audio: str | None = None
    reference_text: str | None = None
    reference_id: str | None = None
    server_url: str | None = None
    consent: dict[str, Any] = field(default_factory=dict)
    created: str = ""
    notes: str = ""
    source_voice: str | None = None  # id in the MeanVC2 voice library it was built from
    character: str = ""  # optional personality prompt for CHARACTER mode
    extra: dict[str, Any] = field(default_factory=dict)

    def dir(self) -> Path:
        return tts_voices_dir() / self.id

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "VoiceProfile":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    def reference_bytes(self) -> bytes:
        if not self.reference_audio or not Path(self.reference_audio).is_file():
            raise VoiceProfileError(f"voice {self.name!r} has no reference audio on disk")
        return Path(self.reference_audio).read_bytes()


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _consent(attested: bool) -> dict[str, Any]:
    if not attested:
        raise VoiceProfileError(
            "Voice profiles require an authorization attestation:\n  "
            + AUTHORIZATION_STATEMENT
            + "\nRe-run with --i-have-rights after confirming it is true."
        )
    return {"attested": True, "statement": AUTHORIZATION_STATEMENT, "at": _now()}


def _write(p: VoiceProfile) -> VoiceProfile:
    p.dir().mkdir(parents=True, exist_ok=True)
    (p.dir() / "profile.json").write_text(json.dumps(p.to_dict(), indent=2) + "\n", encoding="utf-8")
    return p


def _new_id(name: str) -> str:
    base = slugify(name)
    vid = base
    i = 2
    while (tts_voices_dir() / vid).exists():
        vid = f"{base}-{i}"
        i += 1
    return vid


def validate_reference(audio: Path, text: str | None) -> list[str]:
    """Problems that would make a reference clip a poor or unusable conditioning sample."""
    issues: list[str] = []
    if not audio.is_file():
        return [f"missing reference audio: {audio}"]
    if not text or not text.strip():
        issues.append("reference transcript is empty: Fish models need the exact words spoken in the clip")
    try:
        import soundfile as sf  # type: ignore

        info = sf.info(str(audio))
        dur = info.frames / info.samplerate
        if dur < 3:
            issues.append(f"reference is only {dur:.1f}s; 5-30 s of clean speech works best")
        elif dur > 60:
            issues.append(f"reference is {dur:.0f}s; trim to under 30 s (longer adds latency for every request)")
    except ImportError:
        pass
    except Exception as e:  # noqa: BLE001
        issues.append(f"could not read reference audio: {e}")
    return issues


def add_local_voice(
    name: str,
    audio: Path,
    text: str,
    *,
    i_have_rights: bool,
    character: str = "",
    source_voice: str | None = None,
) -> VoiceProfile:
    consent = _consent(i_have_rights)
    issues = [i for i in validate_reference(Path(audio), text) if i.startswith(("missing", "reference transcript"))]
    if issues:
        raise VoiceProfileError("; ".join(issues))
    vid = _new_id(name)
    d = tts_voices_dir() / vid
    d.mkdir(parents=True)
    ext = Path(audio).suffix.lower() or ".wav"
    dest = d / f"reference{ext}"
    shutil.copy2(audio, dest)
    (d / "reference.txt").write_text(text.strip() + "\n", encoding="utf-8")
    return _write(
        VoiceProfile(
            id=vid,
            name=name.strip(),
            kind="local_voice",
            reference_audio=str(dest),
            reference_text=text.strip(),
            consent=consent,
            created=_now(),
            character=character,
            source_voice=source_voice,
        )
    )


def add_local_voice_id(
    name: str,
    reference_id: str,
    *,
    server_url: str,
    i_have_rights: bool,
    audio: Path | None = None,
    text: str | None = None,
    register: bool = False,
    client: Any | None = None,
) -> VoiceProfile:
    """Point at (or create, with ``register=True``) a reference saved on a fish-speech server."""
    consent = _consent(i_have_rights)
    if not _REF_ID.match(reference_id):
        raise VoiceProfileError("reference id may contain letters, digits, spaces, '-' and '_' (max 255)")
    vid = _new_id(name)
    d = tts_voices_dir() / vid
    d.mkdir(parents=True)
    ref_audio = None
    if audio is not None:
        ref_audio = d / f"reference{Path(audio).suffix.lower() or '.wav'}"
        shutil.copy2(audio, ref_audio)
        if text:
            (d / "reference.txt").write_text(text.strip() + "\n", encoding="utf-8")
    prof = VoiceProfile(
        id=vid,
        name=name.strip(),
        kind="local_voice_id",
        reference_audio=str(ref_audio) if ref_audio else None,
        reference_text=text.strip() if text else None,
        reference_id=reference_id,
        server_url=server_url,
        consent=consent,
        created=_now(),
    )
    if register:
        if ref_audio is None or not text:
            shutil.rmtree(d)
            raise VoiceProfileError("registering on the server needs --audio and --text")
        try:
            from .tts.fish import FishServerClient

            (client or FishServerClient(server_url)).add_reference(reference_id, ref_audio, text)
        except Exception:
            shutil.rmtree(d, ignore_errors=True)
            raise
    return _write(prof)


def add_hosted(name: str, reference_id: str, *, i_have_rights: bool, notes: str = "") -> VoiceProfile:
    consent = _consent(i_have_rights)
    if not _HOSTED_ID.match(reference_id):
        raise VoiceProfileError("hosted Fish Audio ids are alphanumeric (e.g. 802e3bc2b27e49c2995d23ef70e6ac89)")
    vid = _new_id(name)
    return _write(
        VoiceProfile(
            id=vid,
            name=name.strip(),
            kind="hosted_fish_id",
            reference_id=reference_id,
            consent=consent,
            created=_now(),
            notes=notes or "Usable only through the official Fish Audio API (tts.backend=fish-api).",
        )
    )


def list_profiles() -> list[VoiceProfile]:
    root = tts_voices_dir()
    if not root.is_dir():
        return []
    out = []
    for p in sorted(root.iterdir()):
        f = p / "profile.json"
        if f.is_file():
            try:
                out.append(VoiceProfile.from_dict(json.loads(f.read_text(encoding="utf-8"))))
            except (json.JSONDecodeError, TypeError, KeyError):
                continue
    return out


def load_profile(name_or_id: str) -> VoiceProfile:
    f = tts_voices_dir() / name_or_id / "profile.json"
    if f.is_file():
        return VoiceProfile.from_dict(json.loads(f.read_text(encoding="utf-8")))
    for p in list_profiles():
        if p.name.lower() == name_or_id.lower() or p.id == slugify(name_or_id):
            return p
    raise FileNotFoundError(f"voice profile not found: {name_or_id}")


def remove_profile(name_or_id: str) -> VoiceProfile:
    p = load_profile(name_or_id)
    shutil.rmtree(p.dir())
    return p


def verify_profile(p: VoiceProfile) -> list[str]:
    issues: list[str] = []
    if p.kind not in KINDS:
        issues.append(f"unknown kind {p.kind!r}")
    if not p.consent.get("attested"):
        issues.append("no authorization attestation recorded")
    if p.kind == "local_voice":
        issues += validate_reference(Path(p.reference_audio or ""), p.reference_text)
    elif p.kind == "local_voice_id":
        if not p.reference_id or not p.server_url:
            issues.append("missing reference_id/server_url")
    elif p.kind == "hosted_fish_id" and not p.reference_id:
        issues.append("missing hosted reference_id")
    return issues


def export_profile(p: VoiceProfile, dest: Path) -> Path:
    dest = Path(dest)
    if dest.suffix != ".zip":
        dest = dest.with_suffix(".zip")
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        data = p.to_dict()
        if p.reference_audio:
            arc = "reference" + Path(p.reference_audio).suffix
            zf.write(p.reference_audio, arc)
            data["reference_audio"] = arc
        zf.writestr("profile.json", json.dumps(data, indent=2))
    return dest


def import_profile(archive: Path, *, i_have_rights: bool, name: str | None = None) -> VoiceProfile:
    from .download import safe_extract

    consent = _consent(i_have_rights)
    tmp = tts_voices_dir() / ".import-tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    safe_extract(Path(archive), tmp, max_files=10, max_total_bytes=200 * 1024**2)
    try:
        data = json.loads((tmp / "profile.json").read_text(encoding="utf-8"))
        if data.get("kind") not in KINDS:
            raise VoiceProfileError("archive does not contain a valid aivoice voice profile")
        vid = _new_id(name or data.get("name") or "imported")
        d = tts_voices_dir() / vid
        d.mkdir(parents=True)
        if data.get("reference_audio"):
            src = tmp / Path(data["reference_audio"]).name
            if src.is_file():
                shutil.copy2(src, d / src.name)
                data["reference_audio"] = str(d / src.name)
            else:
                data["reference_audio"] = None
        data.update(id=vid, name=name or data.get("name") or vid, consent=consent, created=_now())
        return _write(VoiceProfile.from_dict(data))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
