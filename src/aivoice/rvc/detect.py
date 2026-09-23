"""Detect and describe RVC v1/v2 packages without executing pickle payloads."""

from __future__ import annotations

import json
import re
import struct
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus"}
PTH_EXTS = {".pth", ".pt"}
INDEX_EXTS = {".index"}


@dataclass
class RvcPackageInfo:
    generation: str | None  # "RVC v1" | "RVC v2" | None
    checkpoint: str | None = None
    index: str | None = None
    sample_rate: int | None = None
    pitch_guidance: bool | None = None
    speaker: str | None = None
    reference_audio: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    root: str | None = None
    raw_meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def has_usable_reference(self) -> bool:
        return bool(self.reference_audio)


def inspect_path(path: Path) -> RvcPackageInfo:
    path = Path(path)
    if path.is_dir():
        return inspect_rvc_package(path)
    if zipfile.is_zipfile(path):
        # Inspect listing without full extract when possible; extract to sibling for deep inspect
        return _inspect_zip_listing(path)
    if path.suffix.lower() in PTH_EXTS:
        info = RvcPackageInfo(generation=None, checkpoint=str(path), root=str(path.parent))
        info.files = [path.name]
        info.notes.append("single checkpoint; pass --index if available")
        _probe_pth_header(path, info)
        return info
    raise FileNotFoundError(f"not an RVC package path: {path}")


def inspect_rvc_package(root: Path) -> RvcPackageInfo:
    root = Path(root)
    files = [p for p in root.rglob("*") if p.is_file()]
    rel = [str(p.relative_to(root)) for p in files]
    info = RvcPackageInfo(generation=None, files=sorted(rel), root=str(root))

    pths = [p for p in files if p.suffix.lower() in PTH_EXTS]
    indexes = [p for p in files if p.suffix.lower() in INDEX_EXTS]
    audios = [p for p in files if p.suffix.lower() in AUDIO_EXTS]

    if pths:
        # Prefer non-hubert/non-rmvpe weights
        ranked = sorted(
            pths,
            key=lambda p: (
                0 if re.search(r"G_|generator|model|rvc", p.name, re.I) else 1,
                0 if "hubert" not in p.name.lower() and "rmvpe" not in p.name.lower() else 2,
                -p.stat().st_size,
            ),
        )
        info.checkpoint = str(ranked[0])
        _probe_pth_header(ranked[0], info)
    if indexes:
        info.index = str(sorted(indexes, key=lambda p: -p.stat().st_size)[0])
    if audios:
        # Prefer names suggesting reference/preview
        def score(p: Path) -> tuple:
            n = p.name.lower()
            return (
                0 if any(k in n for k in ("ref", "preview", "sample", "demo", "target")) else 1,
                -p.stat().st_size,
            )

        info.reference_audio = [str(p) for p in sorted(audios, key=score)[:8]]

    # sidecar json metadata
    for jp in files:
        if jp.suffix.lower() == ".json" and jp.stat().st_size < 2_000_000:
            try:
                data = json.loads(jp.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, UnicodeError):
                continue
            if isinstance(data, dict):
                info.raw_meta.update({k: data[k] for k in list(data)[:40] if _is_json_safe(data[k])})
                if "sr" in data and info.sample_rate is None:
                    try:
                        info.sample_rate = int(data["sr"])
                    except (TypeError, ValueError):
                        pass
                if "name" in data and not info.speaker:
                    info.speaker = str(data["name"])[:120]

    if info.generation is None:
        blob = " ".join(rel).lower()
        if "v2" in blob or "rvc2" in blob:
            info.generation = "RVC v2"
        elif "v1" in blob or "rvc1" in blob:
            info.generation = "RVC v1"
        elif info.checkpoint:
            info.generation = "RVC (unknown generation)"
            info.notes.append("could not confidently detect v1 vs v2")

    if info.checkpoint is None:
        info.notes.append("no .pth/.pt checkpoint found")
    if not info.reference_audio:
        info.notes.append("no reference/preview audio found in package")
    return info


def _inspect_zip_listing(path: Path) -> RvcPackageInfo:
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
    info = RvcPackageInfo(generation=None, files=names, root=str(path))
    pths = [n for n in names if Path(n).suffix.lower() in PTH_EXTS]
    idxs = [n for n in names if Path(n).suffix.lower() in INDEX_EXTS]
    auds = [n for n in names if Path(n).suffix.lower() in AUDIO_EXTS]
    if pths:
        info.checkpoint = pths[0]
    if idxs:
        info.index = idxs[0]
    if auds:
        info.reference_audio = auds[:8]
    blob = (" ".join(names) + " " + path.name).lower()
    if "v2" in blob or "rvc2" in blob:
        info.generation = "RVC v2"
    elif "v1" in blob or "rvc1" in blob:
        info.generation = "RVC v1"
    else:
        info.generation = "RVC (unknown generation)" if pths else None
    info.notes.append("zip inspected by listing only; extract for deeper header probe")
    return info


def _is_json_safe(v: Any) -> bool:
    return isinstance(v, (str, int, float, bool)) or v is None


def _probe_pth_header(path: Path, info: RvcPackageInfo) -> None:
    """Best-effort metadata without unpickling arbitrary code.

    Official torch.load(weights_only=True) is preferred when torch is present.
    Fallback: scan ASCII strings in the first few MB for sr / version hints.
    """
    try:
        import torch

        try:
            obj = torch.load(path, map_location="cpu", weights_only=True)  # type: ignore[call-arg]
        except TypeError:
            # older torch — refuse full pickle by default
            info.notes.append(
                "torch present but weights_only unsupported; skipped unsafe pickle load"
            )
            _scan_strings(path, info)
            return
        except Exception as e:  # noqa: BLE001
            info.notes.append(f"weights_only load failed ({e}); scanned strings only")
            _scan_strings(path, info)
            return
        if isinstance(obj, dict):
            # common RVC keys
            for key in ("weight", "model", "config", "sr", "f0", "version", "info", "name"):
                if key in obj and _is_json_safe(obj[key]):
                    info.raw_meta[key] = obj[key]
            cfg = obj.get("config")
            if isinstance(cfg, (list, tuple)) and cfg:
                # RVC config[0] often related to sample rate-ish; also search for 32000/40000/48000
                pass
            if "sr" in obj:
                try:
                    info.sample_rate = int(obj["sr"])
                except (TypeError, ValueError):
                    pass
            ver = str(obj.get("version", "")).lower()
            if "v2" in ver or ver == "2":
                info.generation = "RVC v2"
            elif "v1" in ver or ver == "1":
                info.generation = "RVC v1"
            if "f0" in obj:
                info.pitch_guidance = bool(obj["f0"])
            # nested config dict
            if isinstance(cfg, dict):
                if "sr" in cfg and info.sample_rate is None:
                    try:
                        info.sample_rate = int(cfg["sr"])
                    except (TypeError, ValueError):
                        pass
        else:
            info.notes.append(f"checkpoint root type is {type(obj).__name__}")
    except ImportError:
        _scan_strings(path, info)


def _scan_strings(path: Path, info: RvcPackageInfo) -> None:
    try:
        data = path.read_bytes()[: 4 * 1024 * 1024]
    except OSError as e:
        info.notes.append(f"cannot read checkpoint: {e}")
        return
    # sample rate literals
    for sr in (48000, 40000, 32000, 16000):
        if struct.pack("<i", sr) in data or str(sr).encode() in data:
            if info.sample_rate is None:
                info.sample_rate = sr
            break
    text = "".join(chr(b) if 32 <= b < 127 else " " for b in data)
    if re.search(r"\bv2\b|version.?2", text, re.I) and info.generation is None:
        info.generation = "RVC v2"
    elif re.search(r"\bv1\b|version.?1", text, re.I) and info.generation is None:
        info.generation = "RVC v1"
    if re.search(r"\bf0\b|rmvpe|harvest|crepe", text, re.I) and info.pitch_guidance is None:
        info.pitch_guidance = True
