"""HuggingFaceModelManager: search, inspect, download, verify, register, remove.

Built on the official ``huggingface_hub`` library (snapshot_download, HfApi,
scan_cache_dir). Weights stay in the standard content-addressed Hugging Face
cache, shared with every other tool on the machine; aivoice keeps only a small
JSON registry that points at verified snapshots by commit hash.

Rules this module enforces:

* nothing is downloaded without a size estimate, a disk-space check and, above
  the configured threshold, explicit confirmation;
* gated repositories go through the user's own Hugging Face login; there is no
  attempt to get around a license gate;
* a snapshot is ``verified`` only after file sizes and LFS SHA-256 hashes match
  the Hub's manifest for the pinned commit;
* "downloaded" never means "loadable": that is the lifecycle manager's call;
* removal only deletes cache revisions aivoice registered, and never a revision
  another registry entry still points at.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .. import settings
from ..paths import registry_dir
from .registry import (
    PIPELINE_TAGS,
    Assessment,
    KnownModel,
    assess,
    find_known,
)

GIB = 1024**3


class HubError(RuntimeError):
    pass


class HubUnavailable(HubError):
    """huggingface_hub is not installed, or the Hub cannot be reached."""


class GatedModelError(HubError):
    pass


class DownloadCancelled(HubError):
    pass


class ConfirmationRequired(HubError):
    def __init__(self, plan: "DownloadPlan") -> None:
        super().__init__(
            f"{plan.repo_id}@{plan.commit[:10] if plan.commit else plan.revision}: "
            f"{_fmt_bytes(plan.download_bytes)} to download "
            f"(total {_fmt_bytes(plan.total_bytes)}). Re-run with --yes to confirm."
        )
        self.plan = plan


def _fmt_bytes(n: int | None) -> str:
    if n is None:
        return "unknown size"
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024  # type: ignore[assignment]
    return f"{n} B"


fmt_bytes = _fmt_bytes


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _require_hub():
    try:
        import huggingface_hub  # type: ignore
    except ImportError as e:
        raise HubUnavailable("huggingface_hub is not installed: pip install 'aivoice[hf]'") from e
    return huggingface_hub


# --- data -------------------------------------------------------------------


@dataclass
class ModelSummary:
    repo_id: str
    author: str | None
    downloads: int | None
    likes: int | None
    pipeline_tag: str | None
    library_name: str | None
    tags: list[str]
    gated: Any
    last_modified: str | None
    known: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FileEntry:
    path: str
    size: int | None
    sha256: str | None  # LFS sha256, None for small git-tracked files


@dataclass
class ModelDetails:
    repo_id: str
    commit: str | None
    license: str | None
    gated: Any
    private: bool | None
    pipeline_tag: str | None
    library_name: str | None
    tags: list[str]
    downloads: int | None
    last_modified: str | None
    params: int | None  # from safetensors metadata when the Hub reports it (verified)
    files: list[FileEntry]
    total_bytes: int | None
    assessment: Assessment
    known: KnownModel | None = None
    installed: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["assessment"] = self.assessment.to_dict()
        d["known"] = asdict(self.known) if self.known else None
        return d


@dataclass
class DownloadPlan:
    repo_id: str
    revision: str
    commit: str | None
    files: list[FileEntry]
    total_bytes: int | None
    download_bytes: int | None
    disk_free_bytes: int
    gated: Any
    allow_patterns: list[str] | None
    warnings: list[str] = field(default_factory=list)

    @property
    def fits_disk(self) -> bool:
        need = self.download_bytes or 0
        return need + 512 * 1024**2 < self.disk_free_bytes

    def needs_confirmation(self, threshold_gib: float) -> bool:
        return (self.download_bytes is None) or (self.download_bytes > threshold_gib * GIB)


@dataclass
class VerifyReport:
    repo_id: str
    commit: str | None
    ok: bool
    checked: int
    problems: list[str]
    hashed: int = 0


class _Progress:
    """Aggregated byte progress across huggingface_hub's per-file tqdm bars."""

    def __init__(self, cb: Callable[[dict[str, Any]], None] | None, cancel: threading.Event | None) -> None:
        self.cb = cb
        self.cancel = cancel
        self.lock = threading.Lock()
        self.done = 0
        self.total = 0
        self.t0 = time.monotonic()

    def tqdm_class(self):
        from tqdm.auto import tqdm  # huggingface_hub depends on tqdm

        prog = self

        class _T(tqdm):  # type: ignore[misc, valid-type]
            def __init__(self, *a, **kw):
                self._aiv_bytes = kw.get("unit") == "B"
                kw["disable"] = True
                super().__init__(*a, **kw)
                total = kw.get("total")
                if self._aiv_bytes and total:
                    with prog.lock:
                        prog.total += int(total)

            def update(self, n=1):
                if prog.cancel is not None and prog.cancel.is_set():
                    raise DownloadCancelled("download cancelled")
                if self._aiv_bytes:
                    prog.advance(int(n or 0))
                return super().update(n)

        return _T

    def advance(self, n: int) -> None:
        with self.lock:
            self.done += n
            done, total = self.done, self.total
        if self.cb:
            elapsed = max(1e-6, time.monotonic() - self.t0)
            speed = done / elapsed
            eta = (total - done) / speed if total and speed > 0 else None
            self.cb({"done": done, "total": total or None, "speed": speed, "eta_s": eta})


# --- manager ----------------------------------------------------------------


class HuggingFaceModelManager:
    def __init__(
        self,
        *,
        api: Any | None = None,
        snapshot_download: Callable[..., Any] | None = None,
        scan_cache_dir: Callable[..., Any] | None = None,
        cache_dir: str | Path | None = None,
        registry_path: Path | None = None,
        offline: bool | None = None,
    ) -> None:
        self._api = api
        self._snapshot_download = snapshot_download
        self._scan_cache_dir = scan_cache_dir
        cfg_cache = settings.get("hf.cache_dir")
        self.cache_dir = Path(cache_dir) if cache_dir else (Path(cfg_cache) if cfg_cache else None)
        self.registry_path = registry_path or (registry_dir() / "registry.json")
        if offline is None:
            offline = os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in ("1", "true", "yes")
        self.offline = offline
        self._lock = threading.Lock()

    # --- lazily bound library calls (injectable for tests) ---------------

    @property
    def api(self):
        if self._api is None:
            hub = _require_hub()
            self._api = hub.HfApi(library_name="aivoice", library_version=_version())
        return self._api

    def _snapshot(self, **kw):
        fn = self._snapshot_download
        if fn is None:
            fn = _require_hub().snapshot_download
        if self.cache_dir:
            kw.setdefault("cache_dir", str(self.cache_dir))
        kw.setdefault("library_name", "aivoice")
        kw.setdefault("library_version", _version())
        return fn(**kw)

    def _scan(self):
        fn = self._scan_cache_dir
        if fn is None:
            fn = _require_hub().scan_cache_dir
        return fn(self.cache_dir) if self.cache_dir else fn()

    # --- registry --------------------------------------------------------

    def _load_registry(self) -> dict[str, Any]:
        if self.registry_path.is_file():
            try:
                data = json.loads(self.registry_path.read_text(encoding="utf-8"))
                return data if isinstance(data, dict) else {}
            except json.JSONDecodeError:
                return {}
        return {}

    def _save_registry(self, data: dict[str, Any]) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.registry_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, self.registry_path)

    def installed(self) -> dict[str, dict[str, Any]]:
        return self._load_registry()

    def _find(self, name: str) -> tuple[str | None, dict[str, Any] | None]:
        reg = self._load_registry()
        if name in reg:
            return name, reg[name]
        known = find_known(name)
        for key in filter(None, (known.repo_id if known else None, known.key if known else None)):
            if key in reg:
                return key, reg[key]
        return None, None

    def get_entry(self, name: str) -> dict[str, Any] | None:
        return self._find(name)[1]

    def _register(self, key: str, entry: dict[str, Any]) -> None:
        with self._lock:
            reg = self._load_registry()
            reg[key] = entry
            self._save_registry(reg)

    def _unregister(self, key: str) -> None:
        with self._lock:
            reg = self._load_registry()
            reg.pop(key, None)
            self._save_registry(reg)

    # --- discovery -------------------------------------------------------

    def search(
        self,
        query: str | None = None,
        *,
        author: str | None = None,
        category: str | None = None,
        library: str | None = None,
        language: str | None = None,
        license: str | None = None,
        quantization: str | None = None,
        sort: str = "downloads",
        limit: int = 20,
    ) -> list[ModelSummary]:
        if self.offline:
            raise HubUnavailable("offline mode (HF_HUB_OFFLINE=1): search needs the Hub")
        filters: list[str] = []
        if library:
            filters.append(library)
        if language:
            filters.append(language)
        if license:
            filters.append(f"license:{license}")
        if quantization and quantization.lower() == "gguf":
            filters.append("gguf")
        pipeline = PIPELINE_TAGS.get(category or "", [None])[0] if category else None
        try:
            rows = self.api.list_models(
                search=query or None,
                author=author,
                filter=filters or None,
                pipeline_tag=pipeline,
                sort=sort,
                limit=limit,
                full=False,
            )
            out = []
            for m in rows:
                known = find_known(m.id)
                out.append(
                    ModelSummary(
                        repo_id=m.id,
                        author=getattr(m, "author", None) or m.id.split("/")[0],
                        downloads=getattr(m, "downloads", None),
                        likes=getattr(m, "likes", None),
                        pipeline_tag=getattr(m, "pipeline_tag", None),
                        library_name=getattr(m, "library_name", None),
                        tags=list(getattr(m, "tags", None) or []),
                        gated=getattr(m, "gated", None),
                        last_modified=str(getattr(m, "last_modified", "") or "") or None,
                        known=known.key if known else None,
                    )
                )
            if quantization and quantization.lower() != "gguf":
                q = quantization.lower()
                out = [r for r in out if q in r.repo_id.lower() or any(q in t.lower() for t in r.tags)]
            return out
        except HubError:
            raise
        except Exception as e:  # noqa: BLE001
            raise HubUnavailable(f"Hub search failed: {e}") from e

    def info(self, repo_id: str, *, revision: str | None = None) -> ModelDetails:
        known = find_known(repo_id)
        repo = known.repo_id if known and known.repo_id else repo_id
        if known and known.repo_id is None:
            raise HubError(
                f"{known.key} is not distributed through the Hugging Face Hub ({known.source}). "
                f"Use `aivoice models download {known.key}`."
            )
        installed = self.get_entry(repo)
        if self.offline:
            if not installed:
                raise HubUnavailable(f"offline and {repo} is not installed")
            return self._details_from_entry(repo, installed, known)
        try:
            mi = self.api.model_info(repo, revision=revision, files_metadata=True)
        except Exception as e:  # noqa: BLE001
            name = type(e).__name__
            if name in ("RepositoryNotFoundError",):
                raise HubError(f"repository not found (or private without access): {repo}") from e
            if name in ("RevisionNotFoundError",):
                raise HubError(f"revision not found: {repo}@{revision}") from e
            if name == "GatedRepoError":
                raise GatedModelError(_gated_message(repo)) from e
            if installed:
                return self._details_from_entry(repo, installed, known)
            raise HubUnavailable(f"could not read {repo} from the Hub: {e}") from e
        files = [
            FileEntry(
                path=s.rfilename,
                size=getattr(s, "size", None),
                sha256=(getattr(s, "lfs", None) or {}).get("sha256")
                if isinstance(getattr(s, "lfs", None), dict)
                else getattr(getattr(s, "lfs", None), "sha256", None),
            )
            for s in (mi.siblings or [])
        ]
        sizes = [f.size for f in files]
        total = sum(s for s in sizes if s) if all(s is not None for s in sizes) else None
        card = getattr(mi, "card_data", None)
        lic = None
        if card is not None:
            lic = card.get("license") if hasattr(card, "get") else getattr(card, "license", None)
        if not lic:
            lic = next((t.split(":", 1)[1] for t in (mi.tags or []) if t.startswith("license:")), None)
        st = getattr(mi, "safetensors", None)
        params = getattr(st, "total", None) if st is not None else None
        a = assess(
            files=[f.path for f in files],
            library_name=getattr(mi, "library_name", None),
            pipeline_tag=getattr(mi, "pipeline_tag", None),
            tags=list(mi.tags or []),
            config=getattr(mi, "config", None) or None,
            repo_id=repo,
        )
        return ModelDetails(
            repo_id=repo,
            commit=getattr(mi, "sha", None),
            license=lic,
            gated=getattr(mi, "gated", None),
            private=getattr(mi, "private", None),
            pipeline_tag=getattr(mi, "pipeline_tag", None),
            library_name=getattr(mi, "library_name", None),
            tags=list(mi.tags or []),
            downloads=getattr(mi, "downloads", None),
            last_modified=str(getattr(mi, "last_modified", "") or "") or None,
            params=params,
            files=files,
            total_bytes=total,
            assessment=a,
            known=known,
            installed=installed,
        )

    def _details_from_entry(self, repo: str, entry: dict[str, Any], known: KnownModel | None) -> ModelDetails:
        files = [FileEntry(**f) for f in entry.get("files", [])]
        a = assess(files=[f.path for f in files], repo_id=repo, category_hint=entry.get("category"))
        return ModelDetails(
            repo_id=repo,
            commit=entry.get("commit"),
            license=entry.get("license"),
            gated=entry.get("gated"),
            private=None,
            pipeline_tag=None,
            library_name=None,
            tags=[],
            downloads=None,
            last_modified=None,
            params=None,
            files=files,
            total_bytes=entry.get("size_bytes"),
            assessment=a,
            known=known,
            installed=entry,
        )

    # --- download --------------------------------------------------------

    def plan(
        self,
        repo_id: str,
        *,
        revision: str | None = None,
        allow_patterns: list[str] | None = None,
    ) -> DownloadPlan:
        known = find_known(repo_id)
        repo = known.repo_id if known and known.repo_id else repo_id
        if allow_patterns is None and known:
            allow_patterns = known.allow_patterns
        details = self.info(repo, revision=revision)
        files = details.files
        if allow_patterns:
            import fnmatch

            files = [f for f in files if any(fnmatch.fnmatch(f.path, p) or fnmatch.fnmatch(f.path.lower(), p.lower()) for p in allow_patterns)]
        sizes = [f.size for f in files]
        total = sum(s for s in sizes if s) if all(s is not None for s in sizes) else None
        download = total
        warnings = list(details.assessment.warnings)
        # Ask the library what is already cached (dry run downloads nothing).
        try:
            dry = self._snapshot(
                repo_id=repo,
                revision=details.commit or revision,
                allow_patterns=allow_patterns,
                dry_run=True,
            )
            if isinstance(dry, list):
                download = sum(int(getattr(d, "file_size", 0) or 0) for d in dry if getattr(d, "will_download", True))
        except TypeError:
            pass  # older huggingface_hub without dry_run: assume everything downloads
        except Exception as e:  # noqa: BLE001
            if type(e).__name__ == "GatedRepoError":
                raise GatedModelError(_gated_message(repo)) from e
            warnings.append(f"could not compute cached bytes ({type(e).__name__}); assuming full download")
        if allow_patterns and not files:
            warnings.append(f"no files match {allow_patterns}")
        target = self.cache_dir or _default_hf_cache()
        target.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(target).free
        quota = settings.get("hf.quota_gib")
        if quota and download:
            used = self.cache_size_bytes()
            if used + download > float(quota) * GIB:
                warnings.append(
                    f"would exceed the configured cache quota ({quota} GiB; {_fmt_bytes(used)} used)"
                )
        if details.gated:
            warnings.append("gated repository: requires accepting the license on the Hub and a token")
        return DownloadPlan(
            repo_id=repo,
            revision=revision or "main",
            commit=details.commit,
            files=files,
            total_bytes=total,
            download_bytes=download,
            disk_free_bytes=free,
            gated=details.gated,
            allow_patterns=allow_patterns,
            warnings=warnings,
        )

    def download(
        self,
        repo_id: str,
        *,
        revision: str | None = None,
        allow_patterns: list[str] | None = None,
        confirm: bool = False,
        progress: Callable[[dict[str, Any]], None] | None = None,
        cancel: threading.Event | None = None,
        max_retries: int = 4,
        verify: bool = True,
        plan: DownloadPlan | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> dict[str, Any]:
        known = find_known(repo_id)
        if known and known.repo_id is None:
            return self._download_native(known, confirm=confirm)
        plan = plan or self.plan(repo_id, revision=revision, allow_patterns=allow_patterns)
        threshold = float(settings.get("hf.confirm_over_gib") or 2.0)
        if not plan.fits_disk:
            raise HubError(
                f"not enough disk space: need {_fmt_bytes(plan.download_bytes)}, "
                f"{_fmt_bytes(plan.disk_free_bytes)} free"
            )
        if plan.needs_confirmation(threshold) and not confirm:
            raise ConfirmationRequired(plan)
        if any("exceed the configured cache quota" in w for w in plan.warnings) and not confirm:
            raise ConfirmationRequired(plan)
        prog = _Progress(progress, cancel)
        last: Exception | None = None
        path: str | None = None
        for attempt in range(1, max_retries + 1):
            if cancel is not None and cancel.is_set():
                raise DownloadCancelled("download cancelled")
            try:
                kw: dict[str, Any] = dict(
                    repo_id=plan.repo_id,
                    revision=plan.commit or plan.revision,
                    allow_patterns=plan.allow_patterns,
                    local_files_only=self.offline,
                )
                try:
                    kw["tqdm_class"] = prog.tqdm_class()
                except ImportError:
                    pass
                path = str(self._snapshot(**kw))
                break
            except DownloadCancelled:
                raise
            except Exception as e:  # noqa: BLE001
                name = type(e).__name__
                if name == "GatedRepoError":
                    raise GatedModelError(_gated_message(plan.repo_id)) from e
                if name in ("RepositoryNotFoundError", "RevisionNotFoundError", "LocalEntryNotFoundError"):
                    raise HubError(f"{name}: {e}") from e
                if isinstance(e.__context__, DownloadCancelled) or "cancelled" in str(e).lower():
                    raise DownloadCancelled("download cancelled") from e
                last = e
                if attempt < max_retries:
                    sleep(min(2**attempt, 16))
        if path is None:
            raise HubError(f"download failed after {max_retries} attempts (partial files are kept for resume): {last}")
        details_known = known
        a = assess(files=[f.path for f in plan.files], repo_id=plan.repo_id)
        entry = {
            "repo_id": plan.repo_id,
            "key": details_known.key if details_known else None,
            "source": "huggingface",
            "requested_revision": plan.revision,
            "commit": plan.commit,
            "path": path,
            "category": a.category,
            "runtime": a.runtime,
            "adapter": a.adapter,
            "allow_patterns": plan.allow_patterns,
            "files": [asdict(f) for f in plan.files],
            "size_bytes": plan.total_bytes,
            "gated": plan.gated,
            "installed_at": _now(),
            "verified": False,
            "verified_at": None,
        }
        self._register(plan.repo_id, entry)
        if verify:
            rep = self.verify(plan.repo_id)
            entry["verified"] = rep.ok
            if not rep.ok:
                entry["problems"] = rep.problems
        return self.get_entry(plan.repo_id) or entry

    def _download_native(self, known: KnownModel, *, confirm: bool) -> dict[str, Any]:
        """Models fetched by their runtime's own downloader (Moonshine Voice)."""
        if known.runtime != "moonshine-voice":
            raise HubError(f"{known.key}: {known.source}; nothing to download")
        if not confirm:
            raise HubError(
                f"{known.key} is downloaded by moonshine-voice from download.moonshine.ai "
                "(size not reported in advance; tens to a few hundred MB). Re-run with --yes."
            )
        try:
            import moonshine_voice as mv  # type: ignore
        except ImportError as e:
            raise HubUnavailable("moonshine-voice is not installed: pip install 'aivoice[moonshine]'") from e
        arch = getattr(mv.ModelArch, known.key.split("-", 1)[1].upper() + "_STREAMING")
        lang = settings.get("asr.language") or "en"
        path, got_arch = mv.get_model_for_language(lang, arch)
        entry = {
            "repo_id": None,
            "key": known.key,
            "source": known.source,
            "commit": None,
            "path": str(path),
            "category": "asr",
            "runtime": "moonshine-voice",
            "adapter": "moonshine",
            "language": lang,
            "model_arch": int(got_arch),
            "files": [],
            "installed_at": _now(),
            "verified": Path(path).exists(),
            "verified_at": _now(),
        }
        self._register(known.key, entry)
        return entry

    # --- verification ----------------------------------------------------

    def verify(self, name: str, *, hash_files: bool = True) -> VerifyReport:
        entry = self.get_entry(name)
        if not entry:
            raise HubError(f"not installed: {name}")
        root = Path(entry["path"])
        problems: list[str] = []
        checked = hashed = 0
        if not root.exists():
            problems.append(f"snapshot path missing: {root}")
        for f in entry.get("files", []):
            p = root / f["path"]
            checked += 1
            if not p.is_file():
                problems.append(f"missing file: {f['path']}")
                continue
            if f.get("size") is not None and p.stat().st_size != f["size"]:
                problems.append(f"size mismatch: {f['path']} ({p.stat().st_size} != {f['size']})")
                continue
            if hash_files and f.get("sha256"):
                hashed += 1
                if _sha256(p) != f["sha256"]:
                    problems.append(f"sha256 mismatch: {f['path']}")
        if entry.get("source") == "huggingface" and not entry.get("files"):
            problems.append("no file manifest recorded; re-download to verify")
        ok = not problems
        entry["verified"] = ok
        entry["verified_at"] = _now()
        if problems:
            entry["problems"] = problems
        else:
            entry.pop("problems", None)
        self._register(self._find(name)[0] or name, entry)
        return VerifyReport(repo_id=name, commit=entry.get("commit"), ok=ok, checked=checked, problems=problems, hashed=hashed)

    # --- import / update / remove ---------------------------------------

    def import_path(self, path: Path, *, category: str | None = None, name: str | None = None) -> dict[str, Any]:
        path = Path(path).expanduser().resolve()
        if not path.exists():
            raise HubError(f"no such path: {path}")
        root = path if path.is_dir() else path.parent
        files = [path.name] if path.is_file() else [
            str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and ".git" not in p.parts
        ]
        cfg = None
        if (root / "config.json").is_file():
            try:
                cfg = json.loads((root / "config.json").read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                cfg = None
        a = assess(files=files, config=cfg, category_hint=category)
        key = name or f"local/{path.name}"
        entry = {
            "repo_id": None,
            "key": key,
            "source": "import",
            "commit": None,
            "path": str(root),
            "file": str(path) if path.is_file() else None,
            "category": a.category,
            "runtime": a.runtime,
            "adapter": a.adapter,
            "files": [
                {"path": f, "size": (root / f).stat().st_size, "sha256": None} for f in files[:5000]
            ],
            "installed_at": _now(),
            "verified": True,
            "verified_at": _now(),
            "warnings": a.warnings,
        }
        self._register(key, entry)
        return entry

    def check_update(self, name: str) -> tuple[str | None, str | None]:
        entry = self.get_entry(name)
        if not entry or entry.get("source") != "huggingface":
            raise HubError(f"{name} is not a Hub-installed model")
        mi = self.api.model_info(entry["repo_id"], revision=entry.get("requested_revision") or "main")
        return entry.get("commit"), getattr(mi, "sha", None)

    def update(self, name: str, *, confirm: bool = False, remove_old: bool = False, **kw: Any) -> dict[str, Any]:
        entry = self.get_entry(name)
        if not entry:
            raise HubError(f"not installed: {name}")
        old, new = self.check_update(name)
        if old == new:
            return entry
        req = entry.get("requested_revision") or "main"
        if len(req) == 40 and req == old:
            return entry  # pinned to an exact commit: nothing to update
        fresh = self.download(entry["repo_id"], revision=req, allow_patterns=entry.get("allow_patterns"), confirm=confirm, **kw)
        if remove_old and old and fresh.get("verified"):
            self._delete_revisions(entry["repo_id"], [old])
        return fresh

    def remove(self, name: str, *, delete_files: bool = True) -> dict[str, Any]:
        key, entry = self._find(name)
        if not entry or key is None:
            raise HubError(f"not installed: {name}")
        freed = 0
        kept_reason = None
        if entry.get("source") == "huggingface" and delete_files and entry.get("commit"):
            others = [
                e for k, e in self._load_registry().items()
                if k != key and e.get("repo_id") == entry["repo_id"] and e.get("commit") == entry["commit"]
            ]
            if others:
                kept_reason = "another registry entry uses the same snapshot"
            else:
                freed = self._delete_revisions(entry["repo_id"], [entry["commit"]])
        elif entry.get("source") == "import":
            kept_reason = "imported paths are never deleted; only unregistered"
        elif entry.get("source", "").startswith("moonshine-voice"):
            kept_reason = "moonshine-voice manages its own cache; only unregistered"
        self._unregister(key)
        return {"removed": key, "freed_bytes": freed, "kept_files_reason": kept_reason}

    def _delete_revisions(self, repo_id: str, commits: list[str]) -> int:
        info = self._scan()
        targets: list[str] = []
        for repo in info.repos:
            if repo.repo_id == repo_id and getattr(repo, "repo_type", "model") == "model":
                targets += [r.commit_hash for r in repo.revisions if r.commit_hash in commits]
        if not targets:
            return 0
        strategy = info.delete_revisions(*targets)
        freed = int(getattr(strategy, "expected_freed_size", 0) or 0)
        strategy.execute()
        return freed

    # --- cache -----------------------------------------------------------

    def cache_size_bytes(self) -> int:
        try:
            return int(self._scan().size_on_disk)
        except Exception:  # noqa: BLE001
            return 0

    def cache_report(self) -> dict[str, Any]:
        info = self._scan()
        reg = self._load_registry()
        managed = {(e.get("repo_id"), e.get("commit")) for e in reg.values()}
        repos = []
        for repo in sorted(info.repos, key=lambda r: -r.size_on_disk):
            revs = [
                {
                    "commit": r.commit_hash,
                    "size": r.size_on_disk,
                    "refs": sorted(r.refs),
                    "managed": (repo.repo_id, r.commit_hash) in managed,
                }
                for r in repo.revisions
            ]
            repos.append(
                {
                    "repo_id": repo.repo_id,
                    "type": getattr(repo, "repo_type", "model"),
                    "size": repo.size_on_disk,
                    "revisions": revs,
                    "managed": any(v["managed"] for v in revs),
                }
            )
        return {
            "cache_dir": str(getattr(info, "cache_dir", None) or self.cache_dir or _default_hf_cache()),
            "size": info.size_on_disk,
            "repos": repos,
            "warnings": [str(w) for w in getattr(info, "warnings", [])],
        }

    def cleanup(self, *, dry_run: bool = True) -> dict[str, Any]:
        """Old revisions of aivoice-managed repos that no registry entry pins. Never touches other repos."""
        info = self._scan()
        reg = self._load_registry()
        pinned = {(e.get("repo_id"), e.get("commit")) for e in reg.values()}
        managed_repos = {e.get("repo_id") for e in reg.values() if e.get("source") == "huggingface"}
        candidates = []
        for repo in info.repos:
            if repo.repo_id not in managed_repos:
                continue
            for r in repo.revisions:
                if (repo.repo_id, r.commit_hash) not in pinned:
                    candidates.append({"repo_id": repo.repo_id, "commit": r.commit_hash, "size": r.size_on_disk})
        freed = 0
        if candidates and not dry_run:
            strategy = info.delete_revisions(*[c["commit"] for c in candidates])
            freed = int(getattr(strategy, "expected_freed_size", 0) or 0)
            strategy.execute()
        unmanaged = [r.repo_id for r in info.repos if r.repo_id not in managed_repos]
        return {"dry_run": dry_run, "candidates": candidates, "freed_bytes": freed, "left_alone": unmanaged}


def _gated_message(repo: str) -> str:
    return (
        f"{repo} is a gated repository. Open https://huggingface.co/{repo}, accept its license "
        "with your account, then authenticate with `hf auth login` (or set HF_TOKEN). "
        "aivoice uses your own Hugging Face credentials and does not bypass access controls."
    )


def _default_hf_cache() -> Path:
    env = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if env:
        return Path(env)
    home = os.environ.get("HF_HOME")
    if home:
        return Path(home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _version() -> str:
    from .. import __version__

    return __version__
