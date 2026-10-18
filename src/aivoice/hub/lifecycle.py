"""Model lifecycle: NOT_INSTALLED -> ... -> READY -> IN_USE, with one load at a time per model.

Loading is delegated to *loaders*: one function per adapter that turns a
registry entry into a :class:`LoadedModel`. A loader is the only place a model
file is opened, so "is this compatible?" has one answer (the registry's
assessment) and "did it load?" has another (the loader succeeded and warmup
ran). Memory is reserved with the :class:`~aivoice.resources.ResourceScheduler`
before a loader runs, so a model that cannot fit is refused instead of
crashing CUDA halfway through.
"""

from __future__ import annotations

import enum
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from ..resources import InsufficientMemory, Requirement, ResourceScheduler
from .registry import find_known


class ModelState(str, enum.Enum):
    NOT_INSTALLED = "NOT_INSTALLED"
    DOWNLOADING = "DOWNLOADING"
    INSTALLED = "INSTALLED"
    VALIDATING = "VALIDATING"
    LOADING = "LOADING"
    READY = "READY"
    IN_USE = "IN_USE"
    UNLOADING = "UNLOADING"
    ERROR = "ERROR"


class LifecycleError(RuntimeError):
    pass


@dataclass
class LoadedModel:
    name: str
    handle: Any
    device: str
    reserved_mib: int
    unload: Callable[[], None] = lambda: None
    warmup: Callable[[], float] | None = None  # returns warmup ms
    warmup_ms: float | None = None
    load_ms: float | None = None
    info: dict[str, Any] = field(default_factory=dict)


Loader = Callable[[dict[str, Any], str], LoadedModel]


@dataclass
class _Slot:
    state: ModelState = ModelState.INSTALLED
    model: LoadedModel | None = None
    error: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    users: int = 0


class ModelLifecycleManager:
    def __init__(self, scheduler: ResourceScheduler | None = None) -> None:
        self.scheduler = scheduler or ResourceScheduler()
        self._loaders: dict[str, Loader] = {}
        self._slots: dict[str, _Slot] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="aivoice-load")

    def register_loader(self, adapter: str, loader: Loader) -> None:
        self._loaders[adapter] = loader

    def _slot(self, name: str) -> _Slot:
        with self._lock:
            return self._slots.setdefault(name, _Slot())

    def state(self, name: str) -> ModelState:
        with self._lock:
            slot = self._slots.get(name)
        return slot.state if slot else ModelState.NOT_INSTALLED

    def loaded(self) -> dict[str, LoadedModel]:
        with self._lock:
            return {k: s.model for k, s in self._slots.items() if s.model is not None}

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            items = list(self._slots.items())
        out = []
        for name, s in items:
            m = s.model
            out.append(
                {
                    "name": name,
                    "state": s.state.value,
                    "device": m.device if m else None,
                    "reserved_mib": m.reserved_mib if m else 0,
                    "load_ms": m.load_ms if m else None,
                    "warmup_ms": m.warmup_ms if m else None,
                    "users": s.users,
                    "error": s.error,
                }
            )
        return out

    # --- load / unload ---------------------------------------------------

    def requirement_for(self, name: str, entry: dict[str, Any]) -> Requirement:
        known = find_known(entry.get("key") or entry.get("repo_id") or name)
        gpu = known.est_vram_mib if known else entry.get("est_vram_mib")
        cpu = known.est_ram_mib if known else entry.get("est_ram_mib")
        if gpu is None and cpu is None:
            cpu = int((entry.get("size_bytes") or 512 * 1024**2) / 1024**2 * 1.3)
        prefer = entry.get("prefer") or ("cpu" if (entry.get("category") in ("asr", "llm", "vad")) else "gpu")
        return Requirement(name=name, gpu_mib=gpu, cpu_mib=cpu, prefer=prefer)

    def load(
        self,
        name: str,
        entry: dict[str, Any],
        *,
        device: str | None = None,
        warmup: bool = True,
    ) -> LoadedModel:
        adapter = entry.get("adapter")
        if not adapter:
            raise LifecycleError(
                f"{name}: no aivoice adapter for runtime {entry.get('runtime')!r} / "
                f"category {entry.get('category')!r}; it can be inspected but not loaded"
            )
        loader = self._loaders.get(adapter)
        if loader is None:
            raise LifecycleError(f"{name}: adapter {adapter!r} is not available in this process")
        if entry.get("source") == "huggingface" and not entry.get("verified"):
            raise LifecycleError(f"{name}: snapshot is not verified; run `aivoice models verify {name}`")
        slot = self._slot(name)
        if not slot.lock.acquire(blocking=False):
            raise LifecycleError(f"{name}: another load/unload is in progress")
        try:
            if slot.model is not None:
                return slot.model
            slot.state = ModelState.VALIDATING
            req = self.requirement_for(name, entry)
            if device is None:
                placement = self.scheduler.plan([req])
                if not placement.ok:
                    slot.state = ModelState.ERROR
                    slot.error = placement.rejected.get(name, "does not fit")
                    raise InsufficientMemory(f"{name}: {slot.error}")
                device = placement.device[name]
            mib = req.gpu_mib if device.startswith("cuda") else req.cpu_mib
            if mib is None:
                slot.state = ModelState.ERROR
                slot.error = f"cannot run on {device}"
                raise LifecycleError(f"{name}: cannot run on {device}")
            self.scheduler.reserve(name, device, mib, unload=lambda n=name: self.unload(n, force=True))
            slot.state = ModelState.LOADING
            t0 = time.monotonic()
            try:
                model = self.scheduler.run_with_oom_recovery(device, lambda: loader(entry, device), need_mib=mib)
            except Exception as e:
                self.scheduler.release(name)
                slot.state = ModelState.ERROR
                slot.error = str(e)
                raise
            model.load_ms = (time.monotonic() - t0) * 1000
            model.reserved_mib = mib
            if warmup and model.warmup:
                try:
                    model.warmup_ms = model.warmup()
                except Exception as e:  # noqa: BLE001
                    model.unload()
                    self.scheduler.release(name)
                    slot.state = ModelState.ERROR
                    slot.error = f"warmup failed: {e}"
                    raise LifecycleError(f"{name}: warmup failed: {e}") from e
            slot.model = model
            slot.state = ModelState.READY
            slot.error = None
            return model
        finally:
            slot.lock.release()

    def load_async(self, name: str, entry: dict[str, Any], **kw: Any) -> Future:
        return self._pool.submit(self.load, name, entry, **kw)

    def unload(self, name: str, *, force: bool = False) -> bool:
        slot = self._slot(name)
        if slot.users and not force:
            raise LifecycleError(f"{name} is in use by {slot.users} pipeline(s); stop the pipeline first")
        got = slot.lock.acquire(timeout=0 if force else 30)
        try:
            if slot.model is None:
                return False
            slot.state = ModelState.UNLOADING
            try:
                slot.model.unload()
            finally:
                slot.model = None
                slot.users = 0
                self.scheduler.release(name)
                slot.state = ModelState.INSTALLED
            return True
        finally:
            if got:
                slot.lock.release()

    def acquire(self, name: str) -> LoadedModel:
        slot = self._slot(name)
        if slot.model is None:
            raise LifecycleError(f"{name} is not loaded")
        slot.users += 1
        slot.state = ModelState.IN_USE
        self.scheduler.touch(name, active=True)
        return slot.model

    def release(self, name: str) -> None:
        slot = self._slot(name)
        slot.users = max(0, slot.users - 1)
        if slot.model is not None and slot.users == 0:
            slot.state = ModelState.READY
            self.scheduler.touch(name, active=False)

    def switch(self, old: str | None, new: str, entry: dict[str, Any], **kw: Any) -> LoadedModel:
        """Load *new* first, then drop *old*; the caller swaps users between utterances."""
        model = self.load(new, entry, **kw)
        if old and old != new:
            try:
                self.unload(old)
            except LifecycleError:
                pass
        return model

    def shutdown(self) -> None:
        for name in list(self.loaded()):
            try:
                self.unload(name, force=True)
            except Exception:  # noqa: BLE001
                pass
        self._pool.shutdown(wait=False)
