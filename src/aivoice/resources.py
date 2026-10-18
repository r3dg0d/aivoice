"""Model memory scheduler: who sits on which device, and is there room for one more.

Budgets come from the hardware probe at start-up. Memory already in use by other
processes at that moment (a browser, a game) is treated as unavailable, then
every load reserves its estimate on top. Estimates are deliberately conservative
and always marked as estimates: the scheduler's job is to refuse a load that
cannot fit *before* CUDA does, not to promise one that can.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, TypeVar

from .hardware import HardwareInfo, detect

T = TypeVar("T")

DEFAULT_VRAM_HEADROOM_MIB = 1536  # CUDA context, allocator fragmentation, cuDNN workspaces
DEFAULT_RAM_HEADROOM_MIB = 2048


class InsufficientMemory(RuntimeError):
    pass


@dataclass
class Requirement:
    name: str
    gpu_mib: int | None  # None: cannot run on GPU
    cpu_mib: int | None  # None: cannot run on CPU
    prefer: str = "gpu"  # gpu | cpu
    priority: int = 0  # higher = placed first (TTS usually wins the GPU)


@dataclass
class Reservation:
    name: str
    device: str  # "cuda:0" | "cpu"
    mib: int
    active: bool = False
    last_used: float = field(default_factory=time.monotonic)
    unload: Callable[[], None] | None = None


@dataclass
class Placement:
    device: dict[str, str]
    rejected: dict[str, str]
    notes: list[str]

    @property
    def ok(self) -> bool:
        return not self.rejected


def is_oom_error(exc: BaseException) -> bool:
    name = type(exc).__name__
    msg = str(exc).lower()
    return name in ("OutOfMemoryError", "CudaOutOfMemoryError") or "out of memory" in msg


class ResourceScheduler:
    def __init__(
        self,
        hw: HardwareInfo | None = None,
        *,
        vram_headroom_mib: int = DEFAULT_VRAM_HEADROOM_MIB,
        ram_headroom_mib: int = DEFAULT_RAM_HEADROOM_MIB,
    ) -> None:
        self.hw = hw or detect()
        self.vram_headroom = vram_headroom_mib
        self.ram_headroom = ram_headroom_mib
        self._lock = threading.RLock()
        self._res: dict[str, Reservation] = {}
        self._budget: dict[str, int] = {"cpu": max(0, self.hw.ram_available_mib - ram_headroom_mib)}
        for g in self.hw.gpus:
            self._budget[f"cuda:{g.index}"] = max(0, g.free_mib - vram_headroom_mib)

    # --- queries ---------------------------------------------------------

    @property
    def devices(self) -> list[str]:
        return list(self._budget)

    def budget(self, device: str) -> int:
        return self._budget.get(device, 0)

    def reserved(self, device: str) -> int:
        with self._lock:
            return sum(r.mib for r in self._res.values() if r.device == device)

    def available(self, device: str) -> int:
        return self.budget(device) - self.reserved(device)

    def reservations(self) -> list[Reservation]:
        with self._lock:
            return list(self._res.values())

    def gpu_device(self) -> str | None:
        gpus = [d for d in self._budget if d.startswith("cuda:")]
        return max(gpus, key=self.available) if gpus else None

    # --- planning --------------------------------------------------------

    def plan(self, reqs: list[Requirement]) -> Placement:
        """Place each requirement without reserving anything (dry run)."""
        free = {d: self.available(d) for d in self._budget}
        device: dict[str, str] = {}
        rejected: dict[str, str] = {}
        notes: list[str] = []
        gpu = self.gpu_device()
        for r in sorted(reqs, key=lambda x: -x.priority):
            order = ["gpu", "cpu"] if r.prefer == "gpu" else ["cpu", "gpu"]
            placed = False
            for kind in order:
                if kind == "gpu":
                    if gpu is None or r.gpu_mib is None:
                        continue
                    if r.gpu_mib <= free[gpu]:
                        free[gpu] -= r.gpu_mib
                        device[r.name] = gpu
                        placed = True
                        break
                    notes.append(f"{r.name}: needs ~{r.gpu_mib} MiB on {gpu}, {free[gpu]} MiB left")
                else:
                    if r.cpu_mib is None:
                        continue
                    if r.cpu_mib <= free["cpu"]:
                        free["cpu"] -= r.cpu_mib
                        device[r.name] = "cpu"
                        placed = True
                        if r.prefer == "gpu" and r.gpu_mib is not None:
                            notes.append(f"{r.name}: placed on CPU (GPU full)")
                        break
                    notes.append(f"{r.name}: needs ~{r.cpu_mib} MiB RAM, {free['cpu']} MiB left")
            if not placed:
                if r.gpu_mib is None and r.cpu_mib is None:
                    rejected[r.name] = "no supported device"
                elif r.cpu_mib is None and gpu is None:
                    rejected[r.name] = "requires an NVIDIA GPU; none detected"
                else:
                    want = r.gpu_mib if r.cpu_mib is None else r.cpu_mib
                    rejected[r.name] = f"does not fit (needs ~{want} MiB)"
        return Placement(device=device, rejected=rejected, notes=notes)

    # --- reservations ----------------------------------------------------

    def reserve(
        self,
        name: str,
        device: str,
        mib: int,
        *,
        unload: Callable[[], None] | None = None,
        evict: bool = True,
    ) -> Reservation:
        with self._lock:
            if name in self._res:
                raise InsufficientMemory(f"{name} already holds a reservation on {self._res[name].device}")
            if device not in self._budget:
                raise InsufficientMemory(f"unknown device {device!r}; have {self.devices}")
            if mib > self.available(device) and evict:
                self._evict(device, mib - self.available(device))
            if mib > self.available(device):
                raise InsufficientMemory(
                    f"{name} needs ~{mib} MiB on {device} but only {self.available(device)} MiB is free "
                    f"(budget {self.budget(device)} MiB after headroom, {self.reserved(device)} MiB reserved)"
                )
            r = Reservation(name=name, device=device, mib=mib, unload=unload)
            self._res[name] = r
            return r

    def release(self, name: str) -> None:
        with self._lock:
            self._res.pop(name, None)

    def touch(self, name: str, *, active: bool | None = None) -> None:
        with self._lock:
            r = self._res.get(name)
            if r:
                r.last_used = time.monotonic()
                if active is not None:
                    r.active = active

    def _evict(self, device: str, need_mib: int) -> list[str]:
        """Unload inactive models on *device*, least recently used first."""
        evicted: list[str] = []
        candidates = sorted(
            (r for r in self._res.values() if r.device == device and not r.active and r.unload),
            key=lambda r: r.last_used,
        )
        freed = 0
        for r in candidates:
            if freed >= need_mib:
                break
            try:
                r.unload()  # type: ignore[misc]
            except Exception:  # noqa: BLE001
                continue
            self._res.pop(r.name, None)
            freed += r.mib
            evicted.append(r.name)
        return evicted

    def run_with_oom_recovery(self, device: str, fn: Callable[[], T], *, need_mib: int = 0) -> T:
        """Run *fn*; on a CUDA OOM, evict inactive models on that device and retry once."""
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if not is_oom_error(e):
                raise
            with self._lock:
                evicted = self._evict(device, need_mib or 1)
            _empty_cuda_cache()
            if not evicted:
                raise InsufficientMemory(
                    f"CUDA out of memory on {device} and nothing inactive to unload: {e}"
                ) from e
            return fn()


def _empty_cuda_cache() -> None:
    try:
        import importlib.util

        if importlib.util.find_spec("torch") is None:
            return
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def format_placement(p: Placement) -> str:
    lines = [f"  {name:<24} -> {dev}" for name, dev in p.device.items()]
    lines += [f"  {name:<24} !! {why}" for name, why in p.rejected.items()]
    lines += [f"  note: {n}" for n in p.notes]
    return "\n".join(lines)
