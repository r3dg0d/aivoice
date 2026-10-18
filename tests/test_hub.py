"""HuggingFaceModelManager against fake Hub objects (no network, no real downloads)."""

import hashlib
import importlib.util
import threading
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from aivoice.hub.lifecycle import LifecycleError, LoadedModel, ModelLifecycleManager, ModelState
from aivoice.hub.manager import (
    ConfirmationRequired,
    DownloadCancelled,
    GatedModelError,
    HubError,
    HuggingFaceModelManager,
)
from aivoice.hub.registry import assess, find_known
from aivoice.hardware import mock_rtx4090
from aivoice.resources import InsufficientMemory, ResourceScheduler

FILES = {"model.gguf": b"G" * 4096, "README.md": b"# readme\n"}
COMMIT = "a" * 40


class GatedRepoError(Exception):
    pass


class RepositoryNotFoundError(Exception):
    pass


@dataclass
class Sib:
    rfilename: str
    size: int
    lfs: dict | None


@dataclass
class MI:
    id: str
    sha: str = COMMIT
    siblings: list = field(default_factory=list)
    tags: list = field(default_factory=lambda: ["gguf", "license:apache-2.0"])
    gated: object = False
    private: bool = False
    pipeline_tag: str = "text-generation"
    library_name: str | None = None
    downloads: int = 10
    likes: int = 1
    last_modified: str = "2026-10-01"
    card_data: dict | None = None
    safetensors: object = None
    config: dict | None = None
    author: str = "Qwen"


class FakeApi:
    def __init__(self, files=FILES, gated=False, missing=False):
        self.files = files
        self.gated = gated
        self.missing = missing
        self.calls = []

    def list_models(self, **kw):
        self.calls.append(kw)
        return [MI("Qwen/Qwen2.5-1.5B-Instruct-GGUF"), MI("someone/random-tts", pipeline_tag="text-to-speech", tags=[])]

    def model_info(self, repo, revision=None, files_metadata=False):
        if self.missing:
            raise RepositoryNotFoundError(repo)
        if self.gated:
            raise GatedRepoError(repo)
        sibs = [Sib(n, len(b), {"sha256": hashlib.sha256(b).hexdigest()} if n.endswith(".gguf") else None) for n, b in self.files.items()]
        return MI(repo, siblings=sibs, card_data={"license": "apache-2.0"})


class FakeSnapshot:
    """Writes files into a content-addressed-looking tree; supports dry_run, tqdm_class, failures."""

    def __init__(self, root: Path, files=FILES, fail_times=0):
        self.root = root
        self.files = files
        self.fail_times = fail_times
        self.calls = []

    def __call__(self, repo_id, revision=None, allow_patterns=None, dry_run=False, tqdm_class=None, **kw):
        import fnmatch

        self.calls.append({"repo_id": repo_id, "revision": revision, "dry_run": dry_run, **kw})
        names = [n for n in self.files if not allow_patterns or any(fnmatch.fnmatch(n.lower(), p.lower()) for p in allow_patterns)]
        snap = self.root / repo_id.replace("/", "--") / "snapshots" / (revision or COMMIT)
        if dry_run:
            @dataclass
            class D:
                filename: str
                file_size: int
                will_download: bool

            return [D(n, len(self.files[n]), not (snap / n).exists()) for n in names]
        if self.fail_times:
            self.fail_times -= 1
            raise ConnectionError("network blip")
        snap.mkdir(parents=True, exist_ok=True)
        for n in names:
            if tqdm_class is not None:
                bar = tqdm_class(total=len(self.files[n]), unit="B")
                bar.update(len(self.files[n]))
            (snap / n).write_bytes(self.files[n])
        return str(snap)


@dataclass
class Rev:
    commit_hash: str
    size_on_disk: int
    refs: set = field(default_factory=set)


@dataclass
class Repo:
    repo_id: str
    revisions: list
    size_on_disk: int
    repo_type: str = "model"


class Strategy:
    def __init__(self, info, revs):
        self.info, self.revs = info, revs
        self.expected_freed_size = 123

    def execute(self):
        self.info.deleted.extend(self.revs)


class CacheInfo:
    def __init__(self, repos):
        self.repos = repos
        self.size_on_disk = sum(r.size_on_disk for r in repos)
        self.deleted = []
        self.warnings = []

    def delete_revisions(self, *revs):
        return Strategy(self, list(revs))


def make(tmp_path, **kw):
    snap = FakeSnapshot(tmp_path / "hub", fail_times=kw.pop("fail_times", 0))
    cache = kw.pop("cache", CacheInfo([]))
    mgr = HuggingFaceModelManager(api=FakeApi(**kw), snapshot_download=snap, scan_cache_dir=lambda *a: cache,
                                  cache_dir=tmp_path / "hub")
    return mgr, snap, cache


def test_assess_rules():
    assert assess(files=["m.Q4_K_M.gguf"]).runtime == "llama.cpp"
    a = assess(files=["m.Q4_K_M.gguf"])
    assert a.adapter == "llama-cpp" and a.quantization == "Q4_K_M"
    a = assess(files=["config.json", "modeling_x.py", "pytorch_model.bin"], library_name="transformers", pipeline_tag="text-to-speech")
    assert a.requires_custom_code and not a.usable and any("pickle" in w for w in a.warnings)
    assert assess(files=["x.nemo"], pipeline_tag="automatic-speech-recognition").adapter == "parakeet"
    assert assess(files=["weights.xyz"]).runtime is None
    assert assess(files=["config.json"], repo_id="fishaudio/drama-3").adapter == "fish-server"


def test_known_models_are_honest():
    d3 = find_known("fishaudio/drama-3")
    assert d3.est_vram_mib >= 48 * 1024 and d3.est_ram_mib is None
    assert find_known("parakeet").repo_id == "nvidia/parakeet-tdt-0.6b-v3"
    assert find_known("moonshine-small").repo_id is None  # not a Hub download in its runtime's format
    for k in ("drama-3", "s2-pro", "parakeet"):
        assert find_known(k).facts, k


def test_search_maps_category_and_marks_known(tmp_path):
    mgr, _, _ = make(tmp_path)
    rows = mgr.search("qwen", category="llm", quantization="gguf", limit=5)
    assert mgr.api.calls[-1]["pipeline_tag"] == "text-generation"
    assert "gguf" in mgr.api.calls[-1]["filter"]
    assert rows[0].known == "qwen2.5-1.5b"


def test_info_reports_size_license_and_runtime(tmp_path):
    mgr, _, _ = make(tmp_path)
    d = mgr.info("Qwen/Qwen2.5-1.5B-Instruct-GGUF")
    assert d.total_bytes == sum(len(b) for b in FILES.values())
    assert d.license == "apache-2.0" and d.assessment.adapter == "llama-cpp"


def test_gated_and_missing(tmp_path):
    mgr, _, _ = make(tmp_path, gated=True)
    with pytest.raises(GatedModelError, match="hf auth login"):
        mgr.info("google/gemma-3-1b-it-qat-q4_0-gguf")
    mgr, _, _ = make(tmp_path, missing=True)
    with pytest.raises(HubError, match="not found"):
        mgr.info("nobody/nothing")


def test_plan_download_verify_register(tmp_path, monkeypatch):
    mgr, snap, _ = make(tmp_path)
    plan = mgr.plan("qwen2.5-1.5b")  # registry alias -> repo + allow_patterns
    assert plan.repo_id == "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
    assert [f.path for f in plan.files] == []  # *q4_k_m.gguf matches nothing in the fake repo
    plan = mgr.plan("Qwen/Qwen2.5-1.5B-Instruct-GGUF", allow_patterns=["*.gguf"])
    assert plan.download_bytes == 4096 and plan.fits_disk
    prog = []
    entry = mgr.download("Qwen/Qwen2.5-1.5B-Instruct-GGUF", allow_patterns=["*.gguf"], confirm=True, progress=prog.append)
    assert entry["commit"] == COMMIT and entry["verified"] is True and entry["adapter"] == "llama-cpp"
    if importlib.util.find_spec("tqdm"):  # progress rides on huggingface_hub's tqdm dependency
        assert prog and prog[-1]["done"] == 4096
    assert snap.calls[-1]["revision"] == COMMIT  # pinned by commit, not branch
    # corrupt -> verify fails
    p = Path(entry["path"]) / "model.gguf"
    p.write_bytes(b"X" * 4096)
    rep = mgr.verify("Qwen/Qwen2.5-1.5B-Instruct-GGUF")
    assert not rep.ok and "sha256 mismatch" in rep.problems[0]


def test_large_download_needs_confirmation(tmp_path, monkeypatch):
    from aivoice import settings as st

    st.set_value("hf.confirm_over_gib", 0.000001)
    mgr, _, _ = make(tmp_path)
    with pytest.raises(ConfirmationRequired):
        mgr.download("Qwen/Qwen2.5-1.5B-Instruct-GGUF", allow_patterns=["*.gguf"])


def test_retry_with_backoff_then_success(tmp_path):
    mgr, snap, _ = make(tmp_path, fail_times=2)
    sleeps = []
    e = mgr.download("Qwen/Qwen2.5-1.5B-Instruct-GGUF", allow_patterns=["*.gguf"], confirm=True, sleep=sleeps.append)
    assert e["verified"] and sleeps == [2, 4]


def test_retry_exhausted_keeps_partials_message(tmp_path):
    mgr, snap, _ = make(tmp_path, fail_times=9)
    with pytest.raises(HubError, match="partial files are kept"):
        mgr.download("Qwen/Qwen2.5-1.5B-Instruct-GGUF", allow_patterns=["*.gguf"], confirm=True, max_retries=2, sleep=lambda s: None)


def test_cancellation(tmp_path):
    mgr, _, _ = make(tmp_path)
    ev = threading.Event()
    ev.set()
    with pytest.raises(DownloadCancelled):
        mgr.download("Qwen/Qwen2.5-1.5B-Instruct-GGUF", allow_patterns=["*.gguf"], confirm=True, cancel=ev)
    assert mgr.get_entry("Qwen/Qwen2.5-1.5B-Instruct-GGUF") is None


def test_offline_uses_registry(tmp_path):
    mgr, _, _ = make(tmp_path)
    mgr.download("Qwen/Qwen2.5-1.5B-Instruct-GGUF", allow_patterns=["*.gguf"], confirm=True)
    off = HuggingFaceModelManager(api=None, snapshot_download=None, cache_dir=tmp_path / "hub", offline=True)
    d = off.info("Qwen/Qwen2.5-1.5B-Instruct-GGUF")
    assert d.commit == COMMIT and d.installed
    with pytest.raises(HubError):
        off.search("anything")


def test_remove_only_own_revision_and_shared_snapshot_kept(tmp_path):
    cache = CacheInfo([Repo("Qwen/Qwen2.5-1.5B-Instruct-GGUF", [Rev(COMMIT, 4096), Rev("b" * 40, 100)], 4196),
                       Repo("other/app-model", [Rev("c" * 40, 999)], 999)])
    mgr, _, _ = make(tmp_path, cache=cache)
    mgr.download("Qwen/Qwen2.5-1.5B-Instruct-GGUF", allow_patterns=["*.gguf"], confirm=True)
    # a second registry entry (alias) pointing at the same snapshot keeps the files
    reg = mgr.installed()
    reg["alias"] = dict(reg["Qwen/Qwen2.5-1.5B-Instruct-GGUF"])
    mgr._save_registry(reg)
    r = mgr.remove("Qwen/Qwen2.5-1.5B-Instruct-GGUF")
    assert r["kept_files_reason"] and cache.deleted == []
    r = mgr.remove("alias")
    assert cache.deleted == [COMMIT]  # never "b"*40 or the other app's repo


def test_cleanup_only_touches_managed_unpinned(tmp_path):
    cache = CacheInfo([Repo("Qwen/Qwen2.5-1.5B-Instruct-GGUF", [Rev(COMMIT, 4096), Rev("b" * 40, 100)], 4196),
                       Repo("other/app-model", [Rev("c" * 40, 999)], 999)])
    mgr, _, _ = make(tmp_path, cache=cache)
    mgr.download("Qwen/Qwen2.5-1.5B-Instruct-GGUF", allow_patterns=["*.gguf"], confirm=True)
    r = mgr.cleanup(dry_run=True)
    assert [c["commit"] for c in r["candidates"]] == ["b" * 40] and cache.deleted == []
    assert r["left_alone"] == ["other/app-model"]
    mgr.cleanup(dry_run=False)
    assert cache.deleted == ["b" * 40]


def test_import_path_detects_runtime_and_is_never_deleted(tmp_path):
    d = tmp_path / "mymodel"
    d.mkdir()
    (d / "tiny.Q8_0.gguf").write_bytes(b"x" * 10)
    mgr, _, cache = make(tmp_path)
    e = mgr.import_path(d)
    assert e["runtime"] == "llama.cpp" and e["adapter"] == "llama-cpp"
    r = mgr.remove(e["key"])
    assert (d / "tiny.Q8_0.gguf").exists() and "never deleted" in r["kept_files_reason"]


def test_native_moonshine_download_requires_confirmation(tmp_path):
    mgr, _, _ = make(tmp_path)
    with pytest.raises(HubError, match="--yes"):
        mgr.download("moonshine-small")


# --- lifecycle -------------------------------------------------------------


def _lm():
    lm = ModelLifecycleManager(ResourceScheduler(mock_rtx4090()))
    events = []

    def loader(entry, device):
        events.append(("load", device))
        return LoadedModel(entry["key"], object(), device, 0, unload=lambda: events.append("unload"), warmup=lambda: 5.0)

    lm.register_loader("llama-cpp", loader)
    return lm, events


ENTRY = {"key": "qwen2.5-1.5b", "repo_id": "Qwen/Qwen2.5-1.5B-Instruct-GGUF", "source": "huggingface",
         "verified": True, "adapter": "llama-cpp", "category": "llm"}


def test_lifecycle_states_and_memory():
    lm, events = _lm()
    m = lm.load("qwen", ENTRY)
    assert lm.state("qwen") == ModelState.READY and m.device == "cpu" and m.warmup_ms == 5.0
    assert lm.scheduler.reserved("cpu") == 1800
    lm.acquire("qwen")
    assert lm.state("qwen") == ModelState.IN_USE
    with pytest.raises(LifecycleError, match="in use"):
        lm.unload("qwen")
    lm.release("qwen")
    assert lm.unload("qwen") and lm.state("qwen") == ModelState.INSTALLED
    assert lm.scheduler.reserved("cpu") == 0 and "unload" in events


def test_lifecycle_rejects_unverified_unsupported_and_oversize():
    lm, _ = _lm()
    with pytest.raises(LifecycleError, match="not verified"):
        lm.load("x", {**ENTRY, "verified": False})
    with pytest.raises(LifecycleError, match="no aivoice adapter"):
        lm.load("y", {**ENTRY, "adapter": None})
    lm.register_loader("fish-server", lambda e, d: LoadedModel("d3", None, d, 0))
    with pytest.raises(InsufficientMemory):
        lm.load("drama", {"key": "drama-3", "source": "huggingface", "verified": True, "adapter": "fish-server", "category": "tts"})
    assert lm.state("drama") == ModelState.ERROR


def test_lifecycle_warmup_failure_releases_memory():
    lm = ModelLifecycleManager(ResourceScheduler(mock_rtx4090()))

    def bad_warm():
        raise RuntimeError("bad")

    lm.register_loader("llama-cpp", lambda e, d: LoadedModel("q", None, d, 0, warmup=bad_warm))
    with pytest.raises(LifecycleError, match="warmup failed"):
        lm.load("q", ENTRY)
    assert lm.state("q") == ModelState.ERROR and lm.scheduler.reserved("cpu") == 0


def test_lifecycle_concurrent_load_is_refused():
    lm = ModelLifecycleManager(ResourceScheduler(mock_rtx4090()))
    gate = threading.Event()
    started = threading.Event()

    def slow(entry, device):
        started.set()
        gate.wait(2)
        return LoadedModel("q", None, device, 0)

    lm.register_loader("llama-cpp", slow)
    fut = lm.load_async("q", ENTRY)
    started.wait(2)
    with pytest.raises(LifecycleError, match="in progress"):
        lm.load("q", ENTRY)
    gate.set()
    assert fut.result(2).device == "cpu"
