import pytest

from aivoice import settings as st
from aivoice.credentials import redact
from aivoice.hardware import fingerprint, mock_rtx4090, parse_nvidia_smi
from aivoice.resources import InsufficientMemory, Requirement, ResourceScheduler, is_oom_error


def test_defaults_and_persistence():
    assert st.get("experimental.enabled") is False
    st.set_value("asr.backend", "moonshine-medium")
    assert st.load()["asr"]["backend"] == "moonshine-medium"
    assert st.load()["asr"]["policy"] == "balanced"  # defaults still merged in
    assert st.settings_path().is_file()


def test_refuses_secrets():
    with pytest.raises(st.SettingsError):
        st.set_value("tts.api_key", "sk-123")
    with pytest.raises(st.SettingsError):
        st.save({**st.load(), "hf": {"token": "hf_abcdefghijklmnop"}})
    assert "api_key" not in st.settings_path().read_text() if st.settings_path().exists() else True


def test_remote_consent_roundtrip():
    assert not st.has_remote_consent("fish-api")
    st.grant_remote_consent("fish-api")
    assert st.has_remote_consent("fish-api")
    st.grant_remote_consent("fish-api", False)
    assert not st.has_remote_consent("fish-api")
    assert st.is_local_url("http://127.0.0.1:8080") and not st.is_local_url("http://gpu-box:8080")


def test_parse_value():
    assert st.parse_value("true") is True
    assert st.parse_value("12") == 12
    assert st.parse_value("null") is None
    assert st.parse_value("hello") == "hello"


def test_redact():
    msg = redact("Authorization: Bearer abcdef123456 failed key=abcdef123456 hf_ABCDEFGHIJKLMNOP", "abcdef123456")
    assert "abcdef123456" not in msg and "hf_ABCDEFGHIJ" not in msg


def test_parse_nvidia_smi():
    gpus = parse_nvidia_smi("0, NVIDIA GeForce RTX 4090, 24564, 23000, 560.35, 8.9\n")
    assert gpus[0].name.endswith("4090") and gpus[0].total_mib == 24564 and gpus[0].compute_capability == "8.9"


def test_fingerprint_changes_with_runtime():
    hw = mock_rtx4090()
    a = fingerprint(hw)
    hw.runtimes = {**hw.runtimes, "torch": "9.9"}
    assert fingerprint(hw) != a


def test_plan_on_rtx4090_rejects_drama3_and_places_s2pro():
    sched = ResourceScheduler(mock_rtx4090())
    plan = sched.plan(
        [
            Requirement("drama-3", 48 * 1024, None, priority=5),
            Requirement("s2-pro", 12 * 1024, None, priority=3),
            Requirement("moonshine-small", None, 700, prefer="cpu", priority=1),
            Requirement("qwen", 1800, 1800, prefer="cpu"),
        ]
    )
    assert "drama-3" in plan.rejected
    assert plan.device["s2-pro"] == "cuda:0"
    assert plan.device["moonshine-small"] == "cpu"
    assert plan.device["qwen"] == "cpu"


def test_no_gpu_rejects_gpu_only():
    hw = mock_rtx4090()
    hw.gpus = []
    plan = ResourceScheduler(hw).plan([Requirement("s2-pro", 12000, None)])
    assert "requires an NVIDIA GPU" in plan.rejected["s2-pro"]


def test_reserve_evicts_inactive_lru_and_oom_recovery():
    sched = ResourceScheduler(mock_rtx4090(), vram_headroom_mib=0)
    budget = sched.budget("cuda:0")
    unloaded = []
    sched.reserve("a", "cuda:0", budget - 1000, unload=lambda: unloaded.append("a"))
    sched.reserve("b", "cuda:0", 1000, unload=lambda: unloaded.append("b"))
    sched.touch("b", active=True)
    sched.reserve("c", "cuda:0", 5000)  # evicts inactive a, never active b
    assert unloaded == ["a"]
    with pytest.raises(InsufficientMemory):
        sched.reserve("d", "cuda:0", budget * 2)

    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return "ok"

    sched.touch("b", active=False)
    assert sched.run_with_oom_recovery("cuda:0", flaky) == "ok"
    assert "b" in unloaded
    assert is_oom_error(RuntimeError("CUDA out of memory"))
