import json
import os
import shutil
import subprocess

import pytest
from click.testing import CliRunner

from aivoice import settings as st
from aivoice import virtualmic as vm
from aivoice.cli import main


class FakePactl:
    """In-memory pactl: tracks loaded modules, sinks and sources."""

    def __init__(self):
        self.modules = {}
        self.next = 10
        self.calls = []

    def __call__(self, *args, timeout=5):
        self.calls.append(args)
        out = ""
        if args[:2] == ("list", "short"):
            kind = args[2]
            rows = []
            for mid, (name, kw) in self.modules.items():
                if kind == "modules":
                    rows.append(f"{mid}\t{name}\t")
                elif kind == "sinks" and name == "module-null-sink":
                    rows.append(f"{mid}\t{kw['sink_name']}\tmodule-null-sink.c")
                elif kind == "sources" and name == "module-remap-source":
                    rows.append(f"{mid}\t{kw['source_name']}\tmodule-remap-source.c")
            out = "\n".join(rows)
        elif args[0] == "load-module":
            kw = dict(a.split("=", 1) for a in args[2:] if "=" in a)
            mid = self.next
            self.next += 1
            self.modules[mid] = (args[1], kw)
            out = str(mid)
        elif args[0] == "unload-module":
            self.modules.pop(int(args[1]), None)
        return subprocess.CompletedProcess(args, 0, out, "")


@pytest.fixture
def pactl(monkeypatch):
    fake = FakePactl()
    monkeypatch.setattr(vm, "_pactl", fake)
    monkeypatch.setattr(vm.shutil, "which", lambda n: "/usr/bin/" + n if n in ("pactl",) else None)
    return fake


def test_virtual_mic_idempotent_and_cleanup_only_own(pactl):
    pactl.modules[1] = ("module-null-sink", {"sink_name": "someone_elses_sink"})
    a = vm.ensure_virtual_mic("aivoice")
    assert not a.reused and a.sink == "aivoice_sink" and a.source == "aivoice_mic"
    b = vm.ensure_virtual_mic("aivoice")
    assert b.reused and len([m for m in pactl.modules.values() if m[0] == "module-null-sink"]) == 2
    mid = vm.enable_monitor("aivoice")
    assert pactl.modules[mid][0] == "module-loopback"
    assert vm.disable_monitor("aivoice")
    done = vm.cleanup_virtual_mic("aivoice")
    assert sorted(done) == sorted(a.module_ids)
    assert list(pactl.modules) == [1]  # the other app's sink is untouched


def test_feedback_risk():
    assert vm.feedback_risk("aivoice_mic")
    assert vm.feedback_risk("aivoice_sink.monitor")
    assert vm.feedback_risk("alsa_input.pci-0000_00_1f.3.analog-stereo") is None


@pytest.mark.audio
@pytest.mark.skipif(os.environ.get("AIVOICE_TEST_REAL_PULSE") != "1" or not shutil.which("pactl"),
                    reason="set AIVOICE_TEST_REAL_PULSE=1 with a running PipeWire/Pulse server")
def test_real_pulse_virtual_mic():
    a = vm.ensure_virtual_mic("aivoicetest")
    try:
        assert vm.ensure_virtual_mic("aivoicetest").reused
        assert vm.status("aivoicetest")["source_present"]
    finally:
        vm.cleanup_virtual_mic("aivoicetest")
    assert not vm.status("aivoicetest")["sink_present"]


# --- CLI ----------------------------------------------------------------------

R = CliRunner()


def ok(args, **kw):
    r = R.invoke(main, args, catch_exceptions=False, **kw)
    assert r.exit_code == 0, (args, r.output)
    return r.output


def test_all_new_helps():
    for args in (
        ["experimental", "--help"], ["experimental", "start", "--help"], ["experimental", "asr", "--help"],
        ["experimental", "tts", "--help"], ["experimental", "preset", "--help"], ["experimental", "remote", "--help"],
        ["llm", "--help"], ["voice", "--help"], ["credentials", "--help"], ["hardware", "--help"],
        ["models", "search", "--help"], ["models", "download", "--help"], ["models", "cleanup", "--help"],
    ):
        ok(args)


def test_enable_disable_and_start_gate():
    r = R.invoke(main, ["experimental", "start", "--consent-ack", "--output", "null"])
    assert r.exit_code != 0 and "experimental enable" in r.output
    ok(["experimental", "enable"])
    assert st.get("experimental.enabled") is True
    ok(["experimental", "disable"])
    assert st.get("experimental.enabled") is False


def test_config_refuses_secrets_and_sets_values():
    r = R.invoke(main, ["experimental", "config", "set", "tts.api_key", "abc"])
    assert r.exit_code != 0 and "keyring" in r.output
    ok(["experimental", "config", "set", "vad.min_silence_ms", "250"])
    assert st.get("vad.min_silence_ms") == 250


def test_presets_and_remote_consent():
    out = ok(["experimental", "preset", "list"])
    assert "ultra-low-latency" in out and "balanced" in out
    ok(["experimental", "preset", "use", "ultra-low-latency"])
    assert st.get("asr.backend") == "moonshine-small" and st.get("audio.prebuffer_ms") == 40
    ok(["experimental", "config", "set", "audio.prebuffer_ms", "55"])
    ok(["experimental", "preset", "save", "mine"])
    ok(["experimental", "preset", "use", "balanced"])
    ok(["experimental", "preset", "use", "mine"])
    assert st.get("audio.prebuffer_ms") == 55
    ok(["experimental", "remote", "allow", "fish-api", "--yes"])
    assert "fish-api" in ok(["experimental", "remote", "list"])
    ok(["experimental", "remote", "revoke", "fish-api"])
    assert not st.has_remote_consent("fish-api")


def test_asr_use_and_list():
    ok(["experimental", "asr", "use", "parakeet"])
    assert st.get("asr.backend") == "parakeet" and st.get("asr.policy") == "manual"
    assert "moonshine-medium" in ok(["experimental", "asr", "list"])
    assert R.invoke(main, ["experimental", "asr", "use", "whisper"]).exit_code != 0


def test_llm_commands():
    ok(["llm", "enable", "--model", "qwen2.5-1.5b"])
    assert st.get("llm.enabled") and st.get("llm.mode") == "hybrid"
    ok(["llm", "style", "character"])
    ok(["llm", "character", "a grumpy pirate"])
    ok(["llm", "intensity", "0.8", "--timeout-ms", "300"])
    assert st.get("llm.intensity") == 0.8 and st.get("llm.timeout_ms") == 300
    out = ok(["llm", "test", "um hello there"])
    assert "rule-based" in out or "not found" in out
    ok(["llm", "disable"])


def test_voice_cli_requires_rights(tmp_path):
    import wave

    import numpy as np

    f = tmp_path / "r.wav"
    with wave.open(str(f), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((np.zeros(16000 * 4)).astype("<i2").tobytes())
    r = R.invoke(main, ["voice", "add", "Me", "--audio", str(f), "--text", "hello"])
    assert r.exit_code != 0 and "i-have-rights" in r.output
    ok(["voice", "add", "Me", "--audio", str(f), "--text", "hello", "--i-have-rights"])
    ok(["voice", "use", "Me"])
    assert st.get("voice.active") == "me"
    assert "local_voice" in ok(["voice", "list"])
    out = ok(["voice", "add-hosted", "Hosted", "--reference-id", "802e3bc2b27e49c2995d23ef70e6ac89", "--i-have-rights"])
    assert "cannot be used by a local Drama 3" in out
    ok(["voice", "remove", "Me", "--yes"])
    assert st.get("voice.active") is None


def test_models_known_and_list_and_plan():
    out = ok(["models", "known"])
    assert "fishaudio/drama-3" in out and "nvidia/parakeet-tdt-0.6b-v3" in out
    assert "models known" in ok(["models", "list"])
    out = ok(["experimental", "plan", "--mock-4090"])
    assert "drama-3 (reference)" in out and "!!" in out  # does not fit 24 GB
    assert "(none)" in ok(["models", "installed"])


def test_status_without_engine():
    out = ok(["experimental", "status"])
    assert "engine: not running" in out
    data = json.loads(ok(["experimental", "status", "--json"]))
    assert data["engine"] is None


@pytest.mark.skipif(not shutil.which("espeak-ng"), reason="espeak-ng not installed")
def test_start_with_file_input_real_local_tts(tmp_path):
    """Full CLI run: WAV as microphone, scripted ASR, real espeak-ng TTS, null output."""
    wav = tmp_path / "in.wav"
    subprocess.run(["espeak-ng", "-w", str(wav), "Testing the experimental mode from a file."], check=True)
    ok(["experimental", "enable"])
    ok(["experimental", "tts", "use", "espeak"])
    ok(["experimental", "config", "set", "asr.scripts", '["Testing the experimental mode from a file."]'])
    out = ok(["experimental", "start", "--input", str(wav), "--output", "null", "--asr", "scripted", "--consent-ack"])
    assert "local:espeak-ng" in out and "utterances 1" in out


def test_benchmark_mock_pipeline_writes_reports(tmp_path):
    if not shutil.which("espeak-ng"):
        pytest.skip("espeak-ng needed to generate clips")
    out = ok(["experimental", "benchmark", "--mock", "--no-scenarios", "--consent-ack"])
    assert "measures aivoice overhead only" in out and "reports:" in out
