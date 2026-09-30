"""Reference synthesis (TTS -> RVC) for RVC packages that ship no demo audio."""

from __future__ import annotations

import json
import os
import sys
import wave
from pathlib import Path

import pytest

from aivoice.adapt import adapt_rvc_to_meanvc2
from aivoice.rvc.detect import RvcPackageInfo
from aivoice.rvc.synth import (
    CommandRvc,
    CommandTts,
    SynthError,
    render_command,
    rvc_version,
    synthesize_reference,
)
from aivoice.voices import load_voice


def _wav(path: Path, seconds: float = 1.0, value: int = 1000) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(value.to_bytes(2, "little", signed=True) * int(16000 * seconds))


class FakeTts:
    name = "fake-tts"

    def __init__(self):
        self.texts = []

    def unavailable_reason(self):
        return None

    def synthesize(self, text, out):
        self.texts.append(text)
        _wav(out, 1.0, 500)


class FakeRvc:
    name = "fake-rvc"

    def __init__(self):
        self.calls = []

    def unavailable_reason(self):
        return None

    def convert(self, src, checkpoint, index, out, version):
        assert src.is_file(), "the TTS output must exist when RVC runs"
        self.calls.append((checkpoint.name, index.name if index else None, version))
        _wav(out, 1.0, 2000)


class Missing:
    def __init__(self, name, why):
        self.name, self._why = name, why

    def unavailable_reason(self):
        return self._why


def _package(tmp_path: Path, *, index: bool = True, generation: str = "RVC v2") -> RvcPackageInfo:
    root = tmp_path / "pkg"
    root.mkdir()
    (root / "voice.pth").write_bytes(b"not a real checkpoint")
    if index:
        (root / "voice.index").write_bytes(b"idx")
    return RvcPackageInfo(
        generation=generation,
        checkpoint=str(root / "voice.pth"),
        index=str(root / "voice.index") if index else None,
        root=str(root),
    )


def test_render_command_keeps_values_as_single_arguments():
    argv = render_command("tool --in {input} --out {output}", {"input": "/tmp/a b; rm -rf x.wav", "output": "/o.wav"})
    assert argv == ["tool", "--in", "/tmp/a b; rm -rf x.wav", "--out", "/o.wav"]


def test_pipeline_runs_tts_then_rvc_and_cleans_up(tmp_path):
    info = _package(tmp_path)
    tts, rvc = FakeTts(), FakeRvc()
    res = synthesize_reference(info, tmp_path / "out", text="hello there", tts_engines=[tts], rvc_engines=[rvc])
    assert res.path.is_file() and res.path.name == "synthesized_reference.wav"
    assert tts.texts == ["hello there"]
    assert rvc.calls == [("voice.pth", "voice.index", "v2")]
    assert not (tmp_path / "out" / "_tts_base.wav").exists()
    assert (res.tts_engine, res.rvc_engine) == ("fake-tts", "fake-rvc")


def test_v1_packages_are_run_as_v1_and_a_missing_index_is_noted(tmp_path):
    info = _package(tmp_path, index=False, generation="RVC v1")
    assert rvc_version(info) == "v1"
    rvc = FakeRvc()
    res = synthesize_reference(info, tmp_path / "out", tts_engines=[FakeTts()], rvc_engines=[rvc])
    assert rvc.calls == [("voice.pth", None, "v1")]
    assert any("no .index" in n for n in res.notes)


def test_falls_through_to_the_first_available_engine_and_explains_when_none(tmp_path):
    info = _package(tmp_path)
    ok = FakeTts()
    res = synthesize_reference(info, tmp_path / "o1", tts_engines=[Missing("a", "nope"), ok], rvc_engines=[FakeRvc()])
    assert res.tts_engine == "fake-tts"
    with pytest.raises(SynthError) as e:
        synthesize_reference(info, tmp_path / "o2", tts_engines=[Missing("espeak-ng", "espeak-ng is not installed")], rvc_engines=[FakeRvc()])
    assert "espeak-ng is not installed" in str(e.value)


def test_no_checkpoint_is_an_error(tmp_path):
    with pytest.raises(SynthError):
        synthesize_reference(RvcPackageInfo(generation="RVC v2"), tmp_path, tts_engines=[FakeTts()], rvc_engines=[FakeRvc()])


def test_command_engines_run_real_subprocesses_without_a_shell(tmp_path):
    """End to end through the command templates, using this interpreter as the 'engine'."""
    tts_script = tmp_path / "tts.py"
    tts_script.write_text(
        "import sys,wave\n"
        "out=sys.argv[2]\n"
        "w=wave.open(out,'wb'); w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)\n"
        "w.writeframes(b'\\x10\\x00'*16000); w.close()\n"
    )
    rvc_script = tmp_path / "rvc.py"
    rvc_script.write_text(
        "import sys,shutil,json\n"
        "src,model,index,out,version=sys.argv[1:6]\n"
        "shutil.copy(src,out)\n"
        "open(out+'.args','w').write(json.dumps([model.split('/')[-1],index.split('/')[-1],version]))\n"
    )
    info = _package(tmp_path)
    tts = CommandTts(template=f"{sys.executable} {tts_script} {{text_file}} {{out}}")
    rvc = CommandRvc(template=f"{sys.executable} {rvc_script} {{input}} {{model}} {{index}} {{output}} {{version}}")
    res = synthesize_reference(info, tmp_path / "out", tts_engines=[tts], rvc_engines=[rvc])
    assert res.path.stat().st_size > 10_000
    assert json.loads((res.path.parent / "synthesized_reference.wav.args").read_text()) == ["voice.pth", "voice.index", "v2"]


def test_a_failing_engine_surfaces_its_error(tmp_path):
    info = _package(tmp_path)
    bad = CommandRvc(template=f"{sys.executable} -c 'import sys; sys.stderr.write(\"boom\"); sys.exit(3)'")
    with pytest.raises(SynthError) as e:
        synthesize_reference(info, tmp_path / "out", tts_engines=[FakeTts()], rvc_engines=[bad])
    assert "exit 3" in str(e.value) and "boom" in str(e.value)


# --- adapt integration -----------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.delenv("AIVOICE_NO_SYNTH", raising=False)


def _rvc_dir(tmp_path: Path) -> Path:
    d = tmp_path / "rvc_pkg"
    d.mkdir()
    (d / "model_v2.pth").write_bytes(b"x" * 64)
    return d


def _patch_engines(monkeypatch):
    from aivoice.rvc import synth

    monkeypatch.setattr(synth, "default_tts_engines", lambda: [FakeTts()])
    monkeypatch.setattr(synth, "default_rvc_engines", lambda: [FakeRvc()])


def test_import_without_reference_audio_synthesizes_one_and_labels_it(tmp_path, monkeypatch):
    _patch_engines(monkeypatch)
    result = adapt_rvc_to_meanvc2(_rvc_dir(tmp_path), display_name="Synth Voice")
    assert result.voice is not None and result.voice.status == "ready" and result.engine == "meanvc2"
    assert result.synthesized
    voice = load_voice("Synth Voice")
    assert voice.extra["reference_kind"] == "synthesized"
    assert voice.extra["adaptation"] == "rvc-synthesized-reference"
    assert voice.extra["synthesis"]["tts"] == "fake-tts"
    assert "synthesized" in voice.notes
    assert Path(voice.reference).is_file()


def test_a_real_recording_wins_over_synthesis(tmp_path, monkeypatch):
    _patch_engines(monkeypatch)
    d = _rvc_dir(tmp_path)
    _wav(d / "demo_preview.wav", 2.0)
    result = adapt_rvc_to_meanvc2(d, display_name="Real Voice")
    assert not result.synthesized
    assert load_voice("Real Voice").extra["reference_kind"] == "recording"


def test_opt_out_flag_and_env_keep_the_honest_incomplete_path(tmp_path, monkeypatch):
    _patch_engines(monkeypatch)
    r1 = adapt_rvc_to_meanvc2(_rvc_dir(tmp_path), display_name="No Synth", synthesize=False)
    assert r1.voice.status == "incomplete" and r1.engine == "rvc" and "disabled" in r1.message
    monkeypatch.setenv("AIVOICE_NO_SYNTH", "1")
    d2 = tmp_path / "second"
    d2.mkdir()
    (d2 / "m_v2.pth").write_bytes(b"y" * 64)
    r2 = adapt_rvc_to_meanvc2(d2, display_name="Env Off")
    assert r2.voice.status == "incomplete"


def test_missing_engines_fall_back_to_incomplete_and_say_why(tmp_path, monkeypatch):
    from aivoice.rvc import synth

    monkeypatch.setattr(synth, "default_tts_engines", lambda: [Missing("espeak-ng", "espeak-ng is not installed")])
    monkeypatch.setattr(synth, "default_rvc_engines", lambda: [FakeRvc()])
    r = adapt_rvc_to_meanvc2(_rvc_dir(tmp_path), display_name="Needs Tools")
    assert r.voice.status == "incomplete"
    assert "espeak-ng is not installed" in r.message


def test_importing_again_upgrades_an_incomplete_voice_instead_of_colliding(tmp_path, monkeypatch):
    d = _rvc_dir(tmp_path)
    first = adapt_rvc_to_meanvc2(d, display_name="Upgrade Me", synthesize=False)
    assert first.voice.status == "incomplete"
    _patch_engines(monkeypatch)
    second = adapt_rvc_to_meanvc2(d, display_name="Upgrade Me")
    assert second.voice.status == "ready" and second.synthesized
    assert os.path.isfile(load_voice("Upgrade Me").reference)


def test_files_aivoice_generated_are_never_mistaken_for_a_recording(tmp_path):
    from aivoice.rvc.detect import inspect_rvc_package

    d = _rvc_dir(tmp_path)
    _wav(d / "_aivoice_placeholder.wav", 0.1, 0)
    (d / ".aivoice_synth").mkdir()
    _wav(d / ".aivoice_synth" / "synthesized_reference.wav", 2.0)
    info = inspect_rvc_package(d)
    assert info.reference_audio == []
    assert not info.has_usable_reference
