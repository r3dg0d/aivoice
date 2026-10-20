import json
import shutil
import wave
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from aivoice.cli import main
from aivoice.engines.base import EngineCapabilities, SpeakerRef, SynthesisOptions
from aivoice.engines.registry import EngineStatus
from aivoice.fusion.router import route
from aivoice.speakers import (
    ProfileError,
    create_profile,
    delete_profile,
    list_profiles,
    load_profile,
    migrate_dict,
    migrate_v1,
)

FAKES = Path(__file__).parent / "fakes" / "fakepkgs"
R = CliRunner()


@pytest.fixture(autouse=True)
def _energy_vad(monkeypatch):
    """Tone-based fixtures are not speech to Silero; pin the energy VAD so CI (no Silero) and dev match."""
    import aivoice.vad as vad

    monkeypatch.setattr(vad, "make_vad", lambda engine="auto", sample_rate=16000: vad.EnergyVAD(sample_rate=sample_rate))


def ref_wav(path: Path, seconds=5.0, lead=1.0, sr=16000):
    t = np.arange(int(sr * seconds)) / sr
    x = (0.3 * np.sin(2 * np.pi * 150 * t) * (1 + np.sin(2 * np.pi * 4 * t)) / 2).astype(np.float32)
    x = np.concatenate([np.zeros(int(lead * sr), np.float32), x, np.zeros(int(lead * sr), np.float32)])
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((x * 32767).astype("<i2").tobytes())
    return path


# --- profiles -----------------------------------------------------------------


def test_profile_requires_consent(tmp_path):
    with pytest.raises(ProfileError, match="--i-have-rights"):
        create_profile("Me", ref_wav(tmp_path / "r.wav"), i_have_rights=False)
    assert list_profiles() == []


def test_profile_create_prepares_and_records_provenance(tmp_path):
    p = create_profile("My Voice", ref_wav(tmp_path / "r.wav"), i_have_rights=True, transcript="hello there", publishable=False)
    assert p.profile_version == 2 and p.consent_recorded and len(p.audio_hash) == 64
    assert Path(p.prepared_audio).is_file() and Path(p.original_audio).is_file()
    assert "trim" in p.preparation["steps"]
    d = json.loads((p.dir() / "profile.json").read_text())
    assert d["profile_version"] == 2 and d["consent"]["attested"]
    assert load_profile("my voice").id == p.id
    ref = p.speaker_ref()
    assert ref.audio == Path(p.prepared_audio) and ref.transcript == "hello there"
    delete_profile(p.id)
    assert not p.dir().exists()


def test_no_retention_of_original(tmp_path):
    p = create_profile("Lean", ref_wav(tmp_path / "r.wav"), i_have_rights=True, keep_original=False)
    assert p.original_audio is None and not list(p.dir().glob("original*"))


def test_migration_from_v1(tmp_path):
    from aivoice.tts_voices import add_local_voice
    from aivoice.voices import create_from_reference

    v1 = create_from_reference("Old MeanVC Voice", ref_wav(tmp_path / "a.wav"))
    tv = add_local_voice("Old TTS Voice", ref_wav(tmp_path / "b.wav"), "words here", i_have_rights=True)
    done = migrate_v1()
    assert {s for s, _ in done} == {f"voices:{v1.id}", f"voice:{tv.id}"}
    assert migrate_v1() == []  # idempotent
    mv = load_profile("Old MeanVC Voice")
    assert mv.migrated_from["profile_version"] == 1 and not mv.consent_recorded  # must re-attest
    tvp = load_profile("Old TTS Voice")
    assert tvp.consent_recorded and tvp.transcript == "words here"
    # v1 data untouched
    from aivoice.voices import load_voice

    assert Path(load_voice(v1.id).reference).is_file()


def test_migrate_dict_pure():
    d = migrate_dict({"id": "x", "display_name": "X", "reference": "/a.wav", "meanvc2_profile_version": 1,
                      "checksum_reference": "abc", "extra": {"reference_kind": "synthesized"}})
    assert d["profile_version"] == 2 and d["reference_kind"] == "synthesized" and d["audio_hash"] == "abc"


# --- router -------------------------------------------------------------------


def st(name, ready=True, **caps):
    base = dict(tasks=("tts",), zero_shot=True, transcript_free_prompt=True, devices=("cuda", "cpu"))
    base.update(caps)
    return EngineStatus(name, EngineCapabilities(name=name, display_name=name, **base), None if ready else "missing")


def test_router_filters_and_unmeasured_fallback():
    sts = [st("voxcpm2", style_prompt=True, streaming=True), st("xvoice", cross_lingual=True),
           st("voicestar", requires_prompt_transcript=True, transcript_free_prompt=False), st("espeak", zero_shot=False),
           st("gone", ready=False)]
    spk = SpeakerRef("me", Path("r.wav"))
    d = route(priority="balanced", speaker=spk, options=SynthesisOptions(), statuses=sts, measured={}, free_vram_mib=0)
    assert d.engine == "voxcpm2" and not d.measured and "not a quality claim" in d.reasons[0]
    assert "transcript" in d.rejected["voicestar"] and "cannot clone" in d.rejected["espeak"] and "unavailable" in d.rejected["gone"]
    d = route(speaker=spk, options=SynthesisOptions(style_prompt="soft"), statuses=sts, measured={}, free_vram_mib=0)
    assert d.candidates == ["voxcpm2"]
    d = route(speaker=spk, options=SynthesisOptions(), statuses=sts, streaming=True, measured={}, free_vram_mib=0)
    assert d.candidates == ["voxcpm2"]


def test_router_uses_measurements():
    sts = [st("voxcpm2"), st("xvoice")]
    m = {"voxcpm2": {"speaker_similarity": 0.80, "wer": 0.05, "first_audio_ms": 400},
         "xvoice": {"speaker_similarity": 0.86, "wer": 0.04, "first_audio_ms": 900}}
    assert route(priority="similarity", statuses=sts, measured=m, free_vram_mib=0).engine == "xvoice"
    assert route(priority="latency", statuses=sts, measured=m, free_vram_mib=0).engine == "voxcpm2"
    d = route(priority="similarity", statuses=sts, measured={"xvoice": m["xvoice"]}, free_vram_mib=0)
    assert d.measured and "not measured yet" in d.reasons[1]


def test_router_memory_and_remote_consent():
    sts = [st("big", devices=("cuda",), est_vram_mib=48000), st("remote", remote=True, local=False)]
    d = route(statuses=sts, measured={}, free_vram_mib=24000)
    assert d.engine is None and "GPU memory" in d.rejected["big"] and "not allowed" in d.rejected["remote"]


# --- pipeline + CLI -------------------------------------------------------------


@pytest.fixture
def voxcpm_fake(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(FAKES))
    monkeypatch.setenv("PYTHONPATH", str(FAKES))
    snap = tmp_path / "VoxCPM2"
    snap.mkdir()
    from aivoice import settings

    settings.set_value("engines.voxcpm2.model_path", str(snap))
    return snap


def test_clone_cli_end_to_end_with_fake_engine(voxcpm_fake, tmp_path):
    ref = ref_wav(tmp_path / "ref.wav")
    r = R.invoke(main, ["clone", "create", "Tester", "--reference", str(ref)])
    assert r.exit_code != 0 and "i-have-rights" in r.output
    r = R.invoke(main, ["clone", "create", "Tester", "--reference", str(ref), "--i-have-rights"])
    assert r.exit_code == 0, r.output
    assert "Tester" in R.invoke(main, ["clone", "list"]).output
    out = tmp_path / "out.wav"
    r = R.invoke(main, ["clone", "synthesize", "--voice", "Tester", "--text", "Hello, I paid $5 today.",
                        "--style", "calm", "-o", str(out), "--consent-ack"])
    assert r.exit_code == 0, r.output
    side = json.loads(Path(str(out) + ".json").read_text())
    assert side["engine"] == "voxcpm2" and side["synthetic"] and "five dollars" in side["params"]["text"]
    from aivoice.evaluation.metrics import read_wav_comment

    assert "Synthetic speech" in read_wav_comment(out)
    r = R.invoke(main, ["clone", "synthesize", "--voice", "Tester", "--text", "x", "--engine", "xvoice",
                        "--style", "calm", "-o", str(tmp_path / "x.wav"), "--consent-ack"])
    assert r.exit_code != 0
    r = R.invoke(main, ["clone", "compare", "--voice", "Tester", "--engines", "voxcpm2,meanvc2", "--text", "hi there",
                        "--output-dir", str(tmp_path / "cmp"), "--consent-ack"])
    assert r.exit_code == 0, r.output
    summary = json.loads((tmp_path / "cmp" / "summary.json").read_text())
    rows = {x["engine"]: x for x in summary["rows"]}
    assert rows["voxcpm2"]["status"] == "ok" and rows["meanvc2"]["status"].startswith("unavailable")
    assert R.invoke(main, ["clone", "delete", "Tester", "--yes"]).exit_code == 0


def test_engines_and_research_cli():
    for args in (["engines", "list"], ["engines", "capabilities"], ["engines", "info", "voxcpm2"], ["research", "list"],
                 ["research", "citations"], ["benchmark", "--help"], ["benchmark", "research", "--help"], ["clone", "--help"]):
        r = R.invoke(main, args)
        assert r.exit_code == 0, (args, r.output)
    assert "@inproceedings{peng2026voicestar" in R.invoke(main, ["research", "citations"]).output
    r = R.invoke(main, ["benchmark", "--frames", "3", "--consent-ack"])
    assert r.exit_code == 0 and "rtf=" in r.output  # legacy behaviour preserved


@pytest.mark.skipif(not shutil.which("espeak-ng"), reason="espeak-ng not installed")
def test_benchmark_research_synthetic_records_unavailable(tmp_path):
    r = R.invoke(main, ["benchmark", "research", "--synthetic", "--engines", "voxcpm2,espeak", "--output", str(tmp_path / "b"),
                        "--seeds", "0", "--consent-ack"])
    assert r.exit_code == 0, r.output
    data = json.loads((tmp_path / "b" / "results.json").read_text())
    st_ = {row["engine"]: row["status"] for row in data["rows"]}
    assert st_["voxcpm2"].startswith("unavailable") and st_["espeak"] == "not a cloning engine"
    assert (tmp_path / "b" / "results.csv").is_file()


def test_adapt_dataset_validation(tmp_path):
    ds = tmp_path / "ds"
    ds.mkdir()
    for i in range(12):
        ref_wav(ds / f"c{i:02d}.wav", seconds=2.0 + i * 0.1, lead=0.1)
        (ds / f"c{i:02d}.txt").write_text("this is a clean training sentence")
    (ds / "c99.wav").write_bytes((ds / "c00.wav").read_bytes())  # duplicate
    (ds / "c99.txt").write_text("this is a clean training sentence")
    from aivoice.research.adaptation import validate_dataset, write_voxcpm_manifests

    rep = validate_dataset(ds, min_minutes=0.1)
    assert rep.duplicates and rep.usable == 12 and len(rep.val) == 1 and rep.ready
    paths = write_voxcpm_manifests(rep, tmp_path / "out", ref_audio="/ref.wav")
    row = json.loads(Path(paths["train"]).read_text().splitlines()[0])
    assert set(row) == {"audio", "text", "ref_audio"}
    p = create_profile("Adapt Me", ref_wav(tmp_path / "r.wav"), i_have_rights=True)
    r = R.invoke(main, ["clone", "adapt", "--voice", p.id, "--dataset", str(ds), "--output", str(tmp_path / "o2")])
    assert r.exit_code != 0 and "i-have-rights" in r.output
