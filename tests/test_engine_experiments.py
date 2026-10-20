"""GPU-benchmark plumbing (experiments D/E/H, benchmark audio + report, evaluation ASR) against the fake engines."""

from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

from aivoice import settings
from aivoice.evaluation import engine_experiments as ee
from aivoice.evaluation.benchmark import markdown_report, run_research_benchmark
from aivoice.evaluation.metrics import available_asr, read_wav_comment

FAKES = Path(__file__).parent / "fakes" / "fakepkgs"


def _wav(path: Path, seconds: float, f0: float, sr: int = 16000) -> Path:
    t = np.arange(int(sr * seconds)) / sr
    env = (np.sin(2 * np.pi * 3 * t) > -0.3).astype(np.float32)
    x = 0.3 * np.sin(2 * np.pi * f0 * t) * env
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((x * 32767).astype("<i2").tobytes())
    return path


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "ds"
    for name, f0 in (("alice", 210.0), ("bob", 120.0)):
        d = root / name
        d.mkdir(parents=True)
        _wav(d / "reference.wav", 6.0, f0)
        (d / "reference.txt").write_text("the birch canoe slid on the smooth planks\n")
        _wav(d / "heldout.wav", 3.0, f0 * 1.02)
        (d / "heldout.txt").write_text("glue the sheet to the dark blue background\n")
        (d / "consent.json").write_text(json.dumps({"synthesis_authorized": True, "publication_authorized": False}))
    (root / "texts.txt").write_text("Please call Stella.\nThe meeting moved to Thursday.\n")
    return root


@pytest.fixture
def fake_engines(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(FAKES))
    monkeypatch.setenv("PYTHONPATH", str(FAKES))
    monkeypatch.setenv("AIVOICE_FAKE_LOG", str(tmp_path / "fake.log"))
    monkeypatch.setenv("AIVOICE_EVAL_ASR", "none")
    snap = tmp_path / "VoxCPM2"
    snap.mkdir()
    for e in ("voxcpm2", "voicestar"):
        settings.set_value(f"engines.{e}.python", sys.executable)
    settings.set_value("engines.voxcpm2.model_path", str(snap))
    return snap


def test_eval_asr_can_be_disabled(monkeypatch):
    monkeypatch.setenv("AIVOICE_EVAL_ASR", "none")
    assert available_asr() is None


def test_benchmark_saves_labelled_audio_and_report(dataset, fake_engines, tmp_path):
    out = tmp_path / "bench"
    res = run_research_benchmark(["voxcpm2"], dataset, out, seeds=(0,), save_audio=True)
    ok = [r for r in res["rows"] if r.get("status") == "ok" and r["engine"] == "voxcpm2"]
    assert len(ok) == 4  # 2 speakers x 2 texts x 1 seed
    f = out / ok[0]["audio_file"]
    assert f.is_file() and "Synthetic speech" in (read_wav_comment(f) or "")
    gt = [r for r in res["rows"] if r["engine"] == "ground-truth"]
    assert len(gt) == 2 and all(r["task"] == "real speech" and "speaker_similarity" in r for r in gt)
    summary = next(r for r in res["rows"] if r.get("status") == "summary")
    assert summary["samples"] == 4 and "load_ms" in summary and "vram_engine_mib" in summary
    report = (out / "report.md").read_text()
    assert "| voxcpm2 | 4 |" in report and "Real-speech ceiling" in report


def test_experiment_d_from_results(dataset, fake_engines, tmp_path):
    out = tmp_path / "bench"
    run_research_benchmark(["voxcpm2", "voicestar"], dataset, out, seeds=(0,))
    r = ee.exp_engines(out / "results.json")
    assert r.status == "measured"
    assert set(r.metrics["ranking_by_similarity_among_wer_le_0.2"]) == {"voxcpm2", "voicestar"}
    assert "WER was not measured" in r.decision
    assert ee.exp_engines(tmp_path / "missing.json").status == "not run"


def test_experiment_e_style_runs_with_instructions(dataset, fake_engines, tmp_path):
    r = ee.exp_style(dataset, audio_out=tmp_path / "audio", max_texts=1)
    assert r.status == "measured"
    assert set(ee.STYLES) <= set(r.metrics)
    gens = [json.loads(x) for x in (tmp_path / "fake.log").read_text().splitlines() if '"generate"' in x]
    assert any("speaking slowly and calmly" in g["text"] for g in gens)
    assert any("speaking quickly and excitedly" in g["text"] for g in gens)
    assert (tmp_path / "audio" / "E" / "alice" / "neutral.wav").is_file()


def test_experiment_h_pairs_fusion_and_raw(dataset, fake_engines, tmp_path):
    r = ee.exp_fusion(dataset, ["voxcpm2", "voicestar"], audio_out=tmp_path / "audio", max_texts=1)
    assert r.status == "measured", r.decision
    assert "all/clean" in r.metrics and "all/degraded" in r.metrics
    from aivoice.research.speaker_identity import get_encoder

    key = "sim" if get_encoder() is not None else "rate_dev"
    assert r.metrics["all/degraded"][key]["n"] == 4  # 2 engines x 2 speakers x 1 text
    assert (tmp_path / "audio" / "H" / "voicestar" / "bob" / "degraded-fusion.wav").is_file()
    assert "clean:" in r.decision and "degraded:" in r.decision


def test_experiments_not_run_without_engines(dataset, tmp_path, monkeypatch):
    monkeypatch.setenv("AIVOICE_VOXCPM2_PYTHON", "/nonexistent/python")
    assert ee.exp_style(dataset).status == "not run"
    assert ee.exp_fusion(dataset, ["voxcpm2"]).status == "not run"


def test_degrade_and_pitch_spread():
    sr = 16000
    x = (0.3 * np.sin(2 * np.pi * 150 * np.arange(sr * 2) / sr)).astype(np.float32)
    y = ee.degrade(x, sr)
    assert y.size == x.size + 3 * sr and np.abs(y).max() <= 1
    p = ee.pitch_spread_semitones(x, sr)
    assert p is None or p < 0.5  # a steady tone has (almost) no pitch movement


def test_markdown_report_lists_failures():
    meta = {"created": "now", "device": "GPU", "python": "3.11", "similarity_encoder": "GE2E", "asr": None,
            "seeds": [0], "prepare_reference": True}
    rows = [{"engine": "xvoice", "status": "load failed: boom"},
            {"engine": "voxcpm2", "status": "summary", "samples": 2, "speaker_similarity": 0.8, "rtf": 0.3}]
    md = markdown_report(meta, rows)
    assert "| voxcpm2 | 2 | 0.800 |" in md and "**xvoice**: load failed: boom" in md
