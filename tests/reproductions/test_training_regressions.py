"""Regressions found by the end-to-end quick run of scripts/aivoice-runtime/train-reproductions.sh."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def test_rtvc_pitch_losses_run_under_bf16_autocast():
    from aivoice.research_reproductions.rtvc.model.losses import pitch_losses

    out = {"logits": torch.randn(2, 10, 8, device=DEV), "periodicity": torch.rand(2, 1, 10, device=DEV),
           "loudness": torch.randn(2, 1, 10, device=DEV)}
    with torch.autocast(DEV, dtype=torch.bfloat16):  # BCE refused to run under autocast on CUDA
        ls = pitch_losses(out, torch.rand(2, 10, 8, device=DEV), torch.rand(2, 1, 10, device=DEV),
                          torch.randn(2, 1, 10, device=DEV))
    assert all(torch.isfinite(v) for v in ls.values())


def test_rtvc_vocoder_forward_under_bf16_autocast():
    from aivoice.research_reproductions.rtvc.config import tiny
    from aivoice.research_reproductions.rtvc.model.decoder import DDSPVocoder

    cfg = tiny()
    voc = DDSPVocoder(cfg.vocoder, cfg.speaker.spkr_emb_dim, cfg.ema.channels).to(DEV)
    t = 6
    f0 = torch.full((1, 1, t), 120.0, device=DEV)
    with torch.autocast(DEV, dtype=torch.bfloat16):  # cuFFT has no bf16 kernels
        y = voc(f0, torch.randn(1, 1, t, device=DEV), torch.rand(1, 1, t, device=DEV),
                torch.randn(1, cfg.ema.channels, t, device=DEV), torch.randn(1, cfg.speaker.spkr_emb_dim, device=DEV))
    assert y.dtype == torch.float32 and torch.isfinite(y).all()


def test_teacher_dir_falls_back_to_installed_teachers(monkeypatch, tmp_path):
    from aivoice.research_reproductions.common.teacher_paths import TeacherUnavailable, teacher_dir

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    monkeypatch.delenv("AIVOICE_TEACHER_WAVLM_LARGE", raising=False)
    with pytest.raises(TeacherUnavailable):
        teacher_dir("wavlm-large")
    d = tmp_path / "aivoice" / "teachers" / "wavlm-large"
    d.mkdir(parents=True)
    (d / "config.json").write_text("{}")
    assert teacher_dir("wavlm-large") == d


def test_overfit_gate_rows_respect_duration_filter():
    from aivoice.cli_reproduce import _gate_rows

    rows = [{"id": "a", "split": "train", "duration": 1.5}, {"id": "b", "split": "train", "duration": 5.0},
            {"id": "c", "split": "dev", "duration": 6.0}, {"id": "d", "split": "train", "duration": 4.2}]
    assert [r["id"] for r in _gate_rows(rows, 4.0)] == ["b", "d"]
    assert [r["id"] for r in _gate_rows(rows[:1], 4.0)] == ["a"]  # nothing long enough: never empty


def test_tvtsyn_vq_outputs_raw_code_like_official():
    from aivoice.research_reproductions.tvtsyn.model.encoder import FactorizedVQ

    vq = FactorizedVQ(16, 32, 4, 0.15, 1.0)
    out = vq(torch.randn(2, 16, 10))
    raw = vq.out(vq.codebook(out["indices"]).transpose(1, 2))  # normalisation is only for the lookup
    assert torch.allclose(out["z_q"], raw, atol=1e-6)


def test_tvtsyn_content_transformer_has_no_output_norm():
    from aivoice.research_reproductions.tvtsyn.config import tiny
    from aivoice.research_reproductions.tvtsyn.model.encoder import ContentEncoder

    assert isinstance(ContentEncoder(tiny().content).tf.norm, torch.nn.Identity)


def test_streaming_report_runs_exactness_without_tf32_and_restores_flags():
    from aivoice.research_reproductions.common.evaluate import streaming_report
    from aivoice.research_reproductions.zerovc.config import tiny
    from aivoice.research_reproductions.zerovc.streaming.inference import ZeroVCStreaming

    before = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    rep = streaming_report(ZeroVCStreaming.untrained(tiny()), torch.randn(1, 32))
    assert rep["tf32"] is False and rep["passed"]
    assert (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) == before
