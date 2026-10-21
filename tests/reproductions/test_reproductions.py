"""Paper-reproduction framework: streaming exactness, causality, checkpoints, resume, provenance."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from aivoice.research_reproductions.common import harness
from aivoice.research_reproductions.common.streaming_nn import (
    CausalConv1d,
    CausalConvTranspose1d,
    Transformer,
    state_nbytes,
)

REFS = Path(os.environ.get("AIVOICE_TEST_REFS", "/nonexistent"))


def _chunks(x, sizes, dim=-1):
    out, i, k = [], 0, 0
    while i < x.shape[dim]:
        n = sizes[k % len(sizes)]
        out.append(x.narrow(dim, i, min(n, x.shape[dim] - i)))
        i, k = i + n, k + 1
    return out


@pytest.mark.parametrize("k,s,d", [(7, 1, 1), (3, 1, 4), (8, 4, 1), (16, 8, 1)])
def test_causal_conv_streaming_equals_full(k, s, d):
    torch.manual_seed(0)
    m = CausalConv1d(3, 4, k, s, d).eval()
    x = torch.randn(2, 3, 320)
    st, ys = None, []
    for c in _chunks(x, [1, 7, 13, 40, 3]):
        y, st = m.step(c, st)
        ys.append(y)
    assert torch.allclose(torch.cat(ys, -1), m(x), atol=1e-6)
    x2 = x.clone()
    x2[..., 200:] = 0  # strict causality: outputs before the change are identical
    assert torch.equal(m(x)[..., : 200 // s - (k - 1) * d // s - 1], m(x2)[..., : 200 // s - (k - 1) * d // s - 1])


@pytest.mark.parametrize("k,s", [(4, 2), (16, 8), (10, 5)])
def test_causal_conv_transpose_streaming(k, s):
    m = CausalConvTranspose1d(3, 2, k, s).eval()
    x = torch.randn(1, 3, 40)
    st, ys = None, []
    for c in _chunks(x, [1, 3, 7]):
        y, st = m.step(c, st)
        ys.append(y)
    assert torch.allclose(torch.cat(ys, -1), m(x), atol=1e-6)


@pytest.mark.parametrize("rc", [0, 2, [3, 0]])
def test_transformer_kv_cache_streaming_and_bounded(rc):
    torch.manual_seed(0)
    m = Transformer(16, 2, 4, context=8, right_context=rc).eval()
    x = torch.randn(1, 80, 16)
    st, ys, sizes = None, [], []
    for c in _chunks(x, [1, 4, 9, 2], dim=1):
        y, st = m.step(c, st)
        ys.append(y)
        sizes.append(state_nbytes(st))
    y = torch.cat(ys, 1)
    la = m.lookahead_frames
    assert y.shape[1] == 80 - la
    assert torch.allclose(y, m(x)[:, : y.shape[1]], atol=1e-5)
    assert max(sizes[5:]) == min(sizes[5:])  # KV cache bounded by the window


def test_slerp_is_spherical_not_linear():
    from aivoice.research_reproductions.tvtsyn.model.timbre import slerp

    a, b = torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]])
    for t in (0.0, 0.25, 0.5, 1.0):
        y = slerp(a, b, torch.tensor([[t]]))
        assert torch.allclose(y.norm(dim=-1), torch.ones(1), atol=1e-6)
        assert torch.allclose(torch.atan2(y[0, 1], y[0, 0]), torch.tensor(t * np.pi / 2), atol=1e-5)
    lerp = torch.nn.functional.normalize(0.75 * a + 0.25 * b, dim=-1)
    assert not torch.allclose(slerp(a, b, torch.tensor([[0.25]])), lerp, atol=1e-3)


def test_attention_pooling_permutation_invariant_and_masked():
    from aivoice.research_reproductions.common.blocks import AttentionPooling

    p = AttentionPooling(8, 4)
    h = torch.randn(1, 10, 8)
    perm = torch.randperm(10)
    assert torch.allclose(p(h), p(h[:, perm]), atol=1e-6)
    mask = torch.zeros(1, 10, dtype=torch.bool)
    mask[0, :4] = True
    assert torch.allclose(p(h, mask), p(h[:, :4]), atol=1e-6)


def test_factorized_vq_straight_through():
    from aivoice.research_reproductions.tvtsyn.model.encoder import FactorizedVQ

    vq = FactorizedVQ(16, 32, 4, 0.15, 1.0)
    z = torch.randn(2, 16, 10, requires_grad=True)
    out = vq(z)
    assert out["z_q"].shape == z.shape and int(out["indices"].max()) < 32
    (out["z_q"].sum() + out["commit"] + out["codebook"]).backward()
    assert z.grad is not None and float(z.grad.abs().sum()) > 0


def test_losses_basic():
    from aivoice.research_reproductions.common.losses import (
        MultiScaleSpectralLoss,
        lsgan_d_loss,
        lsgan_g_loss,
    )

    x = torch.randn(1, 4000)
    assert float(MultiScaleSpectralLoss((256, 64))(x, x)) < 1e-6
    assert float(lsgan_d_loss([torch.ones(3)], [torch.zeros(3)])) == 0.0
    assert float(lsgan_g_loss([torch.ones(3)])) == 0.0


def _meta(cfg):
    from aivoice.research_reproductions.common.checkpoint import CheckpointMeta

    return CheckpointMeta(engine="zerovc-repro", model_name="aivoice/test", weights_status="RETRAINED_PAPER_REPRODUCTION",
                          source_status="PAPER_REPRODUCTION", arch_confidence="MEDIUM", config=cfg, step=5)


def test_checkpoint_roundtrip_checksum_fingerprint_shapes(tmp_path):
    from aivoice.research_reproductions.common import checkpoint as ck

    m = torch.nn.Linear(3, 2)
    opt = torch.optim.AdamW(m.parameters())
    m(torch.randn(1, 3)).sum().backward()
    opt.step()
    p = ck.save_checkpoint(tmp_path / "a.pt", _meta({"w": 3}), {"m": m}, optimizers={"o": opt})
    blob = ck.load_checkpoint(p, engine="zerovc-repro", config={"w": 3})
    ck.load_model_state(torch.nn.Linear(3, 2), blob["models"]["m"])
    assert ck.read_meta(p)["param_shapes"]["m"]["weight"] == [2, 3]
    with pytest.raises(ck.CheckpointError, match="fingerprint"):
        ck.load_checkpoint(p, config={"w": 4})
    with pytest.raises(ck.CheckpointError, match="engine"):
        ck.load_checkpoint(p, engine="rtvc")
    with pytest.raises(ck.CheckpointError, match="does not match"):
        ck.load_model_state(torch.nn.Linear(4, 2), blob["models"]["m"])
    raw = bytearray(p.read_bytes())
    raw[-10] ^= 0xFF
    p.write_bytes(bytes(raw))
    with pytest.raises(ck.CheckpointError, match="checksum"):
        ck.load_checkpoint(p)


def test_random_init_and_standin_never_usable():
    from aivoice.research_reproductions.common.checkpoint import is_usable_for_inference

    assert not is_usable_for_inference({"weights_status": "RANDOM_INIT", "step": 0})[0]
    assert not is_usable_for_inference({"weights_status": "RETRAINED_PAPER_REPRODUCTION", "step": 9,
                                        "config": {"test_standin": True}})[0]
    assert is_usable_for_inference({"weights_status": "RETRAINED_PAPER_REPRODUCTION", "step": 9, "config": {}})[0]


class _ToyRecipe:
    """Minimal recipe for trainer tests (GAN-free, dropout-free)."""

    engine = "zerovc-repro"
    overfit_loss_key = "loss"

    def __init__(self, n=8):
        self.n = n

    def model_config(self):
        return {"toy": 1}

    def build(self, tc, device):
        from aivoice.research_reproductions.common.trainer import make_adamw

        torch.manual_seed(0)
        self.net = torch.nn.Sequential(torch.nn.Linear(4, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1))
        self.models = {"net": self.net}
        self.optimizers = {"main": make_adamw(self.net.parameters(), 1e-2)}
        self.schedulers = {"main": torch.optim.lr_scheduler.ExponentialLR(self.optimizers["main"], 0.99)}
        g = torch.Generator().manual_seed(1)
        xs = torch.randn(self.n, 4, generator=g)
        self.train_set = [{"x": xs[i], "y": xs[i].sum(-1, keepdim=True)} for i in range(self.n)]
        self.val_set = None

    def train_step(self, b, ctx):
        noise = torch.randn_like(b["y"]) * 0.01  # consumes RNG: resume must restore it
        loss = torch.nn.functional.mse_loss(self.net(b["x"]), b["y"] + noise)
        ctx.optimize("main", loss)
        return {"loss": float(loss.detach())}

    def evaluate(self, batches, ctx):
        return {}

    def checkpoint_meta(self, step):
        return _meta({"toy": 1})


def test_exact_resume(tmp_path):
    from aivoice.research_reproductions.common.trainer import TrainConfig, Trainer

    def cfg(out, steps, resume=None):
        return TrainConfig(output=str(out), device="cpu", precision="fp32", batch_size=2, workers=0, max_steps=steps,
                           checkpoint_every=3, eval_every=0, log_every=1, resume=resume, keep_checkpoints=10)

    a = _ToyRecipe()
    Trainer(a, cfg(tmp_path / "a", 7)).fit()
    b1 = _ToyRecipe()
    Trainer(b1, cfg(tmp_path / "b", 3)).fit()
    b2 = _ToyRecipe()
    Trainer(b2, cfg(tmp_path / "b", 7, resume="latest")).fit()
    for pa, pb in zip(a.net.parameters(), b2.net.parameters()):
        assert torch.equal(pa, pb)
    st = json.loads((tmp_path / "b" / "status.json").read_text())
    assert st["state"] == "finished" and st["step"] == 7


def test_feature_cache_key_mismatch(tmp_path):
    from aivoice.research_reproductions.common.features import (
        FeatureCache,
        FeatureCacheError,
        FeatureSpec,
    )

    spec = FeatureSpec("f", "1", "m", "rev1", 16000, 320, 4)
    c = FeatureCache(tmp_path, spec)
    c.put("u1", np.zeros((5, 4)))
    assert FeatureCache.open(c.dir).get("u1").shape == (5, 4)
    with pytest.raises(FeatureCacheError):
        c.put("u2", np.zeros((5, 3)))
    other = FeatureCache(tmp_path, FeatureSpec("f", "1", "m", "rev2", 16000, 320, 4))
    assert other.dir != c.dir  # a changed teacher revision never reuses old features


def test_manifest_and_resumable_sampler(tmp_path):
    import soundfile as sf

    from aivoice.research_reproductions.common.data import (
        Filters,
        ResumableSampler,
        build_manifest,
        read_manifest,
    )

    for spk in ("a", "b"):
        (tmp_path / "corpus" / spk).mkdir(parents=True)
        for i in range(2):
            sf.write(tmp_path / "corpus" / spk / f"{i}.wav", 0.1 * np.sin(np.arange(24000) / 10).astype(np.float32), 16000)
    sf.write(tmp_path / "corpus" / "a" / "short.wav", np.zeros(800, np.float32), 16000)
    st = build_manifest(tmp_path / "corpus", "custom", tmp_path / "m.jsonl", Filters(min_duration=1.0))
    assert st["kept"] == 4 and st["rejected"] == {"too_short": 1} and st["speakers"] == 2
    assert {r["speaker"] for r in read_manifest(tmp_path / "m.jsonl")} == {"custom/a", "custom/b"}
    s = ResumableSampler(10, seed=3)
    it = iter(s)
    first = [next(it) for _ in range(4)]
    s2 = ResumableSampler(10, seed=3)
    s2.load_state_dict({"epoch": 0, "pos": 4, "n": 10})
    rest = list(iter(s2))
    full = list(iter(ResumableSampler(10, seed=3)))
    assert first + rest == full


# --- per-reproduction streaming checks on tiny random networks (architecture tests, not models) ---------------

def _zerovc():
    from aivoice.research_reproductions.zerovc.config import tiny
    from aivoice.research_reproductions.zerovc.streaming.inference import (
        ZeroVCStreaming,
    )

    return ZeroVCStreaming.untrained(tiny()), torch.randn(1, 32)


def _rtvc():
    from aivoice.research_reproductions.rtvc.config import tiny
    from aivoice.research_reproductions.rtvc.streaming.inference import (
        RTVCStreaming,
        Target,
    )

    return RTVCStreaming.untrained(tiny()), Target(torch.randn(1, 16), 0.9)


def _tvtsyn():
    from aivoice.research_reproductions.tvtsyn.config import tiny
    from aivoice.research_reproductions.tvtsyn.streaming.inference import (
        TVTSynStreaming,
    )

    return TVTSynStreaming.untrained(tiny(), codebook_init_audio=harness.test_signal(16000, 3.0, 9)), torch.randn(1, 32)


@pytest.mark.parametrize("make", [_zerovc, _rtvc, _tvtsyn], ids=["zerovc", "rtvc", "tvtsyn"])
def test_streaming_causal_reset_chunk_invariant(make):
    model, tgt = make()
    c = harness.causality(model, tgt, seconds=6.0, split_s=5.0)
    assert c["passed"], c
    assert harness.state_reset(model, tgt, seconds=1.0)["passed"]
    ci = harness.chunk_invariance(model, tgt, seconds=1.5)
    assert ci["passed"], ci


def test_long_session_state_bounded():
    model, tgt = _zerovc()
    rep = harness.long_session(model, tgt, minutes=0.2, window_s=3.0, state_bytes=lambda s: s.state_bytes())
    assert rep["passed"], rep
    assert rep["state_bytes_first_last"][0] == rep["state_bytes_first_last"][1]


def test_paper_config_lookahead_is_honest():
    from aivoice.research_reproductions.tvtsyn.config import aivoice_variant, paper
    from aivoice.research_reproductions.tvtsyn.model.tvtsyn import TVTSyn
    from aivoice.research_reproductions.zerovc.config import paper as zpaper
    from aivoice.research_reproductions.zerovc.model.zerovc import ZeroVC

    assert ZeroVC(zpaper()).content.lookahead_frames == 0  # Zero-VC: zero future frames
    assert TVTSyn(paper()).lookahead_frames == 32           # official config: 4 per layer x 8 layers
    assert TVTSyn(aivoice_variant()).lookahead_frames == 4  # aivoice variant


def test_anonymizer_is_training_only():
    code = ("import sys; import aivoice.research_reproductions.zerovc.streaming.inference as m; "
            "print(any('anonymizer' in k for k in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_untrained_checkpoint_refused(tmp_path):
    from dataclasses import asdict

    from aivoice.research_reproductions.common.checkpoint import (
        CheckpointError,
        CheckpointMeta,
        save_checkpoint,
    )
    from aivoice.research_reproductions.zerovc.config import tiny
    from aivoice.research_reproductions.zerovc.model.zerovc import ZeroVC
    from aivoice.research_reproductions.zerovc.streaming.inference import (
        ZeroVCStreaming,
    )

    cfg = tiny()
    net = ZeroVC(cfg)
    p = save_checkpoint(tmp_path / "z.pt", CheckpointMeta("zerovc-repro", "x", "RANDOM_INIT", "s", "MEDIUM", asdict(cfg)),
                        {"content": net.content, "timbre": net.timbre, "decoder": net.decoder})
    with pytest.raises(CheckpointError):
        ZeroVCStreaming.from_checkpoint(p)


def test_registry_engines_experimental_and_need_checkpoint(monkeypatch, tmp_path):
    from aivoice.research_reproductions import registry
    from aivoice.vc.base import EngineState
    from aivoice.vc.engines import ReproVCEngine, get_vc_engine

    monkeypatch.setenv("AIVOICE_REPRO_HOME", str(tmp_path))
    assert set(registry.ENTRIES) >= {"rtvc", "zerovc-repro", "tvtsyn-repro", "streamvoiceanon-plus", "vossa-repro"}
    for k in ("rtvc", "zerovc-repro", "tvtsyn-repro", "tvtsyn-aivoice"):
        st = ReproVCEngine(k).status()
        assert st.state == EngineState.MISSING_WEIGHTS and not st.ready
        assert "NOT official" in st.caps.weights
    assert get_vc_engine("zerovc").name == "zerovc-repro"
    assert registry.PROVENANCE["vossa-repro"].value == "official"
    assert registry.PROVENANCE["tvtsyn-aivoice"].value == "aivoice_variant"


def test_provenance_labels_never_official_for_reproductions():
    from aivoice.research_reproductions import registry

    for k in registry.ENTRIES:
        info = registry.get(k).info
        if k != "vossa-repro":
            assert info.source_status.value != "OFFICIAL_CODE"
            assert info.model_name.startswith("aivoice/")


def test_cli_reproduce_commands():
    from click.testing import CliRunner

    from aivoice.cli import main

    r = CliRunner()
    assert r.invoke(main, ["reproduce", "list"]).exit_code == 0
    out = r.invoke(main, ["reproduce", "info", "zerovc-repro"])
    assert out.exit_code == 0 and "assumptions" in out.output and "NOT official" not in out.output
    assert "Zero-VC" in r.invoke(main, ["reproduce", "paper-map", "zerovc"]).output
    st = r.invoke(main, ["reproduce", "status", "--json"])
    assert st.exit_code == 0 and len(json.loads(st.output)) == 6


@pytest.mark.skipif(not (REFS / "RT-VC").exists(), reason="set AIVOICE_TEST_REFS to a dir with an RT-VC checkout")
def test_rtvc_official_parity():
    from aivoice.research_reproductions.rtvc.eval.official_parity import check

    assert check(REFS / "RT-VC")["passed"]


@pytest.mark.skipif(not (REFS / "VOSSA").exists(), reason="set AIVOICE_TEST_REFS to a dir with a VOSSA checkout")
def test_tvtsyn_official_parity():
    from aivoice.research_reproductions.tvtsyn.eval.official_parity import check

    assert check(REFS / "VOSSA")["passed"]


@pytest.mark.skipif(not (REFS / "StreamVoiceAnon").exists(), reason="needs a StreamVoiceAnon checkout")
def test_streamvoiceanon_plus_extension(monkeypatch):
    monkeypatch.setenv("AIVOICE_STREAMVOICEANON_REPO_DIR", str(REFS / "StreamVoiceAnon"))
    from aivoice.research_reproductions.streamvoiceanon_plus.config import tiny
    from aivoice.research_reproductions.streamvoiceanon_plus.model.extension import (
        StreamVoiceAnonPlus,
    )
    from aivoice.research_reproductions.streamvoiceanon_plus.model.official import (
        tiny_official,
    )

    c = tiny()
    d = c.test_dims
    m = StreamVoiceAnonPlus(tiny_official(d), c)
    B, P, T, K = 2, c.prompt_frames, c.source_frames, d["num_codebooks"]
    g = torch.Generator().manual_seed(0)
    b = {"sem_p": torch.randint(0, d["vocab"], (B, P), generator=g), "ac_p": torch.randint(0, d["codebook"], (B, K, P), generator=g),
         "sem_s": torch.randint(0, d["vocab"], (B, T), generator=g), "ac_s": torch.randint(0, d["codebook"], (B, K, T), generator=g),
         "src_lens": torch.tensor([T, T - 2]), "timbre": torch.randn(B, 32, 16), "style": torch.randn(B, 24),
         "emo": torch.randn(B, T, c.emo_dim)}
    out = m(b)
    assert set(out) == {"slow_ar", "fast_ar", "emo", "loss"}
    assert "emo" not in m(b, with_emotion=False)
    assert not any(k.startswith("emotion_head") for k in m.inference_state_dict())
