"""Standalone streaming VC worker: runs inside an engine's own Python environment.

aivoice starts ``<engine python> -u stream_worker.py <handler>`` and exchanges JSON lines over
stdin/stdout. Like engines/worker_main.py this file imports nothing from aivoice, so X-VC,
MeanVC2, RVC, StyleStream and StreamVoiceAnon can each live in a venv with their own pinned
torch/transformers. The model is loaded once (``load``) and stays resident; the stream state
(caches, crossfade tails, ring buffers) lives here between ``chunk`` requests.

Requests:  {"id": n, "op": "load"|"prepare"|"start"|"chunk"|"flush"|"stats"|"ping"|"shutdown", ...}
           chunk: {"pcm": "<base64 float32 LE>"}
Responses: {"id": n, "ok": true, ...}; chunk/flush responses carry {"pcm": "<base64>", "ms": compute ms}

Upstream checkpoints that are pickles are loaded with ``torch.load(weights_only=True)`` unless the
user explicitly set ``trust_pickle`` for that engine (aivoice prints the risk before doing so).
"""

from __future__ import annotations

import base64
import contextlib
import json
import os
import sys
import time
import traceback

import numpy as np

OUT = sys.stdout
sys.stdout = sys.stderr  # libraries that print must not corrupt the protocol stream


def send(obj):
    OUT.write(json.dumps(obj) + "\n")
    OUT.flush()


def b64(x) -> str:
    return base64.b64encode(np.ascontiguousarray(np.asarray(x, dtype="<f4")).tobytes()).decode()


def unb64(s: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(s), dtype="<f4").copy()


@contextlib.contextmanager
def pushd_path(repo: str):
    """Upstream research repos import from their root and open relative paths."""
    old = os.getcwd()
    os.chdir(repo)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    try:
        yield
    finally:
        os.chdir(old)


@contextlib.contextmanager
def safe_torch_load(trust_pickle: bool):
    """Force weights_only=True on every torch.load inside the block unless the user trusted pickles."""
    import torch

    orig = torch.load

    def load(*a, **k):
        k["weights_only"] = not trust_pickle
        try:
            return orig(*a, **k)
        except Exception as e:  # noqa: BLE001
            if not trust_pickle and ("weights_only" in str(e).lower() or "Unsupported global" in str(e)):
                raise RuntimeError(
                    f"checkpoint {a[0] if a else ''} is not a plain-tensor file and would execute pickle code "
                    "to load. Re-run with engines.<name>.trust_pickle=true only if you trust its source."
                ) from e
            raise

    torch.load = load
    try:
        yield
    finally:
        torch.load = orig


def cuda_mem():
    try:
        import torch

        if torch.cuda.is_available():
            return {"vram_allocated_mib": round(torch.cuda.memory_allocated() / 2**20, 1),
                    "vram_peak_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
                    "gpu": torch.cuda.get_device_name(0)}
    except Exception:  # noqa: BLE001
        pass
    return {}


class SlidingWindowStreamer:
    """X-VC's chunkwise inference (bins/infer_utils.py run_streaming), made incremental.

    Upstream converts a whole file by recomputing a ``chunk_ms`` window per ``current_ms`` step:
    window = [i*cur - history, i*cur + cur + smooth + future), output = window[history : history+cur],
    with a raised-cosine crossfade of ``smooth_ms`` against the previous window's tail. Here the
    same windows are cut from a ring buffer as microphone audio arrives, so the output is
    sample-identical to upstream's offline-simulated streaming for the same input.
    """

    def __init__(self, forward, sr: int, chunk_ms: int, current_ms: int, future_ms: int, smooth_ms: int):
        if current_ms <= 0:
            raise ValueError("current_ms must be > 0")
        self.forward = forward
        self.sr = sr
        self.history_ms = chunk_ms - current_ms - smooth_ms - future_ms
        if self.history_ms < 0:
            raise ValueError("chunk_ms - current_ms - smooth_ms - future_ms must be >= 0")
        ms = lambda v: v * sr // 1000  # noqa: E731  (same integer arithmetic as upstream)
        self.hist, self.cur, self.smooth, self.fut = ms(self.history_ms), ms(current_ms), ms(smooth_ms), ms(future_ms)
        self.win = self.hist + self.cur + self.smooth + self.fut
        n = self.smooth
        if n > 0:
            fade_in = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, n)))
            self.fade_in, self.fade_out = fade_in.astype(np.float32), (1 - fade_in).astype(np.float32)
        self.reset()

    def reset(self):
        self.buf = np.zeros(self.hist, np.float32)  # upstream left-pads the first windows with zeros
        self.tail = np.zeros(self.smooth, np.float32)
        self.first = True
        self.consumed = 0

    @property
    def latency_samples(self) -> int:
        return self.cur + self.smooth + self.fut

    def push(self, x: np.ndarray) -> np.ndarray:
        self.buf = np.concatenate([self.buf, np.asarray(x, np.float32)])
        outs = []
        while self.buf.size >= self.win:
            outs.append(self._step(self.buf[: self.win]))
            self.buf = self.buf[self.cur :]
        return np.concatenate(outs) if outs else np.zeros(0, np.float32)

    def flush(self) -> np.ndarray:
        """Zero-pad the right edge like upstream does at the end of a file."""
        outs = []
        pending = self.buf.size - self.hist
        while pending > 0:
            w = self.buf[: self.win]
            w = np.pad(w, (0, self.win - w.size))
            outs.append(self._step(w)[: max(0, min(self.cur, pending))])
            self.buf = self.buf[self.cur :]
            pending -= self.cur
        self.reset()
        return np.concatenate(outs) if outs else np.zeros(0, np.float32)

    def _step(self, window: np.ndarray) -> np.ndarray:
        y = np.asarray(self.forward(window), np.float32).reshape(-1)
        seg = y[self.hist : self.hist + self.cur].copy()
        if self.smooth > 0:
            if not self.first:
                seg[: self.smooth] = self.tail * self.fade_out + seg[: self.smooth] * self.fade_in
            t0 = self.hist + self.cur
            tail = y[t0 : t0 + self.smooth]
            self.tail = np.pad(tail, (0, self.smooth - tail.size)).astype(np.float32)
        self.first = False
        return seg


# =============================================================================
# handlers
# =============================================================================


class EchoHandler:
    """Protocol self-test: returns input unchanged after a fixed delay. No model."""

    sample_rate = 16000

    def load(self, req):
        self.delay = int(req.get("delay_samples", 0))
        self.gain = float(req.get("gain", 1.0))
        self.crash_after = req.get("crash_after")
        self.sleep_ms = float(req.get("sleep_ms", 0.0))
        self.n = 0
        return {"sample_rate": self.sample_rate}

    def prepare(self, req):
        return {}

    def start(self, req):
        self.q = np.zeros(self.delay, np.float32)
        return {}

    def chunk(self, x):
        self.n += 1
        if self.crash_after is not None and self.n > int(self.crash_after):
            os._exit(3)  # simulate a hard worker crash (segfault / OOM kill)
        if self.sleep_ms:
            time.sleep(self.sleep_ms / 1000)
        self.q = np.concatenate([self.q, x * self.gain])
        out, self.q = self.q[: x.size], self.q[x.size :]
        return out

    def flush(self):
        out, self.q = self.q, np.zeros(0, np.float32)
        return out


class XVCHandler:
    """X-VC (Jerrister/X-VC @ 49df8c5) - load path of models/codec/sac/utils.inference_factory,
    conditions of bins/infer_utils.precompute_conditions, chunk forward of run_stream_chunk_forward."""

    def load(self, req):
        import torch
        from omegaconf import OmegaConf

        self.repo = req["repo_dir"]
        self.device = torch.device("cuda" if req.get("device", "cuda") == "cuda" and torch.cuda.is_available() else "cpu")
        with pushd_path(self.repo):
            from models.codec.sac.model import XVC
            from utils.file import load_config

            cfg = OmegaConf.create(load_config(req.get("config") or "configs/xvc.yaml"))
            if "config" in cfg:
                cfg = cfg["config"]
            gen = cfg["model"]["generator"]
            for node in (gen["semantic_encoder"]["encoder"]["from_pretrained"], gen["semantic_encoder"]["cfg"]):
                node["local_ckpt"] = req["glm_tokenizer_dir"]
            gen["speaker_encoder"]["pretrained_dir"] = req["eres2net_dir"]
            patched = os.path.join(req["run_dir"], "xvc-aivoice.yaml")
            OmegaConf.save(cfg, patched)
            self.cfg = OmegaConf.to_container(cfg, resolve=True)
            with safe_torch_load(bool(req.get("trust_pickle"))):
                model = XVC.load_from_checkpoint(patched, req["ckpt"], device=self.device, ema_load=bool(req.get("ema", False)))
            if hasattr(model, "remove_weight_norm"):
                model.remove_weight_norm()
            self.model = model.eval().to(self.device)
            if req.get("dtype") == "bf16" and self.device.type == "cuda":
                self.model = self.model.to(torch.bfloat16)
        self.sr = int(self.cfg["sample_rate"])
        self.hop = int(req.get("latent_hop_length", 1280))
        self.hp = float(self.cfg.get("highpass_cutoff_freq", 0.0))
        return {"sample_rate": self.sr, **cuda_mem()}

    def prepare(self, req):
        import torch

        with pushd_path(self.repo):
            from bins.infer_utils import precompute_conditions
            from models.codec.sac.utils import process_audio

            tgt = process_audio(req["reference"], self.cfg, self.hop)
        t = torch.from_numpy(tgt).unsqueeze(0).unsqueeze(1).float().to(self.device)
        cond = t
        if req.get("mask_target_condition"):
            cond = torch.cat([t, torch.zeros((1, 1, int(2.4 * self.sr)), device=self.device)], dim=-1)
        self.spk, self.frame = precompute_conditions(self.model, t, cond)
        return {"reference_s": round(tgt.size / self.sr, 2)}

    def start(self, req):
        import torch
        import torchaudio.functional as AF

        with pushd_path(self.repo):
            from bins.infer_utils import run_stream_chunk_forward

        def forward(window):
            w = torch.from_numpy(window).to(self.device)
            if self.hp:
                w = AF.highpass_biquad(w, self.sr, cutoff_freq=self.hp)
            w = w.view(1, 1, -1)
            with torch.inference_mode():
                y = run_stream_chunk_forward(self.model, w, self.spk, self.frame)
            return y.float().squeeze().cpu().numpy()

        self.stream = SlidingWindowStreamer(forward, self.sr, int(req.get("chunk_ms", 2400)), int(req.get("current_ms", 120)),
                                            int(req.get("future_ms", 100)), int(req.get("smooth_ms", 20)))
        return {"latency_ms": 1000 * self.stream.latency_samples / self.sr}

    def chunk(self, x):
        return self.stream.push(x)

    def flush(self):
        return self.stream.flush()


class MeanVC2Handler:
    """MeanVC2 (ASLP-lab/MeanVC2 runtime/run_rt.py VCRunner): resident models, process_chunk() streaming."""

    sample_rate = 16000

    def load(self, req):
        import torch

        vendor = req["vendor_dir"]
        rt = os.path.join(vendor, "runtime")
        for p in (rt, vendor):
            if p not in sys.path:
                sys.path.insert(0, p)
        os.chdir(vendor)
        from run_rt import VCRunner  # type: ignore

        dev = "cuda" if req.get("device", "cuda") == "cuda" and torch.cuda.is_available() else "cpu"
        self.runner = VCRunner(target_wav=req["reference"], device=dev, model=req.get("model", "40ms"))
        self.device = dev
        # VCRunner pins torch to 1 CPU thread; its JIT ASR encoder runs on the CPU (the TorchScript export
        # cannot move to CUDA). 4 threads: 38.5 -> 32.8 ms per 160 ms chunk on an i9-14900K + RTX 4090,
        # output identical within 2e-4.
        torch.set_num_threads(max(1, int(req.get("cpu_threads", 4))))
        return {"sample_rate": 16000, **cuda_mem()}

    def prepare(self, req):
        import torch
        from src.speaker import extract_embedding  # type: ignore  (runtime/src via run_rt's path)

        r = self.runner
        r.vc_spk_emb = extract_embedding(r.spk_model, req["reference"], device=self.device)
        with torch.no_grad():
            r.vc_gtm_kv = r.vc.gtm(r.vc_spk_emb)
        return {}

    def start(self, req):
        import torch

        torch.manual_seed(int(req.get("seed", 42)))
        self.runner._init_cache()
        self.pending = np.zeros(0, np.float32)
        return {}

    def chunk(self, x):
        # VCRunner._encode_chunk interpolates whatever BN frames one call produced to exactly 16 mel
        # frames, i.e. it assumes every call carries CHUNK=2560 samples (160 ms). Feeding the 10 ms live
        # blocks directly time-stretches the content (measured: GE2E target similarity 0.876 -> 0.825).
        # So re-chunk to exactly CHUNK samples, as upstream's own realtime loop (blocksize=CHUNK) does.
        self.pending = np.concatenate([getattr(self, "pending", np.zeros(0, np.float32)), np.asarray(x, np.float32)])
        n = self.runner.CHUNK
        outs = []
        while self.pending.size >= n:
            y = self.runner.process_chunk(self.pending[:n])
            self.pending = self.pending[n:]
            if y is not None:
                outs.append(y)
        return np.concatenate(outs).astype(np.float32) if outs else np.zeros(0, np.float32)

    def flush(self):
        """Feed the partial last chunk (no zero padding, like process_file), then drain the BN buffer and the vocoder overlap exactly like upstream VCRunner.process_file's tail."""
        import torch

        r = self.runner
        parts = []
        rest = getattr(self, "pending", np.zeros(0, np.float32))
        self.pending = np.zeros(0, np.float32)
        if rest.size:
            y = r.process_chunk(rest)
            if y is not None:
                parts.append(y)
        while r.bn_buffer is not None and r.bn_buffer.shape[1] >= r.chunk_size:
            pad_cur = r.bn_buffer[:, :r.chunk_size, :]
            if r.bn_buffer.shape[1] > r.chunk_size:
                pad_fut = r.bn_buffer[:, r.chunk_size:r._min_bn_len, :]
            else:
                pad_fut = r.bn_buffer[:, -1:, :].repeat(1, r.block_size, 1)
            cond = torch.cat([pad_cur, pad_fut], dim=1)
            r.bn_buffer = r.bn_buffer[:, r.chunk_size:, :]
            if r.bn_buffer.shape[1] == 0:
                r.bn_buffer = None
            parts.append(r._decode_mel(r._vc_step(cond)))
        if r.bn_buffer is not None and r.bn_buffer.shape[1] > 0:
            n_rem = r.bn_buffer.shape[1]
            last = r.bn_buffer[:, -1:, :]
            pad_cur = last.repeat(1, r.chunk_size, 1)
            pad_cur[:, :n_rem, :] = r.bn_buffer
            cond = torch.cat([pad_cur, last.repeat(1, r.block_size, 1)], dim=1)
            r.bn_buffer = None
            parts.append(r._decode_mel(r._vc_step(cond)))
        if r.last_wav is not None:
            parts.append(r.last_wav)
            r.last_wav = None
        return np.concatenate(parts).astype(np.float32) if parts else np.zeros(0, np.float32)


class RVCHandler:
    """RVC v1/v2 via aivoice's rvc/rt_worker.py RvcRealtime (rvc-python inside its own venv)."""

    sample_rate = 48000

    def load(self, req):
        import importlib.util

        spec = importlib.util.spec_from_file_location("aivoice_rvc_rt", req["rt_worker"])
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        self.mod = mod
        self.req = req
        self.rt = mod.RvcRealtime(model=req["model"], index=req.get("index") or "", device=req.get("device", "cuda"),
                                  pitch=float(req.get("pitch", 0.0)), index_rate=float(req.get("index_rate", 0.5)),
                                  protect=float(req.get("protect", 0.33)), block=float(req.get("block", 0.25)),
                                  extra=float(req.get("extra", 0.6)), crossfade=float(req.get("crossfade", 0.05)),
                                  threshold=float(req.get("threshold", -50.0)), sr=48000, log=False,
                                  lookahead=float(req.get("lookahead", 0.0)))
        self.rt.warmup()
        self.pending = np.zeros(0, np.float32)
        return {"sample_rate": 48000, "block": self.rt.block, **cuda_mem()}

    def prepare(self, req):
        return {}

    def start(self, req):
        if "pitch" in req:
            self.rt.pitch = float(req["pitch"])
        if req.get("index_rate") is not None:
            self.rt.index_rate = float(req["index_rate"])
        self.rt.reset()
        self.pending = np.zeros(0, np.float32)
        return {}

    def chunk(self, x):
        self.pending = np.concatenate([self.pending, x])
        outs = []
        b = self.rt.block
        while self.pending.size >= b:
            outs.append(self.rt.process(self.pending[:b]))
            self.pending = self.pending[b:]
        return np.concatenate(outs) if outs else np.zeros(0, np.float32)

    def flush(self):
        if self.pending.size:
            out = self.rt.process(np.pad(self.pending, (0, self.rt.block - self.pending.size)))[: self.pending.size]
            self.pending = np.zeros(0, np.float32)
            return out
        return np.zeros(0, np.float32)


class TVTSynHandler:
    """Official TVTSyn (tvtsyn_inference): resident model, one StreamingSession per stream, 16 kHz in/out."""

    sample_rate = 16000

    def load(self, req):
        import torch
        from tvtsyn_inference import TVTSyn

        dev = "cuda" if req.get("device", "cuda") == "cuda" and torch.cuda.is_available() else "cpu"
        # local bundle (pinned, SHA-256 verified at install); no network at run time
        self.model = TVTSyn.from_pretrained(req["bundle"], backend="pytorch", device=dev, cache_dir=req.get("speaker_cache"),
                                            local_files_only=True)
        self.speaker = None
        self.session = None
        return {"sample_rate": 16000, "chunk_samples": int(self.model.runtime.chunk_samples),
                "lookahead_frames": int(self.model.runtime.lookahead_frames), **cuda_mem()}

    def prepare(self, req):
        ref = req.get("reference")
        if not ref:
            raise ValueError("TVTSyn needs a target reference wav")
        cache = ref + ".tvtsyn-spk.npy"
        try:
            if os.path.getmtime(cache) >= os.path.getmtime(ref):
                self.speaker = np.load(cache)
                return {"speaker": "cached"}
        except OSError:
            pass
        self.speaker = self.model.encode_speaker(ref)
        try:
            np.save(cache, self.speaker)
        except OSError:
            pass
        return {"speaker": "encoded"}

    def start(self, req):
        if self.speaker is None:
            raise ValueError("prepare() a reference before start()")
        if self.session is not None:
            self.session.close()
        self.session = self.model.stream(self.speaker, sample_rate=16000)
        return {}

    def chunk(self, x):
        return self.session.push(np.asarray(x, np.float32))

    def flush(self):
        y = self.session.flush()
        self.session = self.model.stream(self.speaker, sample_rate=16000)
        return y


class StyleStreamHandler:
    """StyleStream (Berkeley-Speech-Group/StyleStream @ 38dd18c) - inference/streaming.py processing loop."""

    sample_rate = 16000

    def load(self, req):
        import torch

        self.repo = req["repo_dir"]
        with pushd_path(self.repo):
            from stylizer.cfm_lightning_module import CFMLightningModule
            from stylizer.modules import StreamingCrossfader

            dev = "cuda" if req.get("device", "cuda") == "cuda" and torch.cuda.is_available() else "cpu"
            self.model = CFMLightningModule.load_for_inference(os.path.join(self.repo, "assets/ckpts/stylizer-no-style-enc.ckpt"), device=dev)
            self.model.eval()
            self.Crossfader = StreamingCrossfader
        self.device = dev
        self.chunk_size, self.overlap, self.lookahead = 9600, 640, 80  # upstream constants (600 ms chunks)
        self.total = self.chunk_size + self.overlap + self.lookahead
        self.steps = int(req.get("steps", 10))
        self.cfg_strength = float(req.get("cfg_strength", 2.0))
        return {"sample_rate": 16000, **cuda_mem()}

    def prepare(self, req):
        style_dir = req.get("style_dir") or os.path.dirname(req["reference"])
        with pushd_path(self.repo):
            self.model.clear_cache()
            self.model.prepare_streaming(style_dir, self.chunk_size)
        return {"style_dir": style_dir}

    def start(self, req):
        self.buf = np.zeros(0, np.float32)
        self.xf = self.Crossfader(overlap=self.overlap, device=self.device)
        return {"latency_ms": 1000 * self.total / 16000}

    def chunk(self, x):
        self.buf = np.concatenate([self.buf, x])
        outs = []
        while self.buf.size >= self.total:
            window = self.buf[: self.total].copy()
            self.buf = self.buf[self.chunk_size :]
            y = self.model.realtime_sample(window, steps=self.steps, cfg_strength=self.cfg_strength)
            blk = self.xf.push(y)
            if blk.numel():
                outs.append(blk.detach().float().cpu().numpy().reshape(-1))
        return np.concatenate(outs) if outs else np.zeros(0, np.float32)

    def flush(self):
        t = self.xf.flush()
        return t.detach().float().cpu().numpy().reshape(-1) if t.numel() else np.zeros(0, np.float32)


class StreamVoiceAnonHandler:
    """StreamVoiceAnon (Plachtaa/StreamVoiceAnon @ 2017051) - evaluations/infer_arvc.py stream API. CUDA only."""

    def load(self, req):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("StreamVoiceAnon's streaming path uses CUDA events/autocast: a CUDA GPU is required")
        self.repo = req["repo_dir"]
        with pushd_path(self.repo):
            sys.path.insert(0, os.path.join(self.repo, "evaluations"))
            from infer_arvc import InferenceWrapper  # type: ignore

            with safe_torch_load(bool(req.get("trust_pickle"))):
                self.w = InferenceWrapper(req.get("config", "configs/config_firefly_arvcasr_8192_delay0_8.yaml"),
                                          req.get("checkpoint", "pretrained_checkpoints/dual_ar_delay_0_8.pth"),
                                          compile_ar=bool(req.get("compile", False)))
        self.sample_rate = int(self.w.sr)
        self.frames = int(req.get("decode_chunk_frames", 1))
        self.step = self.w.SAMPLES_PER_FRAME * self.frames
        return {"sample_rate": self.sample_rate, **cuda_mem()}

    def prepare(self, req):
        import librosa
        import torch

        ref, _ = librosa.load(req["reference"], sr=self.sample_rate)
        t = torch.from_numpy(ref).unsqueeze(0).to(self.w.device)
        with pushd_path(self.repo):
            self.w.prefill_prompt([t], max_prompt_frames=256, delay=int(req.get("delay", 2)), alpha=float(req.get("alpha", 1.0)))
        return {}

    def start(self, req):
        self.w.setup_stream_caches(encode_window_frames=128, decode_window_frames=64, max_seq_frames=768,
                                   buffer_frames=32, decode_chunk_frames=self.frames)
        self.buf = np.zeros(0, np.float32)
        return {}

    def chunk(self, x):
        import torch

        self.buf = np.concatenate([self.buf, x])
        outs = []
        while self.buf.size >= self.step:
            c = torch.from_numpy(self.buf[: self.step]).to(self.w.device)[None]
            self.buf = self.buf[self.step :]
            outs.append(self.w.process_one_chunk(c).float().squeeze().cpu().numpy().reshape(-1))
        return np.concatenate(outs) if outs else np.zeros(0, np.float32)

    def flush(self):
        return np.zeros(0, np.float32)


class VossaHandler:
    """VOSSA (Morris88826/VOSSA @ 5dd1a5c, official code + official checkpoints, CC BY-NC 4.0): src.streaming API."""

    def load(self, req):
        repo = req["repo_dir"]
        sys.path.insert(0, repo)
        with pushd_path(repo):
            from src.pretrained import load_vossa  # type: ignore
            from src.streaming import VossaStreamer  # type: ignore

            # Lightning checkpoints are pickles: allowed only with the same explicit opt-in as other engines
            with safe_torch_load(bool(req.get("trust_pickle"))):
                model = load_vossa(local_dir=req["checkpoint_dir"], device=req.get("device"))
        la = req.get("lookahead")
        self.streamer = VossaStreamer(model, chunk_frames=int(req.get("chunk_frames", 3)),
                                      lookahead=None if la is None else int(la),
                                      use_cuda_graph=bool(req.get("cuda_graph", True)))
        self.sample_rate = int(self.streamer.sample_rate)
        self.session = None
        return {"sample_rate": self.sample_rate, "lookahead_frames": self.streamer.step.lookahead,
                "chunk_frames": int(req.get("chunk_frames", 3)), **cuda_mem()}

    def prepare(self, req):
        import librosa

        ref, _ = librosa.load(req["reference"], sr=self.sample_rate)
        self.target = ref
        return {}

    def start(self, req):
        self.session = self.streamer.session(self.target)
        return {}

    def chunk(self, x):
        return np.asarray(self.session.push(x), np.float32).reshape(-1)

    def flush(self):
        return np.asarray(self.session.flush(), np.float32).reshape(-1) if self.session else np.zeros(0, np.float32)


class ReproHandler:
    """aivoice paper reproductions (research_reproductions): trained checkpoints only."""

    def load(self, req):
        from aivoice.research_reproductions import registry

        self.engine = registry.get(req["engine"]).load_streaming(req["checkpoint"], device=req.get("device", "cpu"),
                                                                 teacher_dirs=req.get("teacher_dirs") or {})
        self.sample_rate = int(self.engine.sample_rate)
        self.key = req["engine"]
        return {"sample_rate": self.sample_rate, "lookahead_samples": int(self.engine.lookahead_samples),
                "weights": self.engine.meta.get("weights_label"), "model_name": self.engine.meta.get("model_name")}

    def prepare(self, req):
        import librosa

        ref, _ = librosa.load(req["reference"], sr=self.sample_rate)
        e = self.engine
        if self.key == "rtvc":
            own = req.get("source_reference")
            src = librosa.load(own, sr=self.sample_rate)[0] if own else None
            self.target = e.enroll(ref, source_16k=src)
        elif self.key.startswith("tvtsyn"):
            raise RuntimeError("TVTSyn enrollment needs x-vector + ECAPA teachers in the worker; pass "
                               "precomputed embeddings with --target-embedding (not implemented in the live path yet)")
        else:
            self.target = e.enroll(ref, req.get("cache_dir"))
        return {}

    def start(self, req):
        self.session = self.engine.new_session(self.target)
        return {}

    def chunk(self, x):
        return np.asarray(self.session.push(x), np.float32)

    def flush(self):
        return np.zeros(0, np.float32)


HANDLERS = {"echo": EchoHandler, "xvc": XVCHandler, "meanvc2": MeanVC2Handler, "rvc": RVCHandler, "tvtsyn": TVTSynHandler,
            "stylestream": StyleStreamHandler, "streamvoiceanon": StreamVoiceAnonHandler, "vossa": VossaHandler,
            "repro": ReproHandler}


def main(argv):
    name = argv[1] if len(argv) > 1 else ""
    if name not in HANDLERS:
        send({"id": 0, "ok": False, "error": f"unknown handler {name!r}"})
        return 2
    h = HANDLERS[name]()
    send({"id": 0, "ok": True, "ready": True, "pid": os.getpid()})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            send({"id": -1, "ok": False, "error": f"bad request: {e}"})
            continue
        rid, op = req.get("id"), req.get("op")
        t0 = time.perf_counter()
        try:
            if op == "ping":
                send({"id": rid, "ok": True})
            elif op == "shutdown":
                send({"id": rid, "ok": True})
                return 0
            elif op in ("load", "prepare", "start"):
                out = getattr(h, op)(req) or {}
                send({"id": rid, "ok": True, "ms": (time.perf_counter() - t0) * 1000, **out})
            elif op == "chunk":
                y = h.chunk(unb64(req["pcm"]))
                if not np.all(np.isfinite(y)):
                    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
                    send({"id": rid, "ok": True, "pcm": b64(y), "ms": (time.perf_counter() - t0) * 1000, "nan_repaired": True})
                else:
                    send({"id": rid, "ok": True, "pcm": b64(y), "ms": (time.perf_counter() - t0) * 1000})
            elif op == "flush":
                send({"id": rid, "ok": True, "pcm": b64(h.flush()), "ms": (time.perf_counter() - t0) * 1000})
            elif op == "stats":
                send({"id": rid, "ok": True, **cuda_mem()})
            else:
                send({"id": rid, "ok": False, "error": f"unknown op {op!r}"})
        except Exception as e:  # noqa: BLE001
            send({"id": rid, "ok": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-2000:]})
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
