"""Streaming VC / modulation engines.

Runnable here: ``dsp`` (numpy, CPU) and ``passthrough``. Worker engines run their upstream code
in their own interpreter through vc/stream_worker.py: ``xvc``, ``meanvc2``, ``rvc``,
``stylestream`` (experimental) and ``streamvoiceanon`` (experimental). Catalogue-only entries
(``rtvc``, ``zerovc``, ``tvtsyn``) describe what was investigated and why it cannot run; they can
never be started.

Every status comes from a check on this machine (interpreter, packages, files). Latency figures
in capabilities are *by construction* (chunk + lookahead + smoothing as configured); measured
numbers come from ``aivoice benchmark vc``.
"""

from __future__ import annotations

import base64
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ..paths import cache_home, models_dir, runtime_dir
from .base import (
    TASK_MOD,
    TASK_VC,
    BaseEngine,
    EngineState,
    EngineStatusVC,
    StreamOptions,
    VCCapabilities,
    VCError,
    VCUnavailable,
    VoiceTarget,
)

STREAM_WORKER = Path(__file__).with_name("stream_worker.py")
RT_WORKER = Path(__file__).resolve().parent.parent / "rvc" / "rt_worker.py"

# Upstream commits inspected for this integration (2026-10-21).
PINS = {
    "xvc": ("https://github.com/Jerrister/X-VC", "49df8c591eafc48b096e466d96f9839f9c0dd739"),
    "meanvc2": ("https://github.com/ASLP-lab/MeanVC2", "13acf84c1bf135ea5edad9c245b345289b06b33e"),
    "stylestream": ("https://github.com/Berkeley-Speech-Group/StyleStream", "38dd18cf6fc5d31e17189d86bb01f302829ba3c9"),
    "streamvoiceanon": ("https://github.com/Plachtaa/StreamVoiceAnon", "201705182c045298225071481e7cd59d537e935e"),
    "rtvc": ("https://github.com/Berkeley-Speech-Group/RT-VC", "9d9a55d0fb6cd53ebba108a0bab3380b1fc3c066"),
    "vossa": ("https://github.com/Morris88826/VOSSA", "5dd1a5c51a1da531b0d127ff48aea007cc96ed34"),
}


def _cfg(engine: str, key: str, default: Any = None) -> Any:
    env = os.environ.get(f"AIVOICE_{engine.upper()}_{key.upper()}")
    if env:
        return env
    try:
        from .. import settings

        v = settings.get(f"engines.{engine}.{key}")
    except Exception:  # noqa: BLE001
        v = None
    return default if v is None else v


def vendor(engine: str) -> Path:
    return Path(_cfg(engine, "repo_dir") or cache_home() / "vendor" / {"xvc": "X-VC", "stylestream": "StyleStream",
                                                                          "streamvoiceanon": "StreamVoiceAnon", "vossa": "VOSSA"}.get(engine, engine))


def _python_has(python: str, *modules: str) -> str | None:
    """None if every module imports-resolves under *python*, else the first missing one."""
    import importlib.util
    import subprocess

    if python == sys.executable:
        for m in modules:
            if importlib.util.find_spec(m) is None:
                return m
        return None
    code = ("import importlib.util,sys\nfor m in sys.argv[1:]:\n    if importlib.util.find_spec(m) is None:\n"
            "        print(m); sys.exit(1)\n")
    try:
        r = subprocess.run([python, "-c", code, *modules], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return modules[0]
    return None if r.returncode == 0 else (r.stdout.strip() or modules[0])


def _cuda_available() -> bool:
    try:
        from ..hardware import detect

        return bool(detect().gpus)
    except Exception:  # noqa: BLE001
        return False


# =============================================================================
# local engines
# =============================================================================


class DspEngine(BaseEngine):
    """Identity-preserving attribute modulation (aivoice.modulation.dsp). Always available."""

    name = "dsp"

    def __init__(self, params=None, *, sample_rate: int = 48000, window_ms: float = 32.0) -> None:
        super().__init__()
        from ..modulation.dsp import ModulationParams, SpectralModulator

        self.mod = SpectralModulator(sample_rate, params or ModulationParams(), window_ms=window_ms, hop_ms=window_ms / 4)

    def capabilities(self) -> VCCapabilities:
        return VCCapabilities(
            name="dsp", display_name="aivoice DSP modulation", task=TASK_MOD, zero_shot=False, requires_reference=False,
            supports_pitch_control=True, supports_cross_lingual=True, sample_rate=self.mod.sr,
            chunk_ms=1000 * self.mod.hop / self.mod.sr, lookahead_ms=0.0, algorithmic_latency_ms=self.mod.latency_ms,
            estimated_vram_mib=0, devices=("cpu",), license="Apache-2.0 (aivoice)", code="aivoice/modulation/dsp.py",
            evidence="aivoice implementation of published DSP methods (phase vocoder, cepstral envelope)",
            notes="Changes attributes of the same voice (pitch, formants, resonance, quality). Not voice cloning, "
                  "not anonymisation.")

    def status(self) -> EngineStatusVC:
        return EngineStatusVC("dsp", EngineState.READY, self.capabilities(), "numpy on CPU")

    def load(self) -> None:
        self.loaded = True

    def prepare_reference(self, target: VoiceTarget | None) -> None:
        self.target = target

    def start_stream(self, options: StreamOptions) -> None:
        super().start_stream(options)
        self.mod.reset()

    def set_params(self, params) -> None:
        self.mod.set_params(params)

    def _process(self, pcm: np.ndarray) -> np.ndarray:
        return self.mod.process(pcm)


class PassthroughEngine(BaseEngine):
    name = "passthrough"

    def __init__(self, sample_rate: int = 48000) -> None:
        super().__init__()
        self.sr = sample_rate

    def capabilities(self) -> VCCapabilities:
        return VCCapabilities(name="passthrough", display_name="Passthrough (no conversion)", task=TASK_MOD,
                              requires_reference=False, sample_rate=self.sr, chunk_ms=None, lookahead_ms=0.0,
                              algorithmic_latency_ms=0.0, estimated_vram_mib=0, devices=("cpu",),
                              license="Apache-2.0 (aivoice)", evidence="n/a", notes="Last-resort fallback: your own voice.")

    def status(self) -> EngineStatusVC:
        return EngineStatusVC("passthrough", EngineState.READY, self.capabilities())

    def load(self) -> None:
        self.loaded = True

    def prepare_reference(self, target: VoiceTarget | None) -> None:
        self.target = target

    def _process(self, pcm: np.ndarray) -> np.ndarray:
        return pcm.copy()


# =============================================================================
# worker engines
# =============================================================================


class WorkerVCEngine(BaseEngine):
    """A model resident in its own interpreter, driven chunk by chunk over JSON lines."""

    name = ""
    handler = ""
    modules: tuple[str, ...] = ("torch",)
    chunk_timeout_s = 5.0

    def __init__(self, *, python: str | None = None, device: str | None = None) -> None:
        super().__init__()
        from ..engines.worker import engine_python

        self.python = python or engine_python(self.name)
        self.device = device or _cfg(self.name, "device", "cuda")
        self.worker = None
        self.sample_rate: int | None = None
        self.load_info: dict[str, Any] = {}
        self.last_ms: float | None = None

    # subclasses: capabilities(), _files() -> dict[str, Path|None], _load_request(), _prepare_request()

    def _files(self) -> dict[str, Path | None]:
        return {}

    def status(self) -> EngineStatusVC:
        caps = self.capabilities()
        if os.path.sep in str(self.python) and not Path(self.python).exists():
            return EngineStatusVC(self.name, EngineState.MISSING_RUNTIME, caps,
                                  f"interpreter {self.python} does not exist (set AIVOICE_{self.name.upper()}_PYTHON); {caps.install}")
        missing = _python_has(self.python, *self.modules)
        if missing:
            return EngineStatusVC(self.name, EngineState.MISSING_RUNTIME, caps,
                                  f"`{missing}` not importable by {self.python} (set AIVOICE_{self.name.upper()}_PYTHON); {caps.install}")
        absent = [k for k, p in self._files().items() if p is None or not Path(p).exists()]
        if absent:
            return EngineStatusVC(self.name, EngineState.MISSING_WEIGHTS, caps, f"missing: {', '.join(absent)}; {caps.install}")
        if "cpu" not in caps.devices and not _cuda_available():
            return EngineStatusVC(self.name, EngineState.UNSUPPORTED_HARDWARE, caps, "needs a CUDA GPU (none detected)")
        return EngineStatusVC(self.name, self._ready_state(), caps, "runtime and weights found")

    def _ready_state(self) -> EngineState:
        return EngineState.READY

    def _load_request(self) -> dict[str, Any]:
        return {"device": self.device}

    def _prepare_request(self, target: VoiceTarget | None) -> dict[str, Any]:
        return {"reference": str(target.reference) if target and target.reference else None}

    def load(self) -> None:
        if self.worker is not None and self.worker.running:
            return
        st = self.status()
        if not st.ready:
            raise VCUnavailable(f"{self.name}: {st.state.value}: {st.detail}")
        self._spawn()

    def _spawn(self) -> None:
        from ..engines.worker import WorkerClient

        runtime_dir().mkdir(parents=True, exist_ok=True)
        self.worker = WorkerClient(self.handler, python=self.python, script=STREAM_WORKER, timeout_s=900.0, start_timeout_s=120.0)
        self.load_info = self.worker.call("load", run_dir=str(runtime_dir()), **self._load_request())
        self.sample_rate = int(self.load_info.get("sample_rate") or self.capabilities().sample_rate or 16000)
        self.loaded = True

    def prepare_reference(self, target: VoiceTarget | None) -> None:
        super().prepare_reference(target)
        self.load()
        self.worker.call("prepare", **self._prepare_request(target))  # type: ignore[union-attr]

    def start_stream(self, options: StreamOptions) -> None:
        super().start_stream(options)
        self.load()
        self.worker.call("start", **self._start_request(options))  # type: ignore[union-attr]

    def _start_request(self, options: StreamOptions) -> dict[str, Any]:
        return {}

    def _process(self, pcm: np.ndarray) -> np.ndarray:
        if self.worker is None:
            raise VCError(f"{self.name}: not loaded")
        r = self.worker.call("chunk", timeout_s=self.chunk_timeout_s,
                             pcm=base64.b64encode(np.ascontiguousarray(pcm, "<f4").tobytes()).decode())
        self.last_ms = r.get("ms")
        return np.frombuffer(base64.b64decode(r["pcm"]), dtype="<f4").copy()

    def flush(self) -> np.ndarray:
        if self.worker is None:
            return np.zeros(0, np.float32)
        r = self.worker.call("flush", timeout_s=30.0)
        return np.frombuffer(base64.b64decode(r["pcm"]), dtype="<f4").copy()

    def stats(self) -> dict[str, Any]:
        if self.worker is None or not self.worker.running:
            return {}
        try:
            return self.worker.call("stats", timeout_s=5.0)
        except Exception:  # noqa: BLE001
            return {}

    def unload(self) -> None:
        if self.worker is not None:
            self.worker.stop()
        self.worker = None
        self.loaded = False


class EchoWorkerEngine(WorkerVCEngine):
    """Real subprocess, no model: tests the worker transport, crash handling and IPC overhead."""

    name = "echo"
    handler = "echo"
    modules = ("numpy",)

    def __init__(self, *, delay_samples: int = 0, crash_after: int | None = None, sleep_ms: float = 0.0, **kw: Any) -> None:
        kw.setdefault("python", sys.executable)
        super().__init__(**kw)
        self.delay_samples, self.crash_after, self.sleep_ms = delay_samples, crash_after, sleep_ms

    def capabilities(self) -> VCCapabilities:
        return VCCapabilities(name="echo", display_name="Echo worker (protocol test)", requires_reference=False,
                              sample_rate=16000, chunk_ms=20, lookahead_ms=0, algorithmic_latency_ms=1000 * self.delay_samples / 16000,
                              devices=("cpu",), evidence="n/a", notes="test engine; never listed for users")

    def _load_request(self) -> dict[str, Any]:
        return {"delay_samples": self.delay_samples, "crash_after": self.crash_after, "sleep_ms": self.sleep_ms}


XVC_QUALITY = {  # (chunk, current, future, smooth) ms
    "best-quality": (2400, 120, 100, 20),  # upstream scripts/batch_infer_seedtts_stream.sh
    "balanced": (2400, 80, 80, 20),  # unvalidated by the authors: measure before relying on it
    "lowest-latency": (2400, 60, 60, 20),  # unvalidated by the authors
}


class XVCEngine(WorkerVCEngine):
    name = "xvc"
    handler = "xvc"
    modules = ("torch", "torchaudio", "hydra", "omegaconf", "transformers", "audiotools", "soxr", "librosa")

    def capabilities(self) -> VCCapabilities:
        q = XVC_QUALITY.get(self.options.quality, XVC_QUALITY["best-quality"])
        return VCCapabilities(
            name="xvc", display_name="X-VC (codec-space streaming VC)", task=TASK_VC, zero_shot=True,
            requires_reference=True, supports_cross_lingual=None, supports_pitch_control=False, sample_rate=16000,
            chunk_ms=q[1], lookahead_ms=q[2], algorithmic_latency_ms=float(q[1] + q[2] + q[3]),
            estimated_vram_mib=None, devices=("cuda", "cpu"),
            license="code MIT (LICENSE in repo); weights: see huggingface.co/chenxie95/X-VC model card (not reachable here)",
            code=f"{PINS['xvc'][0]} @ {PINS['xvc'][1][:7]}", weights="chenxie95/X-VC (xvc.pt); zai-org/glm-4-voice-tokenizer; "
            "ModelScope iic/speech_eres2net_sv_en_voxceleb_16k", paper="arXiv:2604.12456",
            evidence="official-implementation (arXiv preprint)", install="aivoice models install xvc --yes",
            notes="Semantic tokens (GLM-4-Voice tokenizer, a Whisper-large-v3 encoder with VQ) + SAC codec latents, "
                  "ERes2Net speaker embedding + target mel. Streaming recomputes a 2.4 s window every `current` ms. "
                  "Cross-lingual reported in the paper; not verified here. Upstream loads xvc.pt with "
                  "torch.load(weights_only=False); aivoice forces weights_only=True unless engines.xvc.trust_pickle.",
            facts={"latency": "chunk/current/future/smooth from upstream scripts (best-quality); others unvalidated"})

    def _files(self) -> dict[str, Path | None]:
        base = models_dir() / "xvc"
        return {
            "X-VC checkout": vendor("xvc") / "bins" / "infer_utils.py",
            "xvc.pt": Path(_cfg("xvc", "ckpt") or base / "xvc.pt"),
            "GLM-4-Voice tokenizer": Path(_cfg("xvc", "glm_tokenizer_dir") or base / "glm-4-voice-tokenizer") / "config.json",
            "ERes2Net": Path(_cfg("xvc", "eres2net_dir") or base / "speech_eres2net_sv_en_voxceleb_16k") / "configuration.json",
        }

    def _load_request(self) -> dict[str, Any]:
        f = self._files()
        return {"device": self.device, "repo_dir": str(vendor("xvc")), "ckpt": str(f["xvc.pt"]),
                "glm_tokenizer_dir": str(Path(f["GLM-4-Voice tokenizer"]).parent), "eres2net_dir": str(Path(f["ERes2Net"]).parent),
                "trust_pickle": bool(_cfg("xvc", "trust_pickle", False)), "dtype": _cfg("xvc", "dtype", "fp32")}

    def _start_request(self, options: StreamOptions) -> dict[str, Any]:
        c, cur, fut, sm = XVC_QUALITY.get(options.quality, XVC_QUALITY["best-quality"])
        return {"chunk_ms": c, "current_ms": int(options.chunk_ms or cur), "future_ms": fut, "smooth_ms": sm}

class MeanVC2StreamEngine(WorkerVCEngine):
    """MeanVC2 with the model resident (upstream VCRunner), instead of one run_rt.py process per session."""

    name = "meanvc2"
    handler = "meanvc2"
    modules = ("torch", "torchaudio", "safetensors")

    def capabilities(self) -> VCCapabilities:
        m = "120ms" if self.options.quality == "best-quality" else "40ms"
        chunk = 120.0 if m == "120ms" else 40.0
        return VCCapabilities(
            name="meanvc2", display_name="MeanVC2 (resident streaming)", task=TASK_VC, zero_shot=True,
            requires_reference=True, supports_cross_lingual=None, sample_rate=16000, chunk_ms=chunk, lookahead_ms=40.0,
            algorithmic_latency_ms=chunk + 40.0, devices=("cuda", "cpu"),
            license="README claims Apache-2.0; LICENSE file missing upstream (see NOTICE)",
            code=f"{PINS['meanvc2'][0]} @ {PINS['meanvc2'][1][:7]} runtime/run_rt.py VCRunner",
            weights="ASLP-lab/MeanVC2 + WavLM speaker model (initialization.py)", paper="arXiv:2606.09050",
            evidence="official-implementation (arXiv preprint)", install="aivoice models install meanvc2 --yes, then "
            "`python initialization.py --task all` in the vendor checkout",
            notes="Same model as `aivoice live` (run_rt.py), but loaded once and fed by aivoice's capture/fallback "
                  "pipeline. Fixed 2 mean-flow steps (upstream VCRunner). Latency = upstream's chunk+lookahead naming; "
                  "the ASR encoder's receptive field is not included - measure it.")

    def _files(self) -> dict[str, Path | None]:
        from ..paths import vendor_dir

        v = vendor_dir()
        m = "120ms_40ms" if self.options.quality == "best-quality" else "40ms_40ms"
        return {"MeanVC2 runtime": v / "runtime" / "run_rt.py",
                "VC checkpoint": v / "ckpts" / "pretrained_models" / f"meanvc2_{m}.safetensors",
                "vocoder": v / "ckpts" / "vocos" / "vocos.pt",
                "speaker model": v / "preprocess" / "ckpts" / "wavlm_large_finetune.pth"}

    def _load_request(self) -> dict[str, Any]:
        from ..paths import vendor_dir

        ref = self.target.reference if self.target and self.target.reference else None
        if ref is None:
            raise VCError("MeanVC2 needs the target reference before loading (VCRunner takes it at construction)")
        return {"device": self.device, "vendor_dir": str(vendor_dir()), "reference": str(ref),
                "model": "120ms" if self.options.quality == "best-quality" else "40ms"}

    def prepare_reference(self, target: VoiceTarget | None) -> None:
        BaseEngine.prepare_reference(self, target)
        if self.worker is None or not self.worker.running:
            self.load()  # VCRunner extracts the embedding at construction
        else:
            self.worker.call("prepare", **self._prepare_request(target))


class RVCStreamEngine(WorkerVCEngine):
    """RVC v1/v2 (per-voice .pth + optional .index) through rvc/rt_worker.py RvcRealtime in the rvc-python venv."""

    name = "rvc"
    handler = "rvc"
    modules = ("torch", "torchaudio", "rvc_python", "faiss")
    RVC_BLOCKS = {"lowest-latency": (0.15, 0.4, 0.04), "balanced": (0.25, 0.6, 0.05), "best-quality": (0.35, 1.0, 0.08)}

    def __init__(self, **kw: Any) -> None:
        if "python" not in kw:
            kw["python"] = os.environ.get("AIVOICE_RVC_PYTHON") or str(Path.home() / ".local/share/aivoice/rvc-venv/bin/python")
        super().__init__(**kw)

    def capabilities(self) -> VCCapabilities:
        b, _, x = self.RVC_BLOCKS.get(self.options.quality, self.RVC_BLOCKS["balanced"])
        return VCCapabilities(
            name="rvc", display_name="RVC v1/v2 (live)", task=TASK_VC, zero_shot=False, requires_reference=False,
            requires_model_file=True, supports_pitch_control=True, supports_index=True, supports_cross_lingual=True,
            sample_rate=48000, chunk_ms=b * 1000, lookahead_ms=0.0, algorithmic_latency_ms=(b + x + 0.01) * 1000,
            devices=("cuda", "cpu"), license="RVC-Project MIT; rvc-python per its package; voice models: per uploader",
            code="aivoice rvc/rt_worker.py (RVC realtime-GUI design) on rvc-python", weights="per-voice .pth/.index; "
            "rvc-python fetches HuBERT/RMVPE base models itself", evidence="community-implementation",
            install="scripts/aivoice-runtime/build-rvc-venv.sh (Python 3.10 venv with rvc-python)",
            notes="Not zero-shot: needs a model trained on the target voice. .pth files are pickles: rvc-python loads "
                  "them with code execution, so only use models you trust. Latency = block + crossfade + 10 ms SOLA search.")

    def status(self) -> EngineStatusVC:
        st = super().status()
        if st.state == EngineState.READY:
            return EngineStatusVC(self.name, EngineState.INSTALLED, st.caps, "runtime found; needs a voice with an RVC .pth")
        return st

    def _load_request(self) -> dict[str, Any]:
        t = self.target
        if t is None or t.model_file is None:
            raise VCError("RVC needs a voice with an RVC .pth")
        b, e, x = self.RVC_BLOCKS.get(self.options.quality, self.RVC_BLOCKS["balanced"])
        rt = t.extra.get("rt", {})
        return {"device": self.device, "rt_worker": str(RT_WORKER), "model": str(t.model_file),
                "index": str(t.index_file) if t.index_file else "", "pitch": self.options.pitch_semitones or rt.get("pitch", 0.0),
                "index_rate": self.options.index_rate if self.options.index_rate is not None else rt.get("index_rate", 0.5),
                "protect": self.options.protect if self.options.protect is not None else rt.get("protect", 0.33),
                "threshold": rt.get("threshold", -50.0), "block": b, "extra": e, "crossfade": x}

    def prepare_reference(self, target: VoiceTarget | None) -> None:
        BaseEngine.prepare_reference(self, target)
        if self.worker is not None:
            self.unload()  # a different .pth means a different model
        self.load()

    def _start_request(self, options: StreamOptions) -> dict[str, Any]:
        return {"pitch": options.pitch_semitones, "index_rate": options.index_rate}

    @property
    def chunk_timeout_s(self) -> float:  # type: ignore[override]
        return 10.0


class StyleStreamEngine(WorkerVCEngine):
    name = "stylestream"
    handler = "stylestream"
    modules = ("torch", "torchaudio", "lightning", "torchdiffeq", "x_transformers", "transformers", "librosa")

    def capabilities(self) -> VCCapabilities:
        return VCCapabilities(
            name="stylestream", display_name="StyleStream (experimental style conversion)", task=TASK_VC, zero_shot=False,
            requires_reference=True, supports_style=True, supports_cross_lingual=None, sample_rate=16000, chunk_ms=600.0,
            lookahead_ms=5.0, algorithmic_latency_ms=645.0, devices=("cuda", "cpu"),
            license="Research, Educational and Not-for-Profit license (UC Regents): NO commercial use",
            code=f"{PINS['stylestream'][0]} @ {PINS['stylestream'][1][:7]}",
            weights="Louis0324/StyleStream (stylizer without style encoder, destylizer, causal Vocos)",
            paper="arXiv:2602.20113 (Interspeech 2026)", evidence="official-implementation",
            install="clone the repo, download its checkpoints into assets/ckpts, set AIVOICE_STYLESTREAM_PYTHON",
            notes="Timbre/accent/emotion style conversion. The public release withholds the style encoder: targets "
                  "must be curated style folders (.wav + pre-extracted .npy), not arbitrary voices. ~1 s end-to-end "
                  "(600 ms chunks + compute): too slow for gaming, offered as an experimental modulation backend.")

    def _ready_state(self) -> EngineState:
        return EngineState.EXPERIMENTAL

    def _files(self) -> dict[str, Path | None]:
        c = vendor("stylestream") / "assets" / "ckpts"
        return {n: c / n for n in ("stylizer-no-style-enc.ckpt", "destylizer.ckpt", "vocos_causal_best.ckpt")}

    def _load_request(self) -> dict[str, Any]:
        return {"device": self.device, "repo_dir": str(vendor("stylestream")), "steps": int(_cfg("stylestream", "steps", 10))}

    def _prepare_request(self, target: VoiceTarget | None) -> dict[str, Any]:
        d = (target.extra.get("style_dir") if target else None) or (str(Path(target.reference).parent) if target and target.reference else None)
        if not d or not list(Path(d).glob("*.npy")):
            raise VCError("StyleStream needs a style folder with a .wav and its pre-extracted .npy embedding "
                          "(the public release cannot embed arbitrary voices)")
        return {"style_dir": d, "reference": str(target.reference) if target and target.reference else None}

    def prepare_reference(self, target: VoiceTarget | None) -> None:
        self.target = target
        self.load()
        self.worker.call("prepare", **self._prepare_request(target))  # type: ignore[union-attr]

    chunk_timeout_s = 10.0


class StreamVoiceAnonEngine(WorkerVCEngine):
    name = "streamvoiceanon"
    handler = "streamvoiceanon"
    modules = ("torch", "torchaudio", "hydra", "omegaconf", "librosa")

    def capabilities(self) -> VCCapabilities:
        return VCCapabilities(
            name="streamvoiceanon", display_name="StreamVoiceAnon (experimental anonymisation / VC)", task=TASK_VC,
            zero_shot=True, requires_reference=True, supports_cross_lingual=None, sample_rate=44100,
            chunk_ms=1000 * 2048 / 44100, lookahead_ms=2 * 1000 * 2048 / 44100, algorithmic_latency_ms=3 * 1000 * 2048 / 44100,
            devices=("cuda",), license="Apache-2.0 (repo LICENSE); weights per Plachta/StreamVoiceAnon model card",
            code=f"{PINS['streamvoiceanon'][0]} @ {PINS['streamvoiceanon'][1][:7]} evaluations/infer_arvc.py",
            weights="Plachta/StreamVoiceAnon", paper="arXiv:2601.13948 (ICASSP 2026); StreamVoiceAnon+ arXiv:2603.06079",
            evidence="official-implementation", install="clone the repo, `hf download Plachta/StreamVoiceAnon --local-dir "
            "pretrained_checkpoints/`, set AIVOICE_STREAMVOICEANON_PYTHON",
            notes="Causal codec language model (Firefly codec, 46 ms frames, 2-frame decoder delay). `alpha` < 1 mixes "
                  "noise into the speaker embedding for anonymisation. The paper reports ~180 ms latency. The "
                  "StreamVoiceAnon+ (emotion-preserving) weights were 'coming soon' at the pinned commit.")

    def _ready_state(self) -> EngineState:
        return EngineState.EXPERIMENTAL

    def _files(self) -> dict[str, Path | None]:
        p = vendor("streamvoiceanon") / "pretrained_checkpoints"
        return {n: p / n for n in ("dual_ar_delay_0_8.pth", "asr_s2s_bsq_8192_causal_down_whisper.pth",
                                   "firefly-gan-vq-fsq-8x1024-21hz-generator.pth", "spark_speaker_encoder.pth")}

    def _load_request(self) -> dict[str, Any]:
        return {"device": self.device, "repo_dir": str(vendor("streamvoiceanon")),
                "trust_pickle": bool(_cfg("streamvoiceanon", "trust_pickle", False))}

    chunk_timeout_s = 10.0


class VossaEngine(WorkerVCEngine):
    name = "vossa"
    handler = "vossa"
    modules = ("torch", "torchaudio", "einops", "soxr", "pytorch_lightning", "librosa")

    def capabilities(self) -> VCCapabilities:
        chunk = int(_cfg("vossa", "chunk_frames", 3))
        la = int(_cfg("vossa", "lookahead", 4))
        return VCCapabilities(
            name="vossa", display_name="VOSSA (official, experimental)", task=TASK_VC, zero_shot=True,
            requires_reference=True, sample_rate=16000, chunk_ms=20.0 * chunk, lookahead_ms=20.0 * la,
            algorithmic_latency_ms=20.0 * (chunk + la), devices=("cuda", "cpu"),
            license="CC BY-NC 4.0 (code and checkpoints): non-commercial use only",
            code=f"{PINS['vossa'][0]} @ {PINS['vossa'][1][:7]} src/streaming (VossaStreamer)",
            weights="morris88826/VOSSA (official)", paper="Interspeech 2026, doi:10.21437/Interspeech.2026-2763",
            evidence="official-implementation",
            install="clone the repo, `hf download morris88826/VOSSA --local-dir checkpoints`, set AIVOICE_VOSSA_PYTHON "
                    "(Python 3.10, torch 2.8, requirements.txt)",
            notes="Official code and weights. The paper reports RTF ~0.25 and ~73 ms end-to-end latency. The encoder "
                  "looks 4 frames ahead per layer; the official streamer windows only those 4 frames (approximation "
                  "of the offline model). AIVOICE_VOSSA_LOOKAHEAD=0 is the README's zero-lookahead setting.")

    def _ready_state(self) -> EngineState:
        return EngineState.EXPERIMENTAL

    def _files(self) -> dict[str, Path | None]:
        d = Path(_cfg("vossa", "checkpoint_dir") or vendor("vossa") / "checkpoints")
        return {"VOSSA checkpoint": next(iter(sorted(d.glob("vossa_*.ckpt"))), None) if d.exists() else None,
                "mimivocoder.ckpt": d / "mimivocoder.ckpt"}

    def _load_request(self) -> dict[str, Any]:
        la = _cfg("vossa", "lookahead", None)
        return {"device": self.device, "repo_dir": str(vendor("vossa")),
                "checkpoint_dir": str(_cfg("vossa", "checkpoint_dir") or vendor("vossa") / "checkpoints"),
                "chunk_frames": int(_cfg("vossa", "chunk_frames", 3)), "lookahead": None if la is None else int(la),
                "trust_pickle": bool(_cfg("vossa", "trust_pickle", False))}


class ReproVCEngine(WorkerVCEngine):
    """A paper reproduction trained with `aivoice reproduce`; startable only with a trained checkpoint."""

    handler = "repro"
    modules = ("torch", "numpy", "librosa")
    chunk_timeout_s = 5.0

    def __init__(self, key: str, **kw: Any) -> None:
        kw.setdefault("python", sys.executable)
        self.name = key
        super().__init__(**kw)
        self.device = kw.get("device") or _cfg(key.replace("-", "_"), "device", "cpu")

    def capabilities(self) -> VCCapabilities:
        from ..research_reproductions import registry

        r = registry.get(self.name)
        info = r.info
        la_ms = 1000 * info.lookahead_samples / info.sample_rate
        if self.name == "tvtsyn-aivoice":
            la_ms = 80.0
        frame_ms = 1000 * info.hop_samples / info.sample_rate
        return VCCapabilities(
            name=self.name, display_name=f"{info.title.split(':')[0]} ({registry.PROVENANCE[self.name].value}, experimental)",
            task=TASK_VC, zero_shot=True, sample_rate=info.sample_rate, chunk_ms=frame_ms, lookahead_ms=la_ms,
            algorithmic_latency_ms=frame_ms + la_ms, devices=("cpu", "cuda"),
            license=f"aivoice code {info.code_license}; weights: trained by you (data licenses apply); "
                    f"official code: {info.official_license or 'none'}",
            code=f"aivoice research_reproductions ({info.source_status.value})",
            weights=f"{info.model_name} (NOT official)", paper=info.paper_key, evidence="unvalidated",
            install=f"aivoice reproduce train {self.name} ... then copy the exported checkpoint to "
                    f"{registry.repro_home() / self.name / 'model.pt'} (or set AIVOICE_"
                    f"{self.name.upper().replace('-', '_')}_CHECKPOINT)",
            notes=f"{info.source_status.value}, architecture confidence {info.arch_confidence.value}. Never chosen by "
                  "auto. Random-init or stand-in checkpoints are refused.")

    def status(self) -> EngineStatusVC:
        from ..research_reproductions import registry
        from ..research_reproductions.common.checkpoint import is_usable_for_inference, read_meta

        caps = self.capabilities()
        ck = registry.checkpoint_for(self.name)
        if ck is None:
            return EngineStatusVC(self.name, EngineState.MISSING_WEIGHTS, caps,
                                  "no trained checkpoint (none exist publicly; train one with `aivoice reproduce`)")
        try:
            ok, why = is_usable_for_inference(read_meta(ck))
        except Exception as e:  # noqa: BLE001
            return EngineStatusVC(self.name, EngineState.MISSING_WEIGHTS, caps, f"{ck}: {e}")
        if not ok:
            return EngineStatusVC(self.name, EngineState.MISSING_WEIGHTS, caps, f"{ck}: {why}")
        return EngineStatusVC(self.name, EngineState.EXPERIMENTAL, caps, f"{ck}: {why}")

    def _load_request(self) -> dict[str, Any]:
        from ..research_reproductions import registry

        return {"engine": self.name, "checkpoint": str(registry.checkpoint_for(self.name)), "device": self.device}


def repro_engines() -> list[str]:
    from ..research_reproductions import registry

    return [k for k in registry.ENTRIES if k not in ("vossa-repro",)]


# =============================================================================
# investigated, not runnable
# =============================================================================


class CatalogOnlyEngine:
    """An investigated architecture with no runnable public release. It can never be started."""

    def __init__(self, caps: VCCapabilities, state: EngineState, detail: str) -> None:
        self._caps, self._state, self._detail = caps, state, detail
        self.name = caps.name

    def capabilities(self) -> VCCapabilities:
        return self._caps

    def status(self) -> EngineStatusVC:
        return EngineStatusVC(self.name, self._state, self._caps, self._detail)

    def load(self) -> None:
        raise VCUnavailable(f"{self.name}: {self._state.value}: {self._detail}")


def catalog_only() -> list[CatalogOnlyEngine]:
    """Investigated architectures that cannot run at all. RT-VC, Zero-VC and TVTSyn moved to `ReproVCEngine`
    (0.7): they are runnable once you train them with `aivoice reproduce`; VOSSA has an official release."""
    return []

ENGINES = {
    "dsp": DspEngine, "passthrough": PassthroughEngine, "xvc": XVCEngine, "meanvc2": MeanVC2StreamEngine,
    "rvc": RVCStreamEngine, "stylestream": StyleStreamEngine, "streamvoiceanon": StreamVoiceAnonEngine,
    "vossa": VossaEngine,
}


def get_vc_engine(name: str, **kw: Any):
    key = name.lower().replace("-", "").replace("_", "")
    key = {"xvc": "xvc", "meanvc": "meanvc2", "rvcv2": "rvc", "svanon": "streamvoiceanon"}.get(key, key)
    if key in ENGINES:
        return ENGINES[key](**kw)
    from ..research_reproductions import registry

    try:
        rk = registry.resolve(name)
    except KeyError:
        rk = None
    if rk == "vossa-repro":
        return VossaEngine(**kw)
    if rk is not None:
        return ReproVCEngine(rk, **kw)
    for c in catalog_only():
        if c.name == key:
            return c
    raise VCError(f"unknown VC engine {name!r}; known: {', '.join(list(ENGINES) + repro_engines())}")


def all_vc_status() -> list[EngineStatusVC]:
    out = []
    for name in ENGINES:
        try:
            out.append(get_vc_engine(name).status())
        except Exception as e:  # noqa: BLE001
            eng = ENGINES[name]
            out.append(EngineStatusVC(name, EngineState.MISSING_RUNTIME, eng().capabilities() if name in ("dsp", "passthrough") else
                                      VCCapabilities(name=name, display_name=name), f"{type(e).__name__}: {e}"))
    for key in repro_engines():
        try:
            out.append(ReproVCEngine(key).status())
        except Exception as e:  # noqa: BLE001
            out.append(EngineStatusVC(key, EngineState.MISSING_WEIGHTS, VCCapabilities(name=key, display_name=key),
                                      f"{type(e).__name__}: {e}"))
    out += [c.status() for c in catalog_only()]
    return out


__all__ = ["ENGINES", "PINS", "CatalogOnlyEngine", "ReproVCEngine", "VossaEngine", "repro_engines", "DspEngine", "EchoWorkerEngine", "MeanVC2StreamEngine", "PassthroughEngine",
           "RVCStreamEngine", "StreamVoiceAnonEngine", "StyleStreamEngine", "WorkerVCEngine", "XVCEngine", "all_vc_status",
           "catalog_only", "get_vc_engine", "vendor"]
