"""Model/runtime compatibility registry.

Two halves:

* ``KNOWN_MODELS`` - models aivoice has an adapter for, with what is known about
  them. Each fact carries where it came from; anything not checked against an
  upstream source is labelled an estimate or "unverified". ``aivoice models info``
  always re-reads live Hub metadata when online, which wins over this table.
* ``assess()`` - inspects *any* repository's file list / metadata and decides
  which runtime could load it, whether aivoice has an adapter for that runtime,
  and what is risky about it (custom code, pickles). Nothing is downloaded or
  executed to answer that.
"""

from __future__ import annotations

import fnmatch
import importlib.util
from dataclasses import dataclass, field
from typing import Any, Iterable

CATEGORIES = ("asr", "tts", "llm", "vad", "audio")

# Hub pipeline tags per category (used to filter searches).
PIPELINE_TAGS = {
    "asr": ["automatic-speech-recognition"],
    "tts": ["text-to-speech", "text-to-audio"],
    "llm": ["text-generation"],
    "vad": ["voice-activity-detection"],
    "audio": ["audio-to-audio", "audio-classification"],
}


@dataclass(frozen=True)
class RuntimeSpec:
    name: str
    module: str | None  # importable module that provides it, None = external server
    install: str
    notes: str = ""

    def installed(self) -> bool:
        return self.module is None or importlib.util.find_spec(self.module) is not None


RUNTIMES: dict[str, RuntimeSpec] = {
    "nemo": RuntimeSpec("nemo", "nemo", "pip install 'nemo_toolkit[asr]'", "NVIDIA NeMo (.nemo checkpoints)"),
    "moonshine-voice": RuntimeSpec(
        "moonshine-voice",
        "moonshine_voice",
        "pip install moonshine-voice",
        "Moonshine C++/ONNX runtime; fetches its own .ort models from download.moonshine.ai",
    ),
    "onnxruntime": RuntimeSpec("onnxruntime", "onnxruntime", "pip install onnxruntime"),
    "llama.cpp": RuntimeSpec("llama.cpp", "llama_cpp", "pip install llama-cpp-python", "GGUF models"),
    "fish-speech-server": RuntimeSpec(
        "fish-speech-server",
        None,
        "run fish-speech tools/api_server.py, SGLang-Omni, or vLLM-Omni; aivoice talks HTTP to it",
        "Drama 3 / S2-Pro are served by a separate process; aivoice never loads them in-process",
    ),
    "transformers": RuntimeSpec("transformers", "transformers", "pip install transformers"),
}

# Which aivoice adapter can drive a runtime for a category. Absent = inspect only.
ADAPTERS: dict[tuple[str, str], str] = {
    ("nemo", "asr"): "parakeet",
    ("moonshine-voice", "asr"): "moonshine",
    ("llama.cpp", "llm"): "llama-cpp",
    ("fish-speech-server", "tts"): "fish-server",
    ("onnxruntime", "vad"): "silero-onnx",
}


@dataclass
class KnownModel:
    key: str
    category: str
    runtime: str
    repo_id: str | None = None  # None: not distributed via the Hub
    source: str = "huggingface"
    params: str = ""
    license: str = ""
    gated: bool | None = None
    est_disk_gib: float | None = None
    est_vram_mib: int | None = None  # None: cannot / should not run on GPU
    est_ram_mib: int | None = None  # None: cannot run on CPU
    allow_patterns: list[str] | None = None
    facts: dict[str, str] = field(default_factory=dict)  # fact -> provenance
    notes: str = ""
    aliases: tuple[str, ...] = ()

    @property
    def adapter(self) -> str | None:
        return ADAPTERS.get((self.runtime, self.category))


VERIFIED = "verified against upstream docs/source (2026-10)"
REPORTED = "reported by third parties; not independently verified"
ESTIMATE = "estimate from parameter count; measure with `aivoice models benchmark`"

KNOWN_MODELS: list[KnownModel] = [
    # --- ASR ---------------------------------------------------------------
    KnownModel(
        key="parakeet",
        category="asr",
        runtime="nemo",
        repo_id="nvidia/parakeet-tdt-0.6b-v3",
        params="0.6B",
        license="CC-BY-4.0 (see model card)",
        gated=False,
        est_disk_gib=2.5,
        est_vram_mib=3500,
        est_ram_mib=5000,
        allow_patterns=["*.nemo", "*.json", "*.md"],
        facts={
            "repo_id": VERIFIED,
            "25 European languages, automatic language detection": VERIFIED,
            "word/segment timestamps via transcribe(timestamps=True)": VERIFIED,
            "NeMo transcribe() accepts numpy arrays": "verified in nemo_toolkit 3.0.0 source",
            "not a cache-aware streaming checkpoint": "no NeMo streaming support documented for v3",
            "memory": ESTIMATE,
        },
        notes=(
            "Offline (full-context) TDT model. aivoice streams it with a chunked re-decoding "
            "adapter, which is NOT native streaming: partials cost a re-decode of the utterance so far."
        ),
        aliases=("parakeet-tdt-0.6b-v3",),
    ),
    *[
        KnownModel(
            key=f"moonshine-{size}",
            category="asr",
            runtime="moonshine-voice",
            repo_id=None,
            source="moonshine-voice (download.moonshine.ai)",
            params=params,
            license=(
                "non-English models: Moonshine Community License (non-commercial), per moonshine-voice; "
                "check https://www.moonshine.ai/license for English"
            ),
            est_vram_mib=None,
            est_ram_mib=ram,
            facts={
                f"ModelArch.{size.upper()}_STREAMING exists in moonshine-voice 0.1.5": "verified in package source",
                "native streaming with cached state, LineStarted/LineTextChanged/LineCompleted events": "verified in package source",
                "params": params_src,
                "memory": ESTIMATE,
            },
            notes=(
                "Runs on CPU via the Moonshine C++ runtime, leaving the GPU to TTS. "
                "The Hugging Face repos (UsefulSensors/moonshine-streaming-*) hold the transformers "
                "checkpoint, not the .ort files this runtime uses; aivoice fetches the runtime's own files."
            ),
        )
        for size, params, params_src, ram in (
            ("tiny", "~34M", REPORTED, 400),
            ("small", "~123M", REPORTED, 700),
            ("medium", "~245M", REPORTED, 1200),
        )
    ],
    # --- TTS ---------------------------------------------------------------
    KnownModel(
        key="drama-3",
        category="tts",
        runtime="fish-speech-server",
        repo_id="fishaudio/drama-3",
        params="35B total / 3B active (MoE)",
        license="Fish Audio Community License v1.0 (non-commercial free; commercial under a revenue threshold)",
        est_disk_gib=41.5,
        est_vram_mib=48 * 1024,
        est_ram_mib=None,
        facts={
            "repo_id": VERIFIED,
            "35B total / 3B active": VERIFIED,
            ">=48 GB GPU in FP8; no CPU/MPS/ROCm": "upstream model card as quoted in Oct 2026 coverage",
            "served by fish-speech api_server / SGLang-Omni / vLLM-Omni": VERIFIED,
            "disk size 41.5 GB": REPORTED,
        },
        notes=(
            "Does NOT fit a 24 GB RTX 4090, even with the community INT4 build (~25.9 GB reported). "
            "Run it on a >=48 GB GPU box (remote) or use the hosted Fish Audio API; use S2-Pro locally."
        ),
    ),
    KnownModel(
        key="drama-3-int4",
        category="tts",
        runtime="fish-speech-server",
        repo_id="fishaudio/drama-3-int4",
        params="35B total / 3B active (INT4)",
        license="see model card",
        est_disk_gib=25.9,
        est_vram_mib=27 * 1024,
        est_ram_mib=None,
        facts={"size 25.9 GB": REPORTED, "affiliation with Fish Audio": "unverified"},
        notes="Weights alone exceed 24 GB; not loadable on a single RTX 4090 without offload.",
    ),
    KnownModel(
        key="s2-pro",
        category="tts",
        runtime="fish-speech-server",
        repo_id="fishaudio/s2-pro",
        params="4B",
        license="see model card",
        est_vram_mib=12 * 1024,
        est_ram_mib=None,
        facts={
            "repo_id": VERIFIED,
            "same api_server and request schema as Drama 3": VERIFIED,
            "small enough for a consumer GPU": "fish-speech README",
            "memory": ESTIMATE,
        },
        notes="The realistic local Fish model on a 24 GB card. Uses [bracket] tags; no Drama 3 paired tags.",
    ),
    # --- LLM ---------------------------------------------------------------
    KnownModel(
        key="qwen2.5-0.5b",
        category="llm",
        runtime="llama.cpp",
        repo_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        params="0.5B",
        license="Apache-2.0",
        gated=False,
        est_disk_gib=0.5,
        est_vram_mib=900,
        est_ram_mib=900,
        allow_patterns=["*q4_k_m.gguf"],
        facts={"memory": ESTIMATE},
        notes="Fastest rewrite model; fine for cleanup, weak at character rewrites.",
    ),
    KnownModel(
        key="qwen2.5-1.5b",
        category="llm",
        runtime="llama.cpp",
        repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        params="1.5B",
        license="Apache-2.0",
        gated=False,
        est_disk_gib=1.1,
        est_vram_mib=1800,
        est_ram_mib=1800,
        allow_patterns=["*q4_k_m.gguf"],
        facts={"memory": ESTIMATE},
        notes="Default for natural/expressive modes on CPU.",
    ),
    KnownModel(
        key="qwen3-1.7b",
        category="llm",
        runtime="llama.cpp",
        repo_id="Qwen/Qwen3-1.7B-GGUF",
        params="1.7B",
        license="Apache-2.0",
        gated=False,
        est_disk_gib=1.9,
        est_vram_mib=2400,
        est_ram_mib=2400,
        allow_patterns=["*Q8_0.gguf"],
        facts={"memory": ESTIMATE},
        notes="Hybrid thinking model; aivoice sends /no_think to keep latency bounded.",
    ),
    KnownModel(
        key="gemma-3-1b",
        category="llm",
        runtime="llama.cpp",
        repo_id="google/gemma-3-1b-it-qat-q4_0-gguf",
        params="1B",
        license="Gemma Terms of Use",
        gated=True,
        est_disk_gib=1.0,
        est_vram_mib=1500,
        est_ram_mib=1500,
        allow_patterns=["*.gguf"],
        facts={"gated": "Google gates Gemma repos behind a license acceptance", "memory": ESTIMATE},
        notes="Gated: accept the license on the Hub and `hf auth login` first.",
    ),
    # --- VAD ---------------------------------------------------------------
    KnownModel(
        key="silero-vad",
        category="vad",
        runtime="onnxruntime",
        repo_id=None,
        source="silero-vad pip package (bundled ONNX)",
        params="~2M",
        license="MIT",
        est_vram_mib=None,
        est_ram_mib=50,
        facts={"ONNX signature input/state/sr -> output/stateN": "verified by loading silero_vad.onnx 6.2.3"},
        notes="aivoice runs the bundled ONNX file directly with onnxruntime, so torch is not needed.",
    ),
]


def find_known(name: str) -> KnownModel | None:
    low = name.strip().lower()
    for m in KNOWN_MODELS:
        if low == m.key or low in m.aliases or (m.repo_id and low == m.repo_id.lower()):
            return m
    return None


def known_by_category(category: str) -> list[KnownModel]:
    return [m for m in KNOWN_MODELS if m.category == category]


# --- assessing arbitrary repositories ---------------------------------------

PICKLE_EXT = (".bin", ".pt", ".pth", ".ckpt", ".pkl")


@dataclass
class Assessment:
    runtime: str | None
    category: str | None
    adapter: str | None
    usable: bool
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    requires_custom_code: bool = False
    missing_dependency: str | None = None
    quantization: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _any(files: Iterable[str], *patterns: str) -> list[str]:
    files = list(files)
    return [f for f in files if any(fnmatch.fnmatch(f.lower(), p) for p in patterns)]


def guess_quantization(files: Iterable[str]) -> str | None:
    for f in files:
        low = f.lower()
        for q in ("q2_k", "q3_k", "q4_0", "q4_k_m", "q4_k_s", "q5_k_m", "q5_0", "q6_k", "q8_0", "iq4", "f16", "bf16"):
            if q in low and low.endswith(".gguf"):
                return q.upper()
        for q in ("int4", "int8", "fp8", "nvfp4", "w4a16", "awq", "gptq"):
            if q in low:
                return q.upper()
    return None


def assess(
    *,
    files: list[str],
    library_name: str | None = None,
    pipeline_tag: str | None = None,
    tags: list[str] | None = None,
    config: dict[str, Any] | None = None,
    repo_id: str | None = None,
    category_hint: str | None = None,
) -> Assessment:
    """Decide how (and whether) aivoice could load a repository, from metadata only."""
    tags = [t.lower() for t in (tags or [])]
    lib = (library_name or "").lower()
    reasons: list[str] = []
    warnings: list[str] = []

    category = category_hint
    if category is None and pipeline_tag:
        for cat, ptags in PIPELINE_TAGS.items():
            if pipeline_tag in ptags:
                category = cat
                break

    runtime: str | None = None
    known = find_known(repo_id) if repo_id else None
    if known:
        runtime, category = known.runtime, known.category
        reasons.append(f"listed in aivoice's registry as {known.key}")
    elif _any(files, "*.gguf"):
        runtime = "llama.cpp"
        reasons.append("contains .gguf files")
        category = category or "llm"
    elif _any(files, "*.nemo") or lib == "nemo":
        runtime = "nemo"
        reasons.append("NeMo checkpoint (.nemo / library_name=nemo)")
    elif lib in ("fish-speech", "fish_speech") or "fish-speech" in tags or (
        repo_id or ""
    ).lower().startswith("fishaudio/"):
        runtime = "fish-speech-server"
        reasons.append("fish-speech model (served by a fish-speech compatible server)")
        category = category or "tts"
    elif _any(files, "*.onnx", "*.ort"):
        runtime = "onnxruntime"
        reasons.append("contains ONNX files")
    elif lib == "transformers" or _any(files, "config.json"):
        runtime = "transformers"
        reasons.append("transformers-style repository")

    custom_code = bool(_any(files, "*.py")) or bool((config or {}).get("auto_map"))
    if custom_code:
        warnings.append(
            "repository ships custom Python code (auto_map / *.py): loading it would need "
            "trust_remote_code. aivoice never enables that without --trust-remote-code."
        )
    pickles = _any(files, *("*" + e for e in PICKLE_EXT))
    if pickles and not _any(files, "*.safetensors", "*.gguf", "*.nemo", "*.onnx"):
        warnings.append(f"weights are pickle-based ({pickles[0]} ...); loading a pickle can execute code")
    if runtime == "nemo":
        warnings.append(".nemo archives contain pickled state; only load from publishers you trust")

    adapter = ADAPTERS.get((runtime, category)) if runtime and category else None
    usable = adapter is not None
    missing = None
    if runtime is None:
        reasons.append("no known runtime matches this repository's files")
    elif adapter is None:
        reasons.append(
            f"runtime {runtime!r} is recognised but aivoice has no {category or 'unknown-category'} "
            "adapter for it (inspect only)"
        )
    else:
        spec = RUNTIMES.get(runtime)
        if spec and not spec.installed():
            missing = spec.install
            reasons.append(f"runtime not installed: {spec.install}")
        if runtime == "transformers" and custom_code:
            usable = False
    return Assessment(
        runtime=runtime,
        category=category,
        adapter=adapter,
        usable=usable,
        reasons=reasons,
        warnings=warnings,
        requires_custom_code=custom_code,
        missing_dependency=missing,
        quantization=guess_quantization(files),
    )


def estimate_memory_mib(params_billion: float, bits: float = 16, overhead: float = 1.25) -> int:
    """Weights * overhead (KV cache, activations). An estimate, never a guarantee."""
    return int(params_billion * 1e9 * bits / 8 / (1024 * 1024) * overhead)
