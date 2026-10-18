# Hugging Face model manager

`aivoice models ...` wraps the official `huggingface_hub` library
(`HfApi.list_models`, `model_info(files_metadata=True)`, `snapshot_download`,
`scan_cache_dir`). Weights live in the standard, content-addressed Hugging Face
cache (`~/.cache/huggingface/hub`, or `HF_HOME` / `HF_HUB_CACHE` /
`hf.cache_dir`), shared with every other tool. aivoice keeps only a small
registry (`~/.local/share/aivoice/models/registry.json`) that points at verified
snapshots by commit hash. Nothing is duplicated.

## Commands

| Command | What it does |
|---|---|
| `models search QUERY [--category asr\|tts\|llm\|vad\|audio] [--author] [--library] [--language] [--license] [--quant gguf\|int4\|...] [--sort]` | Hub search; aivoice-supported repos are marked |
| `models info REPO [--revision] [--files]` | license, gated status, commit, file sizes (from Hub metadata), parameter count when the Hub reports it, runtime/adapter verdict, risks |
| `models known` | models aivoice has adapters for, with facts and their provenance |
| `models download REPO [--revision R] [--include GLOB] [--dry-run] [--yes]` | plan -> confirm -> resumable download -> verify -> register |
| `models installed` | registry entries (Hub, imported, runtime-managed) |
| `models verify NAME [--quick]` | sizes + LFS SHA-256 against the manifest recorded at download |
| `models load NAME [--device]` | validate + load + warmup in this process, report timings, release |
| `models loaded` / `models unload` | models held by the running engine |
| `models update NAME [--remove-old]` | move to the newest commit of the requested revision |
| `models remove NAME [--keep-files]` | unregister and delete that cache revision |
| `models cache` | cache usage, aivoice-managed repos marked |
| `models cleanup [--dry-run\|--execute]` | delete unpinned old revisions of aivoice-managed repos only |
| `models import PATH [--category] [--name]` | register an existing local model (never copied, never deleted) |
| `models default asr\|tts\|llm\|vad NAME` | persistent defaults |

`models list` / `models install meanvc2` are the original MeanVC2 commands, unchanged.

## Downloads

1. **Plan**: file list for the resolved commit, total size, bytes still to fetch
   (`snapshot_download(dry_run=True)`), free disk, cache quota (`hf.quota_gib`).
2. **Confirm**: downloads over `hf.confirm_over_gib` (2 GiB) need `--yes`. A 20 to 40
   GB model is never fetched without it.
3. **Download**: pinned to the commit hash, not a moving branch. Progress shows
   bytes, speed and ETA; Ctrl-C cancels cleanly. Network errors retry with
   backoff (2, 4, 8 s); huggingface_hub resumes from its `.incomplete` files, so
   a retry or a re-run continues instead of restarting.
4. **Verify**: every file's size and, for LFS files, SHA-256 must match the Hub
   manifest. Only then is the entry `verified: true`.
5. **Register**: category, runtime, adapter, commit, manifest.

Downloaded is not loadable: `models load` (or the engine) needs a matching
adapter and a successful warmup.

## Gated repositories and tokens

Gated repos (e.g. Gemma) need you to accept the license on the Hub with your
account and authenticate with `hf auth login` or `HF_TOKEN`. aivoice uses
huggingface_hub's own credential handling, never stores or logs the token, and
does not attempt to get around gates.

## Compatibility

`assess()` decides from metadata only (nothing executed):

| Signal | Runtime | aivoice adapter |
|---|---|---|
| `*.gguf` | llama.cpp | `llama-cpp` (LLM) |
| `*.nemo` / `library_name=nemo` | NeMo | `parakeet` (ASR) |
| `fishaudio/*`, `library_name=fish-speech` | fish-speech server | `fish-server` (TTS) |
| `*.onnx` | onnxruntime | `silero-onnx` (VAD only) |
| transformers repos | transformers | none: inspect only |

Warnings are raised for custom code (`*.py`, `auto_map`: would need
`trust_remote_code`, which aivoice never enables) and for pickle-only weights.
A GGUF is not assumed to load in transformers, and a transformers TTS repo is not
assumed to be real-time capable.

## Offline

`HF_HUB_OFFLINE=1` makes `info` answer from the registry, downloads use
`local_files_only`, and search refuses with a clear message.

## Lifecycle

`NOT_INSTALLED -> DOWNLOADING -> INSTALLED -> VALIDATING -> LOADING -> READY -> IN_USE -> UNLOADING`
(or `ERROR`). One load at a time per model, memory reserved with the scheduler
before loading, warmup required, unloading refused while a pipeline uses the model.
