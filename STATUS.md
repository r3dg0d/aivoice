# STATUS — aivoice 0.6.0

Honest limitations (as of 2026-10-21). The 0.6 build environment had **no GPU, no sound devices, and no access to
huggingface.co / arxiv.org / ModelScope** (GitHub and PyPI were reachable). Everything below says which of those
limits applies.

## What works

- Full CLI: search, voices library, providers, doctor, devices, models, live/file/virtualmic
- voice-models.com search via `fetch_data.php` (site AJAX; not a documented public API)
- HF direct downloads when catalog exposes them; cache under `~/.cache/aivoice/downloads/`
- RVC package inspect + MeanVC2 profile from a real reference/preview recording
- Synthesized MeanVC2 reference when a package has a checkpoint but no demo audio: TTS then the RVC model (`AIVOICE_TTS_CMD`, Piper, or espeak-ng; `AIVOICE_RVC_CMD` or `rvc-python`). The voice is labelled `reference_kind: synthesized`. Opt out with `--no-synthesize` / `AIVOICE_NO_SYNTH=1`
- **Live RVC v1/v2** (c7d5db8): persistent `rvc/rt_worker.py` in the rvc-python venv; in 0.6 also behind the shared pipeline (`--engine rvc --unified`)
- Reference prep trims leading and trailing silence and keeps pauses inside the clip
- Re-import replaces an earlier incomplete voice instead of raising `FileExistsError`
- Offline use of installed voices (`virtualmic --voice`)
- Unit tests (mocked/fixture; no live catalog calls in CI; synthesis tests do not load model weights)

## Live voice conversion and modulation (0.6.0)

### Verified in this repository (measured here)
- **DSP modulation engine** (`aivoice modulate`): identity round trip transparent (1e-8), pitch within 10 cents,
  formant warp independent of pitch, output bit-identical for any capture block size, 32 ms measured delay,
  p95 1.7 ms per 10 ms block on one CPU core (RTF ≈ 0.1). 23 presets build and run.
- **Masculine mode** is pitch + formants (experiment M: formants kept; body/intonation/brightness rejected as
  defaults on the GE2E proxy). Identity retention per preset measured (experiment L); `--identity` guard and
  `modulate calibrate` use GE2E similarity.
- **Shared live pipeline**: worker crash, worker timeout and primary restart fall back / recover without silencing
  the stream (tests with a real subprocess worker); 30-minute streams (experiment O): no NaNs, no fallbacks, no
  latency drift, memory bounded after a ring-buffer fix.
- **Worker transport** for resident models: 0.13 ms mean / 0.46 ms p99 IPC overhead per chunk.
- **X-VC streaming schedule** reproduces upstream `run_streaming` sample for sample (numpy port, irregular blocks).
- **MeanVC2 worker handler** drives upstream's `VCRunner` API correctly (fake vendor with the same signatures).
- **Router** filters on runnable state / voice assets / latency budget and ranks on local measurements only.
- **Engine states** are real checks (interpreter, packages, files, GPU); catalogue-only engines cannot start.
- **QuickShell widget**: `run.sh` command lines tested with a fake `aivoice`; `shell.qml` loads offscreen against stub
  Quickshell modules with real CLI JSON and no QML warnings in any mode.

### Integrated but NOT run here
- **X-VC, MeanVC2 (resident), RVC (shared pipeline), StyleStream, StreamVoiceAnon.** Their worker code follows the
  pinned upstream sources, but no weights could be downloaded and there is no GPU: their audio quality, speaker
  similarity, WER, latency, VRAM and RTX 4090 performance are **unmeasured**. `aivoice benchmark vc` measures them.
- Consequently **no VC engine has been shown better than MeanVC2 here**, and `--engine auto` keeps the previous order
  (RVC for trained voices, then MeanVC2, then X-VC) until local benchmarks exist.
- Experiments I, J, K, P, Q (X-VC vs MeanVC2 quality and latency, leakage, RVC vs zero-shot, neural vs DSP
  modulation): defined, **not run**.
- X-VC `balanced`/`lowest-latency` streaming settings (80/60 ms steps) are aivoice's, not validated by the authors.
- bf16 / torch.compile / CUDA graphs / TensorRT: not wired up or measured for any VC engine.
- Glass-to-glass latency through real PipeWire devices: estimated (~100-130 ms for DSP), not measured.

### Not available
- **RT-VC** (no weights, no streaming code, no license), **Zero-VC** (no code), **TVTSyn** (could not be inspected,
  non-commercial), **StreamVoiceAnon+** (weights not released): catalogued, not runnable.
- Neural accent/style modulation beyond StyleStream's curated, non-commercial styles.
- Breathiness *reduction* (needs a harmonic/noise decomposition).

## Requires install / hardware

| Capability | Requirement |
|------------|-------------|
| MeanVC2 convert / live | `models install meanvc2 --yes` + `initialization.py --task all` in the vendor checkout + torch (CUDA preferred) |
| X-VC | `models install xvc --yes` + its own Python 3.10 venv (`AIVOICE_XVC_PYTHON`), CUDA GPU recommended |
| Live RVC | `scripts/aivoice-runtime/build-rvc-venv.sh` (or `AIVOICE_RVC_PYTHON`) + a voice with a `.pth` |
| StyleStream / StreamVoiceAnon | their repo + checkpoints + venv; StreamVoiceAnon needs CUDA |
| Speaker similarity (benchmarks, `--identity`) | `pip install 'aivoice[research]'` (GE2E weights from the resemblyzer wheel) |
| WER/CER in `benchmark vc` | an ASR backend (Moonshine or Parakeet) |
| Live mic / virtual mic | PipeWire with pipewire-pulse (or PulseAudio): `pactl`, `parec`, `pacat` |
| Google Drive / Mega models | Manual download + `voices import-rvc` |

## Known gaps

1. **No direct RVC↔MeanVC2 weight conversion** — by design; docs are explicit.
2. **Speaker fine-tune** (`voices optimize`) not implemented — zero-shot profiles are the default.
3. **Arrow-key TUI** is numbered/TTY select (not full Textual browser yet).
4. Upstream MeanVC2 LICENSE file gap still documented in NOTICE.
5. MeanVC2 / X-VC are not installed on a fresh machine until `models install`.
6. The repository copy of the QuickShell widget has not been launched in a real Quickshell session.
7. RVC `.pth` loading in the rvc-python runtime executes pickle code (trusted models only).

## Experimental real-time mode (0.4.0)

### Implemented and tested in this repository
- Engine: threaded capture → VAD → ASR → stabilizer → (LLM) → expression → TTS → player, bounded drop-oldest queues, per-utterance monotonic timelines, speculative synthesis, barge-in (queue/interrupt/duck), backlog cap, mute, push-to-talk, emergency stop, live status file, signal controls
- Silero VAD from the pip package's ONNX via onnxruntime (no torch), energy VAD fallback, endpointer
- Transcript stabilizer (no repeated or retracted speech in tests)
- Hugging Face model manager (search/info/plan/download/verify/remove/update/import/cache/cleanup/offline) against fake Hub objects
- Fish clients (fish-speech server, hosted API) against a protocol-faithful fake server: streaming WAV, references, consent, key redaction
- Voice profiles with authorization attestation; expression compiler (drama3/s2/s1/none); LLM engine with JSON contract, timeout and fact guard
- Virtual mic create/reuse/cleanup and real audio routing against a real PulseAudio daemon; espeak-ng fallback TTS end to end
- 153 tests pass (151 in a minimal `.[dev]` install; the opt-in real-Pulse test and Silero are skipped there)

### Integrated but not validated with real weights here
The development container had no GPU and could not reach huggingface.co or download.moonshine.ai.
- Moonshine tiny/small/medium (adapter tested against a fake of the moonshine-voice 0.1.5 API)
- Parakeet TDT 0.6B v3 (fake NeMo model); streamed via chunked re-decoding, **not** native streaming
- Drama 3 / S2-Pro through a real fish-speech server; hosted Fish API with a real key
- llama.cpp rewriting with a real GGUF
- No ASR/TTS model latency or WER has been measured yet: see docs/benchmarks.md for what has been

### Known limits
- **Drama 3 does not fit a 24 GB RTX 4090** (upstream: ≥ 48 GB FP8; INT4 build ~25.9 GB reported). Use S2-Pro locally, Drama 3 remote/hosted
- Hosted Fish voice ids cannot be used by a local server (no official export API)
- No built-in GUI for this mode; status is `experimental status --watch`, and the contrib QuickShell widget has a Speech-to-speech tab
- Hosted streaming uses HTTP response streaming, not the `/v1/tts/live` WebSocket
- `flake.nix` is syntax-checked only; not built here
- Speculative synthesis is disabled automatically while a model-backed LLM mode is active
- Latency targets (Verbatim < 300 ms … Character < 1 s) are design targets, not measurements

## Privacy

Search queries go to the selected provider. Microphone audio, converted speech, and reference clips are **not** uploaded by aivoice. Direct VC and modulation are local-only; no fallback is remote. Experimental mode sends recognised text (and local voice reference audio) only to a remote TTS you explicitly allowed (`aivoice experimental remote allow`). Synthesizing a reference loads the local RVC `.pth` (a pickle) and does not upload it.


## Zero-shot cloning (0.5.0, as of 2026-10-20)

### Verified in this repository
- **Engine layer.** Capability checking, the isolated worker protocol (with a real subprocess tone
  engine), the router, the fusion pipeline, profiles v2 and migration, and the clone/engines/research CLI.
- **Engine adapters.** VoxCPM2, VoiceStar and X-Voice are tested against fakes with the official packages'
  signatures (read from the PyPI wheels).
- **Real measurements here.** GE2E similarity (identical to Resemblyzer), Silero VAD, espeak-ng
  synthetic speakers, and experiments C, F and G (docs/research/experiments).

### Not verified here
- **The neural engines.** No VoxCPM2, VoiceStar, X-Voice, MeanVC2 or Fish model was run. huggingface.co
  was unreachable and there is no GPU. Their quality, speed and VRAM on an RTX 4090 are unmeasured;
  figures in the docs that come from upstream READMEs are labelled as such.
- **Engine-dependent experiments.** A, B, D, E and H (engine comparisons, style separation, the full
  pipeline) were not run.
- **WER / CER.** These need an installed ASR backend, so the synthetic benchmark leaves them empty.
- **No Gradio interface.** It was optional and was not built.
- **Unverified sources.** Confucius4-TTS and the EURASIP adaptation article could not be retrieved, and
  nothing depends on them.
- **Vector-field fusion.** It is implemented but unused, because no installed engine pair shares a
  compatible state space.
