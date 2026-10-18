# STATUS — aivoice 0.4.0

Honest limitations (as of 2026-10-18).

## What works

- Full CLI: search, voices library, providers, doctor, devices, models, live/file/virtualmic
- voice-models.com search via `fetch_data.php` (site AJAX; not a documented public API)
- HF direct downloads when catalog exposes them; cache under `~/.cache/aivoice/downloads/`
- RVC package inspect + MeanVC2 profile from a real reference/preview recording
- Synthesized MeanVC2 reference when a package has a checkpoint but no demo audio: TTS then the RVC model (`AIVOICE_TTS_CMD`, Piper, or espeak-ng; `AIVOICE_RVC_CMD` or `rvc-python`). The voice is labelled `reference_kind: synthesized`. Opt out with `--no-synthesize` / `AIVOICE_NO_SYNTH=1`. This builds reference audio only — it does not convert RVC weights and it is not a live RVC converter
- Reference prep trims leading and trailing silence and keeps pauses inside the clip
- Re-import replaces an earlier incomplete voice instead of raising `FileExistsError`
- Files aivoice wrote (placeholder wav, `.aivoice_synth/`) are ignored when looking for a recording
- Offline use of installed voices (`virtualmic --voice`)
- Unit tests (mocked/fixture; no live catalog calls in CI; synthesis tests do not load model weights)

## Requires install / hardware

| Capability | Requirement |
|------------|-------------|
| MeanVC2 convert / live | `models install meanvc2 --yes` + torch (CUDA preferred) + upstream deps |
| Synthesized reference | A TTS engine and an RVC inference engine on the machine (see README). `rvc-python` pulls torch and may download its own base models on first use; the unit tests do not |
| Live mic / virtualmic stream | Vendor `runtime/run_rt.py` + audio stack |
| Audio prep polish | `ffmpeg` (optional; falls back to copy) |
| Google Drive / Mega models | Manual download + `voices import-rvc` |

## Known gaps

1. **No direct RVC↔MeanVC2 weight conversion** — by design; docs are explicit.
2. **No live RVC converter.** If synthesis is disabled, no engine is available, or an engine fails (including audio under the usable-size floor), import still stores a metadata-only incomplete `engine=rvc` entry. MeanVC2 remains the runtime once a reference exists.
3. **Speaker fine-tune** (`voices optimize`) not implemented — zero-shot profiles are the default.
4. **Quickshell widget** not bundled in this repo.
5. **Arrow-key TUI** is numbered/TTY select (not full Textual browser yet).
6. Upstream MeanVC2 LICENSE file gap still documented in NOTICE.
7. MeanVC2 not installed on a fresh machine until `models install`.

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
- No GUI (the project has none); status is `experimental status --watch`
- Hosted streaming uses HTTP response streaming, not the `/v1/tts/live` WebSocket
- `flake.nix` is syntax-checked only; not built here
- Speculative synthesis is disabled automatically while a model-backed LLM mode is active
- Latency targets (Verbatim < 300 ms … Character < 1 s) are design targets, not measurements

## Privacy

Search queries go to the selected provider. Microphone audio, converted speech, and reference clips are **not** uploaded by aivoice. Experimental mode sends recognised text (and local voice reference audio) only to a remote TTS you explicitly allowed (`aivoice experimental remote allow`). Synthesizing a reference loads the local RVC `.pth` (a pickle) and does not upload it.
