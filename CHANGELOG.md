# Changelog

## Unreleased

### Added
- **Live RVC v1/v2 engine** (`src/aivoice/rvc/rt_worker.py`): the model stays loaded on the GPU and streams mic blocks to `aivoice-virtual-mic`. It is the default engine when a voice has an RVC `.pth`; MeanVC2 remains selectable. Pitch and index-rate controls.
- `voices resynth`: rebuild a voice's reference sample. New references use Piper TTS run through the voice's RVC model instead of espeak-ng.
- Quickshell widget in `contrib/quickshell/aivoice/` (engine/pitch/index sliders, input picker with level meter, doctor fixes) and runtime helpers in `scripts/aivoice-runtime/` (CUDA llama.cpp build, Python 3.10 RVC venv, `rvc-infer` bridge).

### Fixed
- PipeWire pinning: converted output is routed to `aivoice_sink` (no longer leaks to the default speakers) and input is pinned to the chosen source.
- The chosen input source is unmuted automatically on start.
- MeanVC2 now receives `--target-spk` and `--device` as CLI flags (it was running on CPU with its bundled test clip).
- Voice-name parsing no longer truncates titles at the first dash (e.g. "E-Girl" became "E").

## 0.5.0 — 2026-10-20

### Added — zero-shot cloning and research fusion
- **Engine layer** (`aivoice.engines`):
  - `TTSEngine` / `StreamingTTSEngine` / `VoiceConversionEngine` protocols with `EngineCapabilities`.
    Unsupported options raise `UnsupportedOption` (`--allow-unsupported` drops them with a warning).
  - New engines: VoxCPM2 (voxcpm 2.0.3), VoiceStar (voicestar 0.1.0) and X-Voice Stage 2 (x-voice 0.1.1 CLI).
    Each runs in its own interpreter through an isolated JSON-lines worker.
  - Adapters for MeanVC2, the Fish server/API and espeak-ng.
- **Clone CLI:**
  - `aivoice clone create|list|inspect|preview|synthesize|compare|delete|migrate|attest|adapt`
  - `aivoice engines list|info|capabilities|benchmark`
  - `aivoice research list|citations|experiments`
  - `aivoice benchmark research`
- **Speaker profiles v2:**
  - Stored under `~/.local/share/aivoice/speakers/`, with analysis, a prepared reference, consent and
    provenance.
  - `clone migrate` copies v1 profiles in and never modifies the originals.
- **Research modules** (`aivoice.research`):
  - Reference analysis and identity-guarded preparation, with auto-denoise.
  - A GE2E speaker encoder that reproduces Resemblyzer exactly.
  - Silence-aware duration planning (Cross-Lingual F5-TTS 2, applied to VoiceStar and X-Voice).
  - Text normalization, language detection, X-Voice code-switch tags and IPA.
  - Long-form segmentation and joins.
  - Experimental vector-field fusion with strict compatibility checks.
  - Adaptation dataset validation and VoxCPM LoRA manifests (no training is started).
  - A paper catalog with statuses and BibTeX.
- **Fusion pipeline and router:**
  - The router filters engines by capability, availability, VRAM and remote consent, then ranks them by
    this machine's measurements, or by a fixed order.
  - Outputs are labelled as synthetic (RIFF `ICMT`) and get a JSON sidecar.
- **Audio** (`aivoice.audio.loudness`, `aivoice.audio.enhancement`):
  - BS.1770-4 loudness.
  - A robust click detector and repair, crossfade joins, loudness matching, and `--enhance
    off|auto|high-fidelity` with an identity guard.
- **Evaluation:**
  - Speaker similarity, WER/CER, and acoustic metrics.
  - A research benchmark (JSON + CSV, `consent.json` gating), and rights-free synthetic espeak-ng speakers.
  - Ablation experiments A–H with bootstrap CIs and retain/reject decisions. C, F and G were measured;
    the rest are reported as not run, with the reason.
- **Real-time Path B:** `experimental tts use engine:<name> --voice <profile>` puts any engine behind the
  real-time TTS interface.
- **Showcase:**
  - Rights-free audio in `examples/audio/`, plus the generate and validate scripts.
  - An A/B player at `docs/demo/index.html`.
- **Docs:** architecture, zero-shot cloning, model backends, speaker conditioning, research fusion,
  multilingual, style control, audio showcase, `research/` (review, methodology, citations,
  experiments).
- **Packaging:** a `research` extra (torch, librosa, resemblyzer).
- **CI:** lint, build and smoke steps.

### Changed
- `aivoice benchmark` is now a group. Running it with no subcommand still does the previous buffer
  benchmark, unchanged.

## 0.4.0 — 2026-10-18

### Added — experimental real-time mode
- `aivoice experimental …`: microphone → VAD → streaming ASR → transcript stabilizer → optional LLM →
  expression compiler → Fish Drama 3 / S2-Pro / local TTS → PipeWire virtual mic, with per-utterance
  latency timelines, speculative synthesis, barge-in policies, push-to-talk, live controls, presets,
  remote-processing consent, and a benchmark suite (`experimental benchmark`, JSON + text reports).
- ASR backends behind `ASRBackend`: Moonshine Voice tiny/small/medium (native streaming, CPU) and
  NVIDIA Parakeet TDT 0.6B v3 (NeMo; chunked re-decoding adapter, labelled as such); automatic
  selection from cached measurements keyed by hardware/runtime fingerprint.
- Silero VAD run from its bundled ONNX file with onnxruntime (no torch); energy VAD fallback.
- Fish backends: fish-speech `api_server` compatible client (local or consented remote), hosted Fish
  Audio API client; streaming WAV decoding; local espeak-ng/Piper/command fallback that is never remote.
- Voice profiles (`aivoice voice …`): local voice, local voice id (server reference), hosted Fish id;
  authorization attestation required.
- `SpeechIntelligenceEngine` (`aivoice llm …`): verbatim/cleanup/natural/expressive/character/custom/
  hybrid modes over llama.cpp GGUF models, JSON output contract, timeout fallback, fact guard.
- Hugging Face model manager (`aivoice models search|info|download|installed|known|load|verify|update|
  remove|cache|cleanup|import|default`): commit-pinned resumable downloads with confirmation, disk and
  quota checks, SHA-256 verification, gated-repo guidance, offline registry, safe cache removal.
- Resource scheduler (VRAM/RAM budgets, LRU eviction, OOM recovery), `aivoice hardware`,
  `aivoice experimental plan`, `aivoice credentials` (env / system keyring; never settings.json).
- Virtual mic is now idempotent (reuses existing sink/source), tracks and cleans up only its own modules,
  offers a loopback monitor, and refuses feedback-prone inputs; `pacat`/`parec` drivers target named devices.
- Nix: `packages.aivoice-experimental`, a fuller dev shell, and an opt-in `nixosModules.virtual-mic`.
- Docs under `docs/`.

### Fixed
- `models install meanvc2` failed with huggingface_hub ≥ 1.0 (`local_dir_use_symlinks` was removed upstream).
- WAV reading/writing no longer hard-requires `soundfile` (stdlib fallback for PCM WAV).
- `flake.nix` reported version 0.2.0.

## 0.3.0 — 2026-09-30

### Added
- **Synthesized references (TTS → RVC).** An RVC package with a checkpoint but no
  demo/reference audio no longer stops at an incomplete entry: a TTS engine reads a
  neutral passage, the RVC model converts it, and the result becomes the MeanVC2
  reference. Voices built this way are labelled `reference_kind: synthesized`
  (`voices info`). Pluggable engines (`AIVOICE_TTS_CMD`, Piper, espeak-ng;
  `AIVOICE_RVC_CMD`, `rvc-python`), command templates run without a shell.
  Opt out with `--no-synthesize` / `AIVOICE_NO_SYNTH=1`; `--synth-text` sets the text.
- `aivoice doctor` reports which synthesis engines are available.

### Fixed
- **Reference clips were truncated at their first pause.** `prepare_reference` used
  `silenceremove … stop_periods=1`, which cuts the audio at the first silence of 0.4 s
  or more: an 8 s reference became 0.8 s (just the first phrase) before MeanVC2 saw it.
  It now trims only leading and trailing silence and keeps pauses inside the speech.
  Voices created earlier from clips with pauses are worth re-creating.
- Re-importing a package for an earlier *incomplete* voice raised `FileExistsError`; it
  now replaces the stale entry.
- A failed first import left `_aivoice_placeholder.wav` in the package, and the next
  import treated that silent 0.1 s file as a real reference recording and produced a
  "ready" voice from silence. aivoice-generated files are now ignored when looking for
  reference audio.

### Security
- Documented (and printed before synthesis) that loading an RVC `.pth` executes pickle code.

## 0.2.0 — 2026-09-23

- Voice provider architecture (`VoiceProvider`) with `voice-models` catalog client.
- `aivoice search`, `voices install`, `virtualmic --search` (SEARCH→SELECT→DOWNLOAD→ADAPT→USE).
- Voice library under XDG (`voices list|info|create|import-rvc|remove|rename|verify`).
- Safe downloader (resume, checksums, atomic finalize) + zip-slip-safe extraction.
- Honest RVC v1/v2 inspection (`weights_only` when available) and MeanVC2 **reference-audio** adaptation (not weight conversion).
- `aivoice doctor`, `providers list|info`, privacy note on search.
- Unit tests for parser fixtures, archive safety, RVC detect, voices, doctor.

## 0.1.0 — 2026-09-22

- Initial Linux CLI: `devices`, `live`, `file`, `virtualmic`, `profiles`/`profile`, `benchmark`, `models list|install`.
- Modes: `lowest-latency` | `balanced` | `best-quality` (mapped to MeanVC2 40ms/120ms).
- Consent gate; XDG voice profiles; PipeWire/Pulse/ALSA device listing; virtual mic helpers.
- Model manager with `--yes` ack, HF revision pin hook, vendor git checkout under cache — no silent giant downloads.
- Documents Apache-2.0 LICENSE gap on upstream GitHub; wrapper is Apache-2.0.
