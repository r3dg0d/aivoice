# Changelog

## 0.7.0 — 2026-10-21 — Paper reproductions

### Added
- **Reproduction framework** (`aivoice.research_reproductions.common`):
  - provenance labels (source status, weights status, architecture confidence, PAPER-REPORTED / MEASURED);
  - streaming-exact layers (causal conv / transposed conv / windowed MHSA with KV cache and RoPE offset, per-layer
    lookahead);
  - self-describing checkpoints (sha256, JSON sidecar, config fingerprint, environment, `weights_only` loading,
    named shape errors);
  - a generic trainer: BF16/FP16/TF32, grad accumulation and clipping, compile, fused AdamW, DDP, profiling;
    **exact resume** including RNG and the consumed sampler position; OOM reports that change nothing;
    JSONL/CSV/TensorBoard metrics, W&B opt-in;
  - status/control files for a background service;
  - manifests for LibriTTS / VCTK / LibriSpeech / custom with filters, a resampling cache and versioned feature
    caches;
  - a streaming harness (causality, state reset, chunk invariance, long session);
  - a mandatory tiny-overfit gate.
- **RT-VC** (`rtvc`, PARTIAL_OFFICIAL_CODE): official layout re-implemented with parameter-shape parity, causal
  front-end, exact streaming state, and the pitch / EMA / vocoder training pipeline.
- **Zero-VC** (`zerovc-repro`): content variants A (w2v-BERT 2.0 distillation) and B (HuBERT k-means, aivoice),
  WavLM-L7 attention pooling, causal HiFi-GAN with zero future frames, and a training-only McAdams / external
  anonymiser.
- **TVTSyn** (`tvtsyn-repro`, `tvtsyn-aivoice`): content encoder, factorised VQ, Global Timbre Memory + gate +
  Slerp verified against the official VOSSA code, and a decoder with pitch/energy predictors and KV caches.
- **StreamVoiceAnon+** (`streamvoiceanon-plus`): SEP tokens + emotion2vec distillation around the official
  dual-AR model.
- **VOSSA** (`vossa`): official engine (code + checkpoints, CC BY-NC 4.0) in a worker.
- **CLI `aivoice reproduce`** (list, info, status, prepare, train, resume, evaluate, export, verify, paper-map,
  control, cost). `aivoice models info` shows provenance for reproductions and checkpoints. QuickShell gets a
  **Research** tab.
- Docs: `docs/reproductions/` (scorecard, per-paper guides), SPEC.md per reproduction, catalog / research review /
  citations updated.

### Changed
- `rtvc`, `zerovc`, `tvtsyn` are no longer catalogue-only. They are reproduction engines that report
  MISSING_WEIGHTS until a trained checkpoint is installed (`zerovc` / `tvtsyn` remain as aliases).

### Fixed (found by the new tests)
- DataLoader iterators drew their seed from the global RNG, which broke exact resume.
- A resume landing on an epoch boundary was rejected as "fewer items than one batch".

## 0.6.0 — 2026-10-21 — Multi-Engine Streaming Voice Conversion

### Added — three live systems on one virtual mic
- **Streaming VC layer** (`aivoice.vc`):
  - `StreamingVCEngine` interface: `capabilities / status / load / prepare_reference / start_stream /
    process_chunk / flush / reset_stream / unload`, with `VCCapabilities` (task, streaming, zero-shot,
    reference / model-file requirements, verified cross-lingual flag, chunk, lookahead, algorithmic latency, devices,
    license, pinned code, evidence class).
  - Real engine states: READY, INSTALLED, MISSING_WEIGHTS, MISSING_RUNTIME, UNSUPPORTED_HARDWARE, EXPERIMENTAL,
    RESEARCH_ONLY, BLOCKED_LICENSE, BLOCKED_CODE (`aivoice engines list`, `aivoice vc engines`).
  - Resident worker models (`vc/stream_worker.py`, one interpreter per engine, loaded once, JSON-lines chunks;
    0.13 ms mean IPC overhead per chunk measured).
- **Engines:** `xvc` (X-VC, official code @ 49df8c5, incremental chunkwise streaming, sample-identical to upstream),
  `meanvc2` (resident upstream `VCRunner`), `rvc` (the c7d5db8 `RvcRealtime` block processor behind the shared
  pipeline), experimental `stylestream` and `streamvoiceanon`, and `dsp` / `passthrough`. Catalogue-only, never
  runnable: `rtvc` (BLOCKED_CODE), `zerovc` and `tvtsyn` (RESEARCH_ONLY).
- **Shared live pipeline** (`vc/live.py`): capture → gain → optional pre-stage → polyphase resampling → engine →
  optional post-modulation → limiter → `aivoice_sink`; fallback chain on crash / timeout / error (default DSP, then
  passthrough) with primary restart; drift control (`Player.trim_quiet`); status file for widgets and live control
  file (`aivoice vc status|stop|control|set`).
- **Router:** `--engine auto --priority balanced|latency|gaming|similarity|naturalness|streaming-quality|low-vram|cpu`
  ranks on local `benchmark vc` measurements only; without them the previous order (RVC → MeanVC2 → X-VC) is used and
  labelled unmeasured.
- **Voice modulation** (`aivoice.modulation`, `aivoice modulate`): streaming DSP engine with independent pitch
  (excitation phase vocoder, identity phase locking) and formants (cepstral true-envelope warp), intonation range,
  body/resonance/brightness/tilt, calibrated breathiness, roughness, noise suppression, de-esser, compressor,
  limiter, delay-matched bypass; 23 presets incl. an identity-preserving masculine mode (pitch + formants);
  `--identity FLOOR`, `modulate calibrate`, user presets.
- **Benchmark:** `aivoice benchmark vc` — same sources/targets through every runnable engine on the live path:
  chunk p50/p95/p99, RTF, declared vs measured delay, CPU, RSS, worker VRAM, GE2E target similarity, source leakage,
  WER/CER (with an ASR), clicks, F0 jumps; loudness-matched (−23 LUFS) listening set + manifest, JSON, CSV.
- **Experiments** L, M, N, O, R measured (identity retention, masculinisation ablation, formant contribution, 30-minute
  stability, chunk smoothing); I, J, K, P, Q defined and reported as not run. `aivoice research long-session`.
- **Install:** `aivoice models install xvc --yes` (pinned checkout + Hugging Face weights + ModelScope ERes2Net).
- **QuickShell widget:** VOICE CONVERSION / VOICE MODULATION / SPEECH-TO-SPEECH modes, engine buttons from real
  status, experimental section, modulation sliders applied live, A/B bypass, save preset, live latency/RTF/GPU/VRAM/
  underrun metrics; tested offscreen against stub Quickshell modules with real CLI output.
- **Research catalogue:** X-VC, RT-VC, Zero-VC, TVTSyn, StyleStream, StreamVoiceAnon(+), RVC, Laroche-Dolson phase
  locking, true envelope, voice-gender acoustics (unverified); evidence classes.
- **Docs:** streaming-vc, voice-modulation, masculine-voice, xvc, rvc-live, engine-comparison, realtime-architecture,
  quickshell.

### Fixed
- MeanVC2 file conversion now passes `--device` to `infer_e2e.py` (it ran on the CPU).
- `rvc/rt_worker.py` block processing moved into `RvcRealtime` (same behaviour; reusable by the shared pipeline).
- QuickShell `run.sh` failed on machines without `~/.cache`.
- `aivoice experimental status --json` now reports which ASR runtimes are installed.

## Live RVC engine and Quickshell widget (c7d5db8, 02bd6fa, 81ffa29)

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
