# Architecture

## Audit of the starting point (0.4.0)

Before this work, the repository had two independent paths:

- **Voice conversion.** MeanVC2 runs behind `aivoice live | file | virtualmic`. It uses the voices library
  (`voices/`) with a v1 profile format and the RVC → reference synthesis flow.
- **Experimental real-time speech-to-speech.** This path is VAD → ASR → LLM → expression compiler →
  Fish TTS → PipeWire. Its TTS backends are selected by `tts/__init__.py:create_tts`.

There was no zero-shot TTS cloning engine other than Fish. No component could say which options it
supports, and there was no speaker-similarity or intelligibility evaluation. Both paths stay unchanged;
everything below is additive.

## Layers

```
cli_clone.py  (clone / engines / research / benchmark research)
   │
fusion/pipeline.py ── normalize → route → prepared reference → duration plan → long-form → enhance → evaluate
   │                         │
fusion/router.py      research/*  (reference_quality, duration, phonetics, longform, speaker_identity,
   │                              flow_matching, adaptation, catalog)
engines/registry.py
   │
engines/base.py  TTSEngine | StreamingTTSEngine | VoiceConversionEngine  + EngineCapabilities
   │
   ├── research_engines.py  VoxCPM2, X-Voice (CLI), VoiceStar   ── worker.py ⇄ worker_main.py (own venv)
   └── builtin_engines.py   MeanVC2 (VC), Fish server/API, espeak
evaluation/  metrics, benchmark, experiments, synthetic_data
speakers.py  v2 profiles (+ copy-only v1 migration)
audio/loudness.py, audio/enhancement.py
```

## Key decisions

- **TTS and VC are separate protocols.** A VC engine (MeanVC2) never receives text, and a TTS engine never
  receives source audio. `clone compare` reports them in separate tables.
- **Capabilities are data.** `EngineCapabilities` lists the following:
  - tasks, zero-shot support, whether a transcript is needed, cross-lingual support
  - style prompt / style reference, duration control, streaming, adaptation
  - sample rate, devices and estimated VRAM

  `validate_options` compares a request with these fields, so unsupported options fail loudly.
- **Engine isolation.** VoxCPM2, VoiceStar and X-Voice pin incompatible torch/transformers versions.
  Each runs in its own interpreter (`AIVOICE_<ENGINE>_PYTHON` or `engines.<engine>.python`). The
  JSON-lines worker (`worker_main.py`) imports nothing from aivoice.
- **Routing uses only local measurements.**
  - The router first drops engines that lack a needed capability, are not installed, do not fit in free
    VRAM, or are remote without consent.
  - It then ranks the rest by measurements recorded on this machine (`fusion/router.py`).
  - Without measurements it uses a fixed, documented order. Paper numbers are never used for ranking.
- **Labelled outputs.** Every WAV written by the clone path carries a RIFF `ICMT` comment saying it is
  synthetic speech, plus a JSON sidecar. The sidecar records the engine, options, reference hash, plan and
  metrics.
- **Consent.** Creating a profile requires `--i-have-rights`. Publishing samples requires
  `--publishable`. Benchmarks on a user dataset require `consent.json`.

## Real-time paths

- **Path A, VC.** MeanVC2 is unchanged. Clone profiles store a prepared reference that MeanVC2 can
  use as its target clip.
- **Path B, ASR → TTS.** `tts use engine:<name> --voice <profile>` puts any registered engine behind the
  real-time TTS interface (`engines/realtime.py`).
  - Streaming engines (VoxCPM2 `generate_streaming`) stream.
  - Other engines produce one chunk per utterance, and that limitation is reported.
