# Changelog

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
