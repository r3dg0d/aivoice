# Changelog

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
