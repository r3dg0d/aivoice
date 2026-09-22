# Changelog

## 0.1.0 — 2026-09-22

- Initial Linux CLI: `devices`, `live`, `file`, `virtualmic`, `profiles`/`profile`, `benchmark`, `models list|install`.
- Modes: `lowest-latency` | `balanced` | `best-quality` (mapped to MeanVC2 40ms/120ms).
- Consent gate; XDG voice profiles; PipeWire/Pulse/ALSA device listing; virtual mic helpers.
- Model manager with `--yes` ack, HF revision pin hook, vendor git checkout under cache — no silent giant downloads.
- Documents Apache-2.0 LICENSE gap on upstream GitHub; wrapper is Apache-2.0.
