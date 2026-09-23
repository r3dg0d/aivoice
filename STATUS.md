# STATUS — aivoice 0.2.0

Honest limitations (as of 2026-09-23 PT).

## What works

- Full CLI: search, voices library, providers, doctor, devices, models, live/file/virtualmic
- voice-models.com search via `fetch_data.php` (site AJAX; not a documented public API)
- HF direct downloads when catalog exposes them; cache under `~/.cache/aivoice/downloads/`
- RVC package inspect + MeanVC2 profile from reference/preview audio
- Offline use of installed voices (`virtualmic --voice`)
- Unit tests (mocked/fixture; no live hammering of catalogs in CI)

## Requires install / hardware

| Capability | Requirement |
|------------|-------------|
| MeanVC2 convert / live | `models install meanvc2 --yes` + torch (CUDA preferred) + upstream deps |
| Live mic / virtualmic stream | Vendor `runtime/run_rt.py` + audio stack |
| Audio prep polish | `ffmpeg` (optional; falls back to copy) |
| Google Drive / Mega models | Manual download + `voices import-rvc` |

## Known gaps

1. **No direct RVC↔MeanVC2 weight conversion** — by design; docs are explicit.
2. **RVC inference fallback** is metadata-level only in 0.2.0 (incomplete without reference); MeanVC2 remains preferred.
3. **Speaker fine-tune** (`voices optimize`) not implemented — zero-shot profiles are the default.
4. **Quickshell widget** not bundled in this repo.
5. **Arrow-key TUI** is numbered/TTY select (not full Textual browser yet).
6. Upstream MeanVC2 LICENSE file gap still documented in NOTICE.
7. MeanVC2 not installed on a fresh machine until `models install`.

## Privacy

Search queries go to the selected provider. Microphone audio, converted speech, and reference clips are **not** uploaded by aivoice.
