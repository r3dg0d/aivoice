# STATUS — aivoice 0.3.0

Honest limitations (as of 2026-10-02 PT).

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

## Privacy

Search queries go to the selected provider. Microphone audio, converted speech, and reference clips are **not** uploaded by aivoice. Synthesizing a reference loads the local RVC `.pth` (a pickle) and does not upload it.
