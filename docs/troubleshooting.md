# Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `experimental mode is disabled` | `aivoice experimental enable` |
| `no ASR runtime installed` | `pip install 'aivoice[moonshine]'` (CPU) or `nemo_toolkit[asr]` + CUDA torch |
| `moonshine-small is not registered locally` | `aivoice models download moonshine-small --yes` (or set `asr.allow_model_download true`) |
| `no benchmark results for this machine yet` | auto-selection needs measurements: `aivoice experimental asr benchmark` |
| Parakeet partials lag | expected: chunked re-decoding, not streaming. Use Moonshine, or raise the adapter interval |
| `fish-speech server ... is not answering` | start `tools/api_server.py` (docs/drama3.md); `aivoice experimental tts health` |
| `is not on this machine ... remote allow` | remote TTS is opt-in: `aivoice experimental remote allow URL` |
| `No Fish Audio API key` | `export FISH_API_KEY=...` or `aivoice credentials set fish-audio` (needs `keyring`) |
| `hosted Fish Audio id; it only works with tts.backend=fish-api` | hosted and local ids are different systems (docs/voice-profiles.md) |
| CUDA out of memory | `aivoice experimental plan`; Drama 3 needs >= 48 GB; use S2-Pro locally, Moonshine on CPU, LLM on CPU |
| `refusing to store secret-like key` | secrets go in env/keyring, never settings.json |
| `is a gated repository` | accept the license on huggingface.co, then `hf auth login` |
| `Re-run with --yes to confirm` | download is above `hf.confirm_over_gib`; check size first with `--dry-run` |
| download interrupted | re-run the same command: it resumes from huggingface_hub's partial files |
| `verify` reports sha256 mismatch | `aivoice models remove NAME` then download again |
| no `aivoice-virtual-mic` in apps | `pactl list short sources`; pipewire-pulse must be running; `aivoice virtualmic --setup-only` |
| your voice echoes back | you are monitoring on speakers; use headphones or `audio.monitor false` |
| `input ... is aivoice's own virtual mic` | pick the real microphone (`audio.input_device`) |
| crackles / underruns in `status` | raise `audio.prebuffer_ms` (80-120), lower system load, check PipeWire quantum |
| robotic voice | you are on the espeak-ng fallback: the Fish server failed before first audio (`status` errors) |
| LLM output ignored | `llm test "..."` shows `fallback_reason` (timeout, malformed, fact guard) |
| Silero not used | `pip install --no-deps silero-vad onnxruntime`, or `AIVOICE_SILERO_ONNX=/path/silero_vad.onnx` |
| `models install meanvc2` HF error on new huggingface_hub | fixed in 0.4.0 (`local_dir_use_symlinks` was removed upstream) |

Logs and state: `$XDG_RUNTIME_DIR/aivoice/engine.json` (live status),
`~/.config/aivoice/settings.json`, `~/.local/share/aivoice/models/registry.json`,
`~/.cache/aivoice/benchmarks/`.
