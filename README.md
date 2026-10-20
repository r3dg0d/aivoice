# aivoice

Local **real-time AI voice conversion** wrapping [MeanVC2](https://github.com/ASLP-lab/MeanVC2) ([arXiv:2606.09050](https://arxiv.org/abs/2606.09050)), with a searchable voice library and honest RVC → MeanVC2 **profile** migration.

**New in 0.4 (experimental):** a real-time *speech-to-speech* mode: microphone → VAD →
streaming ASR (Moonshine on CPU, or Parakeet TDT 0.6B v3) → transcript stabilizer → optional
local LLM rewrite → expression tags → Fish Audio Drama 3 / S2-Pro (local server, your remote
server, or the hosted API) → PipeWire virtual mic. Includes a Hugging Face model manager.
See [docs/experimental-realtime.md](docs/experimental-realtime.md).

> Drama 3 is a 35B MoE that needs ≥ 48 GB of GPU memory (FP8). It does **not** fit a 24 GB RTX 4090;
> run S2-Pro locally, or Drama 3 on a bigger GPU / the hosted API. Details: [docs/drama3.md](docs/drama3.md).

**New in 0.5:** **zero-shot voice cloning** (`aivoice clone …`).
- **Engines.** VoxCPM2, X-Voice and VoiceStar sit behind one capability-checked engine layer, with Fish
  and MeanVC2 alongside them.
- **Research fusion.** Reference preparation, silence-aware duration planning, long-form joins and
  identity-guarded enhancement, each kept only where an ablation showed it helps.
- **Evaluation.** A benchmark and evaluation suite.

See [docs/zero-shot-cloning.md](docs/zero-shot-cloning.md).

## Listen to aivoice — Voice Cloning Showcase

All samples are **synthetic** and rights-free:

- The speakers are espeak-ng formant voices that imitate no real person.
- Every file carries an embedded "synthetic speech" label.
- Every file is loudness-matched to −23 LUFS.

The A/B player is at [docs/demo/index.html](docs/demo/index.html) (GitHub Pages from the repository
root); details are in [docs/audio-showcase.md](docs/audio-showcase.md).

> No neural cloning engine could be downloaded where these were generated (no Hugging Face access, no GPU).
> The showcase therefore demonstrates aivoice's own pipeline stages, and `manifest.json` lists the engines
> as *unavailable* rather than shipping placeholder audio.

| What | A | B | Measured |
|---|---|---|---|
| Reference voice | [synth-a](examples/audio/synth-a/reference.wav) | [synth-b](examples/audio/synth-b/reference.wav) | |
| Reference preparation | [noisy upload](examples/audio/synth-a/upload-raw.wav) | [prepared](examples/audio/synth-a/upload-prepared.wav) | GE2E similarity to the clean voice 0.816 → 0.833 |
| Duration planning | [whole-clip rate](examples/audio/synth-a/duration-naive.wav) | [silence-aware](examples/audio/synth-a/duration-silence-aware.wav) | planned 6.17 s vs 4.03 s (natural 5.11 s) |
| Long-form | [hard concatenation](examples/audio/synth-a/longform-naive.wav) | [aivoice joins](examples/audio/synth-a/longform-aivoice.wav) | level jumps and splice clicks removed |
| Enhancement | [naive long-form](examples/audio/synth-a/longform-naive.wav) | [`--enhance auto`](examples/audio/synth-a/longform-naive-enhanced.wav) | clicks 2 → 0 |

## Quick start

```bash
# Search community catalogs → select → download → build MeanVC2 voice → virtual mic
aivoice virtualmic --search "Example Voice" --consent-ack

# Use an already-installed voice (fully offline after install)
aivoice virtualmic --voice "Example Voice" --consent-ack

# Zero-shot from your own reference clip
aivoice voices create "My Voice" --reference voice.wav
aivoice virtualmic --voice "My Voice" --consent-ack
```

Flow:

**SEARCH → SELECT → DOWNLOAD → VALIDATE → IMPORT → MEANVC2 PROFILE → CACHE → USE**

## What “RVC → MeanVC2” actually means

RVC and MeanVC2 are **different architectures**. `aivoice` does **not** convert `.pth` weights into MeanVC2 checkpoints.

MeanVC2 is zero-shot: it conditions on **target reference audio** (WavLM + ECAPA → UTTE). When an RVC package includes legitimate preview/reference speech, `aivoice`:

1. Detects RVC v1/v2 safely (prefer `torch.load(..., weights_only=True)`)
2. Prepares reference audio
3. Stores a reusable MeanVC2 voice profile under `~/.local/share/aivoice/voices/`

If the package has no usable reference audio, `aivoice` can **synthesize one from the RVC model itself** (see below). If that is not possible, import records an incomplete RVC-backed entry and says why, instead of inventing a fake conversion.

### Synthesized references (TTS → RVC)

Many RVC packages ship no demo recording. When a package has a checkpoint but no reference audio, `aivoice` (unless you pass `--no-synthesize` or set `AIVOICE_NO_SYNTH=1`):

1. has a **text-to-speech** engine read a neutral passage (Harvard sentences, or `--synth-text`),
2. runs that speech through the **RVC model** so it speaks in the model's voice,
3. uses the result as the MeanVC2 reference.

The voice is labelled `reference_kind: synthesized` (see `aivoice voices info`), because it is a rendering of the model, not a recording of the person. Quality is bounded by the TTS prosody and the RVC model; a real, clean clip (`--reference`) is always better and always wins.

Engines are optional and discovered automatically (`aivoice doctor` lists them):

| Step | Engine | Enable with |
|------|--------|-------------|
| TTS | any command | `AIVOICE_TTS_CMD='mytts {text_file} {out}'` |
| TTS | [Piper](https://github.com/rhasspy/piper) | `piper` on `PATH` + `AIVOICE_PIPER_MODEL=/path/voice.onnx` |
| TTS | espeak-ng | `espeak-ng` on `PATH` (robotic but dependency-free) |
| RVC | any command | `AIVOICE_RVC_CMD='myrvc {input} {model} {index} {output}'` (`{version}` is `v1`/`v2`) |
| RVC | [`rvc-python`](https://pypi.org/project/rvc-python/) | `pip install rvc-python` (pulls torch; downloads its base models on first use) |

Command templates are split like a shell would but run **without** a shell, so paths with spaces or metacharacters stay single arguments.

> **Security:** an RVC `.pth` is a Python pickle. Synthesizing loads it, which executes whatever code it contains. Only synthesize from models you trust. The CLI prints this before it starts.

```bash
aivoice voices import-rvc model.zip                      # synthesizes if there is no demo audio
aivoice voices import-rvc model.zip --synth-text "Any text you like."
aivoice voices import-rvc model.zip --no-synthesize      # old behaviour: incomplete entry
```

The same applies to `aivoice voices install --search …` and `virtualmic --search …`: a downloaded model with no preview audio now becomes a usable voice. Re-importing a package upgrades an earlier *incomplete* entry instead of failing.

## Live RVC v2 engine and Quickshell widget

Voices with an RVC v1/v2 `.pth` now convert live through a persistent RVC engine (default for those voices); MeanVC2 remains available.
New references are built with Piper TTS through the voice's RVC model (`aivoice voices resynth VOICE`).
A Quickshell control panel lives in [`contrib/quickshell/aivoice`](contrib/quickshell/aivoice/README.md).

## Zero-shot voice cloning (0.5)

```bash
aivoice engines list                                   # what is installed and ready
aivoice clone create "Narrator" --reference narrator.wav --transcript "Exact words." --i-have-rights
aivoice clone synthesize --voice Narrator --text "Hello." -o hello.wav            # engine chosen automatically
aivoice clone synthesize --voice Narrator --text "Hello." -o h.wav --engine voxcpm2 --style "warm, unhurried"
aivoice clone compare --voice Narrator --text "Same text, every engine." --output compare/
aivoice benchmark research --synthetic --consent-ack   # similarity / WER / RTF / VRAM, JSON + CSV
aivoice research list                                  # papers, what aivoice took from each, status
```

| Engine | Kind | Zero-shot conditioning | Highlights | Rate |
|---|---|---|---|---|
| VoxCPM2 (OpenBMB, 2B) | TTS | reference audio, transcript optional | 30 languages, `--style` instructions, streaming, LoRA | 48 kHz |
| X-Voice (0.4B) | TTS | transcript-free | 30 languages via IPA, code-switching, duration | 24 kHz |
| VoiceStar (840M) | TTS | reference + transcript | duration control, long-form | 16 kHz |
| Fish S2-Pro / Drama 3 | TTS | reference + transcript | existing server / hosted API | 44.1 kHz |
| MeanVC2 | VC | reference audio | streaming conversion (existing) | 16 kHz |

**How it works**

- **Unsupported options are refused.** An engine never silently ignores an option it cannot honour
  (`--allow-unsupported` drops it with a warning).
- **Each engine runs in its own virtualenv** through an isolated worker
  ([docs/model-backends.md](docs/model-backends.md)).
- **Pipeline stages.** Every result runs through text normalization, routing, a prepared reference, the
  duration plan, long-form joins, optional enhancement and evaluation
  ([docs/research-fusion.md](docs/research-fusion.md)).
- **Labelled output.** Each output WAV is labelled as synthetic and gets a JSON sidecar.
- **Profiles.** Existing `voices` and `voice` profiles are untouched; `aivoice clone migrate` copies them
  into the new format.

Measured ablations, on rights-free synthetic speakers ([docs/research/experiments](docs/research/experiments/)):

| Change | Baseline → aivoice | Decision |
|---|---|---|
| Silence-aware duration plan (Cross-Lingual F5-TTS 2 → VoiceStar) | mean absolute duration error 0.82 (VoiceStar default) → 0.15 | retained |
| Reference denoise when noisy | +0.036 GE2E cosine (95% CI 0.020–0.053) | retained as `auto` |
| Long-form joins | loudness spread 4.0 → 0.7 LU; clicks 3 → 0.5 | retained |
| Engine comparisons (A, B, D, E, H) | not run: weights unreachable, no GPU | — |

## Experimental real-time mode (0.4)

```bash
pip install -e '.[experimental]' && pip install --no-deps silero-vad
aivoice experimental enable
aivoice models download moonshine-small --yes            # CPU streaming ASR
aivoice experimental tts use fish-server --url http://127.0.0.1:8080 --model s2-pro
aivoice voice add "Me" --audio me.wav --text "exact words in me.wav" --i-have-rights && aivoice voice use "Me"
aivoice experimental start --consent-ack                 # select "aivoice-virtual-mic" in Discord/OBS/VRChat
```

| Command | Purpose |
|---------|---------|
| `aivoice experimental enable\|disable\|start\|stop\|status\|latency\|benchmark\|plan` | Run, monitor and measure the pipeline |
| `aivoice experimental control toggle-mute\|interrupt\|ptt-down\|ptt-up\|emergency-stop` | Live controls (bind to hotkeys) |
| `aivoice experimental asr list\|use NAME\|auto\|benchmark` | Moonshine tiny/small/medium, Parakeet, auto selection from measurements |
| `aivoice experimental tts use\|health\|say` | fish-speech server (Drama 3 / S2-Pro), hosted API, local fallback |
| `aivoice experimental preset list\|use\|save` | ultra-low-latency, balanced, expressive, character, accuracy, custom |
| `aivoice experimental remote allow\|revoke\|list` | Opt in to remote TTS (off by default) |
| `aivoice llm enable\|disable\|model\|style\|prompt\|character\|intensity\|test` | Optional llama.cpp rewriting |
| `aivoice voice list\|add\|add-id\|add-hosted\|use\|preview\|remove\|export\|import` | TTS voice profiles (local voice, local voice id, hosted Fish id) |
| `aivoice models search\|info\|download\|installed\|known\|load\|verify\|update\|remove\|cache\|cleanup\|import\|default` | Hugging Face model manager |
| `aivoice hardware`, `aivoice credentials status\|set` | Hardware/runtime probe; API keys via env or system keyring |

Docs: [experimental-realtime](docs/experimental-realtime.md) · [moonshine](docs/moonshine.md) ·
[parakeet](docs/parakeet.md) · [drama3](docs/drama3.md) ·
[huggingface-model-manager](docs/huggingface-model-manager.md) · [voice-profiles](docs/voice-profiles.md) ·
[llm-transformation](docs/llm-transformation.md) · [latency-optimization](docs/latency-optimization.md) ·
[nixos-pipewire](docs/nixos-pipewire.md) · [benchmarks](docs/benchmarks.md) · [troubleshooting](docs/troubleshooting.md)

Cloning docs: [architecture](docs/architecture.md) · [zero-shot-cloning](docs/zero-shot-cloning.md) ·
[model-backends](docs/model-backends.md) · [speaker-conditioning](docs/speaker-conditioning.md) ·
[research-fusion](docs/research-fusion.md) · [multilingual-tts](docs/multilingual-tts.md) ·
[voice-style-control](docs/voice-style-control.md) · [audio-showcase](docs/audio-showcase.md) ·
[research review](docs/research/research-review.md) · [methodology](docs/research/methodology.md)

Privacy: local by default. Remote TTS destinations need `experimental remote allow`; only text
(and the reference audio of local voice profiles) is sent, never microphone audio. No telemetry.

## Research & consent

For **research, VFX, avatars, filmmaking, consenting demos, and disclosed synthetic media**.

- Consent gate: `--consent-ack` or `AIVOICE_CONSENT_ACK=1`
- No mic/converted audio upload; search queries go only to the selected catalog provider
- Use voice models only with necessary rights; do not use generated audio deceptively

## Install

```bash
pip install -e ".[dev]"
# optional: pip install -e ".[audio,hf]"
aivoice --help
aivoice doctor
```

Nix: `nix develop` via `flake.nix` (CPU-friendly; CUDA/torch not forced). `nix build .#aivoice-experimental`
adds the experimental mode's CPU dependencies; an optional NixOS module provides a persistent virtual mic
([docs/nixos-pipewire.md](docs/nixos-pipewire.md)).

MeanVC2 weights are **not** in this repo:

```bash
aivoice models install meanvc2 --yes
```

## CLI

| Command | Purpose |
|---------|---------|
| `aivoice search "…"` | Search provider (network) |
| `aivoice virtualmic --search "…"` | Search → install → virtual mic |
| `aivoice virtualmic --voice NAME` | Offline installed voice |
| `aivoice voices list\|info\|remove\|rename\|verify` | Voice library |
| `aivoice voices create NAME --reference wav` | Zero-shot profile |
| `aivoice voices import-rvc model.zip` | RVC import → MeanVC2 profile (`--no-synthesize`, `--synth-text`) |
| `aivoice voices install --search "…"` | Install without starting mic |
| `aivoice providers list\|info` | Catalog capabilities |
| `aivoice models list\|install` | MeanVC2 **base** models |
| `aivoice doctor` | Environment checks |
| `aivoice devices` / `live` / `file` | Devices & conversion |
| `aivoice clone create\|list\|inspect\|preview\|synthesize\|compare\|delete\|migrate\|attest\|adapt` | Zero-shot cloning (0.5) |
| `aivoice engines list\|info\|capabilities\|benchmark` | Cloning / conversion engines |
| `aivoice benchmark research` | Similarity, WER/CER, latency, memory across engines |
| `aivoice research list\|citations\|experiments` | Papers, status, ablations |

### Modes

| Mode | MeanVC2 | Intent |
|------|---------|--------|
| `lowest-latency` | 40ms | Minimum chunk |
| `balanced` | 40ms | Default |
| `best-quality` | 120ms | Higher quality |

### Providers

`voice-models` talks to [voice-models.com](https://voice-models.com/) using the site’s own `fetch_data.php` search AJAX (no documented public REST API). Downloads follow third-party links (Hugging Face preferred). Google Drive / Mega require manual download + `voices import-rvc`. Rate-limited (~1 req/s). Respect robots.txt / ToS; no CAPTCHA bypass.

## License

| Layer | License |
|-------|---------|
| **Our wrapper** | **Apache-2.0** — [LICENSE](LICENSE) |
| **MeanVC2 upstream** | README claims Apache-2.0; see [NOTICE](NOTICE) |
| **HF weights** | Apache-2.0 claimed on model card |
| **VoxCPM2** (optional, own venv) | Apache-2.0 code and weights (official README) |
| **VoiceStar** (optional) | code MIT, weights CC-BY-4.0 |
| **X-Voice** (optional) | code MIT; check the weights' model card |
| **Resemblyzer GE2E weights** (evaluation) | Apache-2.0 |

## Citation

```bibtex
@article{ma2026meanvc2,
  title={MeanVC2: Robust Low-Latency Streaming Zero-Shot Voice Conversion},
  author={Ma, Guobin and Xia, Yuxuan and Jiang, Yuepeng and Guo, Dake and Xie, Hanke and Hu, Jingbin and Wang, Yanbo and Xie, Lei and Zhu, Pengcheng},
  journal={arXiv preprint arXiv:2606.09050},
  year={2026}
}
```

Zero-shot cloning builds on VoxCPM2, X-Voice, VoiceStar, Cross-Lingual F5-TTS 2, PFluxTTS, FC-TTS, FlexiVoice and
Confucius4-TTS. BibTeX for all of them: [docs/research/citations.bib](docs/research/citations.bib)
(`aivoice research citations`). What each one contributed, and what could not be verified, is in
[docs/research/research-review.md](docs/research/research-review.md).

See [STATUS.md](STATUS.md) for honest limitations.
