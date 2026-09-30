# aivoice

Local **real-time AI voice conversion** wrapping [MeanVC2](https://github.com/ASLP-lab/MeanVC2) ([arXiv:2606.09050](https://arxiv.org/abs/2606.09050)), with a searchable voice library and honest RVC → MeanVC2 **profile** migration.

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

Nix: `nix develop` via `flake.nix` (CPU-friendly; CUDA/torch not forced).

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

## Citation

```bibtex
@article{ma2026meanvc2,
  title={MeanVC2: Robust Low-Latency Streaming Zero-Shot Voice Conversion},
  author={Ma, Guobin and Xia, Yuxuan and Jiang, Yuepeng and Guo, Dake and Xie, Hanke and Hu, Jingbin and Wang, Yanbo and Xie, Lei and Zhu, Pengcheng},
  journal={arXiv preprint arXiv:2606.09050},
  year={2026}
}
```

See [STATUS.md](STATUS.md) for honest limitations.
