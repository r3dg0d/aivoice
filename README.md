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

If no usable reference audio exists, import fails honestly (or records an incomplete RVC-backed entry) instead of inventing a fake conversion.

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
| `aivoice voices import-rvc model.zip` | RVC import → MeanVC2 profile |
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
