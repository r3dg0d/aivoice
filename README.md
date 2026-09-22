# aivoice

Local **real-time AI voice changer** wrapping [MeanVC2](https://github.com/ASLP-lab/MeanVC2) ([arXiv:2606.09050](https://arxiv.org/abs/2606.09050), weights on [Hugging Face](https://huggingface.co/ASLP-lab/MeanVC2)).

This repository is a **thin Linux CLI + PipeWire helpers** around upstream. We prefer subprocess/API integration and an optional vendor checkout under XDG cache after `models install` — we do **not** copy their entire tree into this git repo.

## Research & consent framing

For **research, VFX, avatars, filmmaking, consenting demos, and disclosed synthetic media**.

- **No anonymity claims.**
- Consent gate: `--consent-ack` or `AIVOICE_CONSENT_ACK=1`.
- Only convert voices you own or have consent to process; disclose synthetic audio when required.
- Misuse for impersonation, fraud, or harassment is prohibited.

## License (important)

| Layer | License |
|-------|---------|
| **Our wrapper** (`aivoice`) | **Apache-2.0** — see [LICENSE](LICENSE) |
| **MeanVC2 upstream** | README claims **Apache-2.0**, but **GitHub root LICENSE file was missing** (API `license: null`) at packaging time — see [NOTICE](NOTICE). We attribute upstream and **do not relicense** their code. |
| **HF weights** | Apache-2.0 claimed on model card — retain notices |

## Install

```bash
pip install -e ".[dev]"
# optional: pip install -e ".[audio,hf]"
aivoice --help
aivoice devices
aivoice models list
```

Nix: `nix develop` via `flake.nix` (CPU-friendly; CUDA/torch not forced).

Weights are **not** in this repo:

```bash
aivoice models install meanvc2 --yes   # git vendor + HF pull with license ack
```

CUDA preferred for realtime; CPU fallback is documented (RTF may exceed 1).

## Usage

```bash
# Devices / models / profiles
aivoice devices
aivoice models list
aivoice profiles list
aivoice profile create --name alice --reference voice.wav --mode balanced

# Live mic (spawns MeanVC2 runtime when installed)
aivoice live --reference voice.wav --consent-ack --mode lowest-latency --device cuda

# File conversion
aivoice file input.wav --reference voice.wav --consent-ack -o out.wav --mode best-quality

# Virtual mic (PipeWire/Pulse null sink + remap source)
aivoice virtualmic --reference voice.wav --consent-ack --setup-only
aivoice virtualmic --reference voice.wav --consent-ack

# Benchmark buffer overhead
aivoice benchmark --frames 50 --consent-ack
```

### Modes

| Mode | MeanVC2 model | Intent |
|------|---------------|--------|
| `lowest-latency` | 40ms | Minimum chunk / steps |
| `balanced` | 40ms | Default |
| `best-quality` | 120ms | Higher quality, more latency |

Expose/override: `--chunk-ms`, `--buffer-chunks`, `--device`, `--mode`.

### Metrics

End-to-end latency, RTF, sample rate, buffer depth, underruns, dropped chunks, CUDA util (when torch present).

### Audio stacks

PipeWire / PipeWire-Pulse / ALSA device listing. Virtual mic via `pactl` null-sink + remap-source (PipeWire-compatible) or manual `pw-loopback` / Helvum hints.

## Citation

```bibtex
@article{ma2026meanvc2,
  title={MeanVC2: Robust Low-Latency Streaming Zero-Shot Voice Conversion},
  author={Ma, Guobin and Xia, Yuxuan and Jiang, Yuepeng and Guo, Dake and Xie, Hanke and Hu, Jingbin and Wang, Yanbo and Xie, Lei and Zhu, Pengcheng},
  journal={arXiv preprint arXiv:2606.09050},
  year={2026}
}
```

- Paper: https://arxiv.org/abs/2606.09050  
- Code: https://github.com/ASLP-lab/MeanVC2  
- Weights: https://huggingface.co/ASLP-lab/MeanVC2  

See [STATUS.md](STATUS.md) for honest limitations.

## License

Apache-2.0 for **this** wrapper — [LICENSE](LICENSE) + [NOTICE](NOTICE).
