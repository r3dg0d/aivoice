# Cloning benchmark on your GPU (RTX 4090, NixOS)

One command installs the cloning engines, downloads their weights and builds a dataset of real voices.
It then measures every engine and writes a report, the experiment results and listenable audio.

```bash
nix run github:r3dg0d/aivoice#bench-4090
# or, from a checkout:
nix run .#bench-4090 -- --help
```

It runs inside a small FHS environment (`bench/rtx4090/env.nix`), so pip/uv wheels such as torch and
CTranslate2 work on NixOS. The GPU is reached through your driver's `/run/opengl-driver/lib`; nothing
in your system configuration changes. Everything goes to `~/aivoice-bench/` (`--home DIR` to change it):

```
venvs/      main (aivoice + faster-whisper), voxcpm2, voicestar, xvoice   (CUDA torch in each; up to ~20 GB)
models/     VoxCPM2, X-Voice weights                                      (~7 GB; VoiceStar goes to ~/.cache/huggingface)
dataset/    the speakers (see below)
results/<stamp>/  SUMMARY.md, benchmark/, benchmark-raw/, experiments/, logs/, consent/
results/aivoice-bench-<stamp>.tar.gz
```

Allow about 35 GB of disk. The first run takes a while (downloads plus one synthesis per engine ×
speaker × text × seed). Later runs can use `--skip-setup`.

## The voices

The robotic samples on the showcase page are espeak-ng: no real model could run where they were made.
This benchmark uses **real speech**:

- **Your own voice** (default): the script asks you to confirm that the voice is yours, and whether the
  clones may be published. It then records you reading a short public-domain passage (20 s) and a
  second sentence (8 s, a held-out clip). Use `--reference my.wav --reference-text "exact words"` to
  provide a recording instead, or `--no-record` to skip your voice.
- **LibriSpeech test-clean** (`--librispeech 4` by default): real audiobook readers from LibriVox
  (public domain), released as CC BY 4.0. The download is 346 MB from openslr.org. Speakers alternate
  female and male. They are marked `publication_authorized: false`: use them to measure, not to
  publish clones.

Each speaker also gets a **held-out** recording. Scoring real speech against the same speaker's
reference gives a ceiling: no clone should be expected to beat it.

## What is measured

| Output | Contents |
|---|---|
| `benchmark/report.md` | Per engine: GE2E similarity, WER/CER (Whisper large-v3 on the GPU), RTF, time to first audio, load time, engine VRAM (peak minus idle), peak RAM, plus the real-speech ceiling |
| `benchmark/audio/<engine>/<speaker>/` | Every synthesized clip, labelled synthetic in the WAV metadata |
| `benchmark-raw/` | The same run with raw instead of prepared references (skipped with `--quick`) |
| `experiments/C,F,G` | The CPU ablations, rerun on your machine |
| `experiments/D` | Engine comparison with confidence intervals, and the routing order it implies |
| `experiments/E` | VoxCPM2 style instructions ("slow, calm" / "fast, excited"): does identity hold while speaking rate and pitch move? Example audio in `experiments/audio/E/` |
| `experiments/H` | Full pipeline (prepared reference + duration plan) vs raw input, on clean and on degraded references. Example audio in `experiments/audio/H/` |
| `SUMMARY.md` | All of the above on one page, plus which steps failed and where their logs are |

The measurements are also recorded for aivoice's engine router, under `~/aivoice-bench/share`.

## Options

```
--engines voxcpm2,voicestar,xvoice   which engines (each in its own venv)
--librispeech N                      LibriSpeech speakers (0 = none)
--record NAME | --reference F --reference-text T | --no-record
--seeds 0,1                          seeds per text
--quick                              1 seed, 3 texts, no raw-reference ablation
--skip-setup                         reuse venvs and weights
--yes                                do not ask before downloading
--fake                               plumbing test with fake engines (no GPU needed)
```

## Expected trouble spots

- **The torch build doesn't match the driver.** The script sets `UV_TORCH_BACKEND=auto`, so uv installs
  the PyTorch build for your driver's CUDA version (the default PyPI build needs a 580+ driver for
  CUDA 13). To force one, run `UV_TORCH_BACKEND=cu128 nix run .#bench-4090`. To rebuild an existing
  venv, delete `~/aivoice-bench/venvs/<name>` first.
- **X-Voice** has many dependencies, some built from source (pyopenjtalk, pynini via
  nemo_text_processing). If its install fails, the log says why and the other engines still run.
- **CUDA errors from faster-whisper.** The script points CTranslate2 at torch's pip-installed
  cuBLAS/cuDNN. If that still fails, WER falls back to Whisper on the CPU (slower). The ASR name in
  the report shows which one ran.
- **Recording is silent.** Check `pactl get-default-source`. The script uses `pw-record` on the default
  input.
- **`nvidia-smi not found`.** The VRAM columns stay empty. On NixOS it comes with
  `hardware.nvidia.package`.

## Sharing the results

Send the `.tar.gz`, or just `SUMMARY.md`, to get a reading of the numbers. Before putting samples on
GitHub Pages, keep only speakers whose `consent/*.json` says `publication_authorized: true`. That
means your own voice, if you said yes.
