# Live engine comparison

**Bottom line (honest):** in the build environment no neural VC engine could run (no GPU; Hugging Face, where all
of their weights live, was unreachable). So this release does **not** declare a winner between X-VC, RVC and
MeanVC2. It gives them one interface, one live path and one benchmark, and `--engine auto` follows whatever that
benchmark measures on your machine. Everything marked *measured* below was measured here; everything else is a
design figure or a paper claim and is labelled so.

## Inventory

| Engine | Evidence | State here | Zero-shot | Rate | Algorithmic latency (by construction) | License | Why / notes |
|---|---|---|---|---|---|---|---|
| `xvc` X-VC | official implementation (arXiv) | MISSING_RUNTIME | yes | 16 kHz | 240 ms (authors' setting); 180 / 140 ms unvalidated | MIT code; weights per card | codec-space one-step VC; integrated |
| `rvc` RVC v1/v2 | community implementation | MISSING_RUNTIME (no rvc venv here) | no (per-voice .pth) | 48 kHz | ~200 / 310 / 440 ms (block 150 / 250 / 350 ms) | MIT; models per uploader | integrated (c7d5db8 + shared pipeline) |
| `meanvc2` MeanVC2 | official implementation (arXiv) | MISSING_RUNTIME | yes | 16 kHz | 80 ms (40+40) / 160 ms (120+40) + ASR window | README Apache-2.0, LICENSE missing | kept, resident; no longer assumed best |
| `stylestream` | official implementation | MISSING_RUNTIME | curated styles only | 16 kHz | ~645 ms + compute (~1 s reported) | non-commercial | EXPERIMENTAL: too slow for gaming |
| `streamvoiceanon` | official implementation (ICASSP 2026) | MISSING_RUNTIME | yes | 44.1 kHz | ~139 ms by frames (180 ms reported) | Apache-2.0 | EXPERIMENTAL: anonymisation research, CUDA only |
| `rtvc` RT-VC | peer-reviewed demo paper | BLOCKED_CODE | yes | 16 kHz | 61.4 ms reported (CPU) | **no LICENSE** | no weights, no streaming code, no license |
| `zerovc` Zero-VC | arXiv | RESEARCH_ONLY | yes | - | zero lookahead (reported) | - | no public code or weights |
| `tvtsyn` TVTSyn | peer-reviewed (ICLR 2026) | RESEARCH_ONLY | yes | - | < 80 ms GPU (reported) | CC BY-NC 4.0 (indexed) | release could not be inspected here |
| `dsp` modulation | aivoice | **READY** | n/a (own voice) | 48 kHz | 32 ms (16 ms low-latency) | Apache-2.0 | measured below |
| `passthrough` | aivoice | **READY** | n/a | 48 kHz | 0 | Apache-2.0 | last-resort fallback |

"Reported" = from the paper/abstract (search-indexed), never used for routing.

## Why engines were not integrated

- **RT-VC**: the official repository contains model definitions and an offline helper only; no checkpoints, no
  streaming loop, and no LICENSE file (all rights reserved by default). Integrating it would mean re-implementing and
  re-training a model, which would not be RT-VC.
- **Zero-VC**: no code or weights found (demo page only).
- **TVTSyn**: an indexed Hugging Face page under CC BY-NC 4.0 could not be opened from the build environment and no
  GitHub code was found; its time-varying timbre idea is a property of the trained model and cannot be bolted onto
  another engine, so nothing was "borrowed" from it.
- **StreamVoiceAnon+**: weights "coming soon" at the inspected commit.

## Measured here (CPU, no GPU)

`aivoice benchmark vc --synthetic --engines dsp,passthrough` (rights-free espeak-ng sources; DSP with
`masculine-natural`; 48 kHz, 10 ms blocks):

| engine | task | RTF | chunk p50 / p95 / p99 ms | algorithmic | measured delay | GE2E to source | clicks | F0 jumps/s |
|---|---|---|---|---|---|---|---|---|
| `dsp` | modulation | 0.1053 | 0.8105 / 1.818 / 2.337 | 32.0 ms | 30.0 ms (10 ms resolution) | 0.8727 | 0.0 | 0.0 |
| `passthrough` | modulation | 0.0002 | 0.001 / 0.002 / 0.004 | 0.0 ms | 0.0 ms (10 ms resolution) | 1.0 | 0.0 | 0.0 |

Skipped with their real state: `xvc`, `meanvc2` (MISSING_RUNTIME: no torchaudio/weights here), `rvc` (no rvc venv), `rtvc` (BLOCKED_CODE). Results: [research/benchmarks/vc-2026-10-21](research/benchmarks/vc-2026-10-21/results.json).

30-minute streams (experiment O, `docs/research/experiments/O-*.json`):

| run | NaN blocks | fallbacks | block p95 first → last window | RSS growth after 1st window | underruns / drops | notes |
|---|---|---|---|---|---|---|
| DSP masculine, 30 min of audio, unpaced | 0 | 0 | 1.687 → 1.721 ms | 2.0 MiB | - / - | after the ring-buffer fix (before it: +16.3 MiB) |
| subprocess worker (echo) + post-modulation, 30 min, unpaced | 0 | 0 | 1.821 → 3.61 ms | 11.0 MiB | - / - | ran before the ring-buffer memory fix (bounded growth from the timing list) |
| DSP masculine, 30 min real time (capture + playback clocks) | 0 | 0 | 5.37 → 2.8 ms | 18.5 MiB | 36 / 0 | ran concurrently with three other CPU jobs and the test suite; buffered output stayed 30-50 ms, drift trim removed 340 ms of silence in total; started before the ring-buffer memory fix (+ 345 MB is the 30 min test signal held in RAM) |

## How to decide on an RTX 4090

```bash
aivoice benchmark vc --engines xvc,meanvc2,rvc --voice "Target" --quality balanced --consent-ack
aivoice benchmark vc --engines xvc,meanvc2 --voice "Target" --quality best-quality --consent-ack
aivoice vc route --priority balanced --voice "Target"     # now ranked by your measurements
```

The benchmark writes `benchmarks/listening/<date>/` (sources, targets, one folder per engine, all at −23 LUFS,
`manifest.json`, `results.json`, `results.csv`). Listen before trusting any number: GE2E similarity is a similarity
score, not proof of identity, and no objective naturalness predictor is installed (WER/CER is the intelligibility
proxy).

Decision rule used by `auto`: an engine is preferred for `balanced` when its measured target similarity minus WER is
highest after a small latency penalty, within a 250 ms latency budget, and only if it runs faster than real time
(RTF < 1). X-VC would become the default exactly when it wins that comparison on your hardware; MeanVC2 stays
available either way.

## Expectations to verify (not measurements)

- **Lowest latency (by construction):** MeanVC2 40 ms mode (80 ms + ASR window), then StreamVoiceAnon (~140 ms),
  X-VC lowest-latency (140 ms, unvalidated), RVC lowest-latency (~200 ms). DSP modulation (32 ms) is not VC.
- **Quality:** X-VC reports the best streaming WER on Seed-TTS-Eval among the systems it compared; that is the
  authors' claim on their data, not a measurement here.
- **VRAM:** X-VC carries a Whisper-large-v3-size tokenizer encoder; expect several GB. RVC and MeanVC2 are lighter.
  `benchmark vc` records the worker's `torch.cuda.max_memory_allocated()`.
