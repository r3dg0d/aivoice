# Benchmarks

## Running

```bash
aivoice experimental benchmark --consent-ack                 # ASR (all installed), TTS, pipeline
aivoice experimental benchmark --clips my_clips/             # your own name.wav + name.txt pairs
aivoice experimental benchmark --mock                        # aivoice overhead only (scripted ASR + tone TTS)
aivoice experimental asr benchmark [--backend moonshine-small --backend parakeet]
aivoice models benchmark parakeet
```

* ASR backends run on **identical clips** (generated with espeak-ng when no clips are given,
  plus quiet, noisy and interruption variants). Reported: WER (overall and per scenario),
  final latency (last chunk -> final text), first-partial latency, RTF, peak RSS.
  Results feed automatic ASR selection, keyed by a hardware/runtime fingerprint.
* TTS: time to first audio, total time, RTF.
* Pipeline: the per-utterance timeline metrics (p50/p90/p95), underruns, dropped
  capture blocks, dropped segments, CPU share, peak RSS, peak GPU memory (nvidia-smi).
* Reports: `~/.cache/aivoice/benchmarks/<stamp>-<kind>.json` and `.txt`.

Generated espeak-ng speech is robotic: fine for latency, only indicative for WER.
Use real recordings for accuracy decisions. "Rapid" uses a faster espeak rate.

## Measured so far (2026-10-18, development container)

Machine: Intel Xeon @ 2.80 GHz, 4 threads, 15 GiB RAM, **no GPU**, Python 3.13.
ASR model weights and the Hugging Face Hub were not reachable from this
environment, so **no real ASR or Fish model has been benchmarked here**.

| What | Result | Notes |
|---|---|---|
| Silero VAD (bundled ONNX, onnxruntime, 1 thread) | 0.24 ms per 32 ms window (7.6 ms CPU per second of audio) | real model |
| Energy VAD | 0.012 ms per 20 ms frame | |
| espeak-ng TTS, time to first audio | p50 19 ms, p95 22 ms (n=6) | real local engine, whole-utterance rendering |
| aivoice overhead, stable text -> first played sample | p50 9 ms, p95 10 ms | mock pipeline: scripted ASR + tone TTS, 13 clips |
| pipeline with real espeak-ng TTS, stable text -> first played sample | p50 17 ms, p95 27 ms | scripted ASR (simulated, 2.7 words/s) |
| same, TTS request -> first audio | p50 14 ms, p95 20 ms | |
| same, endpoint -> first audio | p50 -1044 ms | negative: speculative output began before the endpoint |
| underruns / dropped capture blocks / dropped segments | 0 / 0 / 0 | |
| end-to-end into a real PulseAudio virtual mic | 13.9 s of audio recorded back from `aivoice_mic`, peak 0.88 | espeak-ng -> pacat -> null sink -> remap source -> parec |

What these numbers mean: aivoice's own queueing, scheduling, buffering and
playback add about 10 to 30 ms. They say nothing about Moonshine, Parakeet, Drama 3 or
S2-Pro speed; those must be measured on the target RTX 4090 machine with
`aivoice experimental benchmark`.

## Not yet measured

* Moonshine tiny/small/medium WER and latency (runtime CDN unreachable here)
* Parakeet TDT 0.6B v3 (no GPU, Hub unreachable here)
* Drama 3 / S2-Pro time to first audio (no GPU; Drama 3 does not fit a 4090)
* llama.cpp rewrite latency (no prebuilt llama-cpp-python wheel and no Hub access here)
* Extended sessions (hours) and VRAM over time

## Zero-shot cloning benchmark (0.5)

On a GPU machine, the one-command kit in [benchmark-rtx4090.md](benchmark-rtx4090.md) does everything below with real voices.

```bash
aivoice benchmark research --synthetic --consent-ack            # rights-free espeak-ng speakers
aivoice benchmark research --dataset my_set/ --engines voxcpm2,voicestar --seeds 0,1,2
aivoice research experiments --run C,F,G --output docs/research/experiments
```

The benchmark runs every engine on the same references, texts and seeds. It reports, per engine:

- GE2E similarity to the reference, and WER/CER when an ASR backend is installed
- first-audio time, RTF and wall time
- peak RSS and peak CUDA memory
- the acoustic metrics: LUFS, peak and clicks

Results are written as `results.json` and `results.csv`, and the router uses them for automatic
selection. A user dataset needs a `consent.json` that states the speakers agreed.

### Measured here (2026-10-20, same container: 4 vCPU Xeon, 15 GiB RAM, no GPU)

No neural cloning engine could run here, because there is no Hugging Face access and no GPU.
`benchmark research --synthetic` therefore reports every cloning engine as unavailable. The timings
below are for aivoice's own processing stages, measured on a 9.4 s, 22.05 kHz reference:

| Stage | Time | When it runs |
|---|---|---|
| GE2E embedding (CPU torch; exact Resemblyzer reproduction) | 226 ms | Evaluation; profile creation |
| Reference analysis (`clone create`) | 0.60 s | Once per profile |
| Reference preparation, including identity-guard checks | 3.1 s | Once per profile |
| Silence-aware duration plan | 159 ms | Per utterance, for engines with duration control |
| `--enhance auto` (click repair, loudness, identity check) | 434 ms | Per utterance, opt-in |

The ablation results (C, F, G) are in [research/experiments](research/experiments/). VoxCPM2's own
documentation reports RTF ≈ 0.3 and about 8 GB of VRAM on an RTX 4090. aivoice has not verified that.
