# Paper reproductions (aivoice 0.7)

aivoice can rebuild streaming voice-conversion systems from their papers when the code or weights are
missing, and integrate the official release when one exists. Every result is labelled by where it came from:

| label | meaning |
|---|---|
| **OFFICIAL_CODE** / **OFFICIAL** weights | the authors' code / checkpoints, run as published |
| **PARTIAL_OFFICIAL_CODE** | authors' code for some parts (here: the model layout), aivoice for the rest |
| **PAPER_REPRODUCTION(_WITH_ASSUMPTIONS)** | aivoice code written from the paper; every open choice is listed in SPEC.md |
| **FINETUNED_PUBLIC_BASE** | an extension fine-tuned from public weights |
| **RETRAINED_PAPER_REPRODUCTION** weights | trained by you with this code, never "official" |
| **RANDOM_INIT** | untrained; refused by every inference path |
| **PAPER-REPORTED** vs **MEASURED BY AIVOICE** | numbers from a paper are never mixed with local measurements or used for routing |

None of the reproductions is a default engine. All of them are EXPERIMENTAL and `--engine auto` never
picks them.

## What exists (2026-10-21)

| engine | paper | source status | arch. confidence | official weights | aivoice weights | streaming lookahead (by construction) | doc |
|---|---|---|---|---|---|---|---|
| `rtvc` | RT-VC (ACL 2025 demo) | PARTIAL_OFFICIAL_CODE: layout parity verified | MEDIUM | none released | none trained | 0 future frames (5 ms frames); causal front-end | [rtvc.md](rtvc.md) |
| `zerovc-repro` | Zero-VC (arXiv 2606.20218) | PAPER_REPRODUCTION_WITH_ASSUMPTIONS | MEDIUM (content/SA LOW) | no code, no weights | none trained | 0 future frames (20 ms frames) | [zerovc.md](zerovc.md) |
| `tvtsyn-repro` | TVTSyn (ICLR 2026) | PAPER_REPRODUCTION_WITH_ASSUMPTIONS: encoder/VQ/timbre verified vs official VOSSA code | MEDIUM (encoder/timbre HIGH) | GTM decoder not inspectable | none trained | 32 frames = 640 ms (official config, exact) | [tvtsyn.md](tvtsyn.md) |
| `tvtsyn-aivoice` | (variant) | aivoice variant of the above | MEDIUM | — | none trained | 4 frames = 80 ms | [tvtsyn.md](tvtsyn.md) |
| `streamvoiceanon-plus` | StreamVoiceAnon+ (arXiv 2603.06079) | FINETUNED_PUBLIC_BASE: extension of the official StreamVoiceAnon | MEDIUM | "coming soon" | none trained | as StreamVoiceAnon (2 frames ≈ 93 ms) | [streamvoiceanon-plus.md](streamvoiceanon-plus.md) |
| `vossa` (`vossa-repro`) | VOSSA (Interspeech 2026) | **OFFICIAL_CODE + OFFICIAL weights** (integrated, not reproduced) | HIGH | yes (CC BY-NC 4.0) | n/a | 4 frames (official streamer) | [vossa.md](vossa.md) |

## Scorecard

Only measured values appear here. Quality columns stay empty until someone trains a checkpoint.
`aivoice reproduce evaluate` fills them.

| engine | streaming exact (causality · reset · chunk 20/40/80 ms) | tiny-overfit gate (CPU, synthetic) | CPU compute, paper size (random weights) | target similarity | source leakage | log-F0 corr | WER |
|---|---|---|---|---|---|---|---|
| `rtvc` | pass · pass · pass (≤ 2e-5 vs offline) | pitch 0.08× ✓, EMA 0.09× ✓, vocoder: tiny 0.53× ✗ (1500 steps), paper-size 0.48× ✓ (500 steps) | 12.5 M params, RTF 1.18, chunk p50/p95 22/33 ms | — | — | — | — |
| `zerovc-repro` | pass · pass · pass (≤ 1.2e-7) | content A 0.04× ✓, content B 0.02× ✓, vc 0.49× ✓ (300 steps, still falling) | 45.3 M params, RTF 1.99, 37/52 ms | — | — | — | — |
| `tvtsyn-repro` / `-aivoice` | pass · pass · pass (≤ 1.2e-7) | content 0.07× ✓ (200 steps); synth mel 0.29× ✓ at step 230 (run stopped by hand at 230/300) | 89.5 M params, RTF 1.79 / 1.90, 36/47 ms | — | — | — | — |
| `streamvoiceanon-plus` | official streaming path (not re-tested) | tiny official model: 0.26× (150 steps) | — | — | — | — | — |
| `vossa` | official | — | not run (no torchaudio-compatible env, no Hub access) | — | — | — | — |

CPU compute: 4 threads of the build machine's CPU, eager PyTorch, 20 ms chunks, random weights, so it
measures the architecture's cost and nothing about quality
([JSON](../research/benchmarks/repro-2026-10-21/streaming-compute-cpu.json)). None of them is real time on
this CPU. Zero-VC reports RTF 0.063 on a Xeon 8468V, so the paper's content student is evidently much smaller
than aivoice's assumed 29 M-parameter student (Z-C1): that size is the first thing to revisit.
"—" means not measured. These cells are never filled from paper claims. The overfit ratios are final over
initial training loss, and the gate passes at ≤ 0.5.

## Workflow

```bash
aivoice reproduce list                                  # what exists, with provenance
aivoice reproduce info zerovc-repro                     # assumptions, requirements, variants
aivoice reproduce paper-map tvtsyn                      # SPEC.md: paper → module → decision → confidence
aivoice reproduce verify rtvc --official-checkout ~/src/RT-VC --overfit 200   # phase A (+ parity) + phase B

aivoice reproduce prepare zerovc-repro --dataset libritts:/data/LibriTTS --output runs/zerovc/prep \
    --stage content --teacher w2v-bert-2.0=/models/w2v-bert-2.0
aivoice reproduce train zerovc-repro --stage content --dataset runs/zerovc/prep --output runs/zerovc/content \
    --device cuda --precision bf16 --batch-size 32 --grad-accum 2 --workers 8 --max-steps 200000 --background
aivoice reproduce status zerovc-repro --run runs/zerovc/content
aivoice reproduce control pause --output runs/zerovc/content     # resume | stop (checkpoint saved first)
aivoice reproduce resume zerovc-repro --output runs/zerovc/content --background
aivoice reproduce cost --output runs/zerovc/content --target-steps 200000 --price-per-hour 0.7
aivoice reproduce evaluate zerovc-repro --checkpoint model.pt --source me.wav --target target.wav --long-minutes 30
aivoice reproduce export zerovc-repro --checkpoint runs/zerovc/vc/checkpoints/step00400000.pt --output model.pt --install
aivoice live --engine zerovc-repro --voice "Target" --consent-ack
```

**Validation phases.** A: architecture (shapes, parameters, causal conv / KV cache / Slerp / pooling / VQ
tests, official parity where code exists). B: **mandatory tiny overfit**. `reproduce train` runs it on two
utterances of your dataset with the exact recipe and refuses to continue if the loss does not fall to
≤ 0.5× (`overfit/overfit.json`). C: streaming exactness on the trained model. D: small-data training
(e.g. LibriTTS dev). E: full training.

## Ablations that exist (real switches, never default)

| engine | switch | meaning |
|---|---|---|
| zerovc-repro | `--set anonymizer.kind='"none"'` | train without anonymisation |
| zerovc-repro | `--set timbre.pooling='"mean"'` | static mean pooling instead of attention pooling |
| zerovc-repro | `--set content.variant='"B"'` | other content encoder (HuBERT k-means student) |
| tvtsyn-repro | `--set timbre.gate='"fixed"' --set timbre.fixed_alpha=1.0` | static global timbre only |
| tvtsyn-repro | `--set timbre.interp='"lerp"'` | memory without Slerp |
| tvtsyn-repro | `--set timbre.num_tokens=16` | different number of memory slots |
| streamvoiceanon-plus | `--set lambda_emo=0` / `--set use_sep=false` | without emotion distillation / without SEP |

Not implemented (they would be new models, not switches): RT-VC with WavLM content features, RT-VC without F0
normalisation. aivoice-improved variants (better content encoder, stronger speaker embedding, modern F0 estimator,
better emotion teacher) are to be built only after a faithful reproduction trains. They will be registered as
separate `*-aivoice` engines, as `tvtsyn-aivoice` is.

**Masculinisation / modulation research path:** not run. It needs trained conversion models to explore, and no
experiment has shown an improvement, so the production masculine mode is unchanged and no latent direction is
labelled "masculinity".

## Training infrastructure

* **Precision / speed:** `--precision bf16` (default; fp16 uses a GradScaler and needs CUDA), TF32 on by
  default (`--no-tf32`), `--compile`, fused AdamW on CUDA, pinned memory + prefetch, `--grad-checkpoint`
  (activation checkpointing in the transformer stacks), `--grad-accum`.
* **OOM:** the run stops and writes `oom_report.json` with peak memory and concrete suggestions. Nothing
  is reduced automatically.
* **Multi-GPU:** `torchrun --nproc_per_node 2 -m aivoice reproduce train ...` (DDP, rank-aware sampler,
  rank-0 logging/checkpoints). It is implemented but **not tested here** (no GPU).
* **Exact resume:** weights, optimisers, schedulers, GradScaler, step, Python/NumPy/Torch/CUDA RNG, sampler
  position (what training consumed, not what workers prefetched), config and git commit. A test checks that
  training 3 steps, resuming and training 4 more gives bit-identical weights to 7 uninterrupted steps.
* **Reproducibility record** (`run_config.json` and every checkpoint): Python, PyTorch, CUDA, cuDNN, GPU, seed,
  manifest path + sha256, git commit + dirty flag, model config + fingerprint, optimiser hyper-parameters.
  The format is versioned (`aivoice-repro-checkpoint/1`); no migrations exist yet because there is only one version.
* **Checkpoints:** `.pt` + `.sha256` + `.json` sidecar (engine, model name, provenance, licenses, datasets,
  config fingerprint, git commit, environment, parameter shapes). Loading is `weights_only`, verifies the
  checksum, engine and fingerprint, and reports shape mismatches by name.
* **Feature caches:** versioned by extractor, source model and revision (file sha256), config, rate and
  hop. A changed teacher never reuses old features.
* **Metrics:** `metrics.jsonl`, `metrics.csv`, TensorBoard. W&B only with `--wandb`. No telemetry.
* **Background service:** `--background` detaches the run, which then talks through `status.json` (step,
  losses, samples/s, ETA, GPU, VRAM, last checkpoint) and `control.json`. The QuickShell **Research** tab
  only reads status and writes control; it never trains in the GUI process.
* **Cost:** measure a few thousand steps on your GPU, then `reproduce cost` extrapolates (labelled as an
  extrapolation).

## Licenses

aivoice's reproduction code is Apache-2.0. Official code is never vendored:
* RT-VC has no license, so its files are imported from your checkout only for the parity check;
* VOSSA/TVTSyn are CC BY-NC 4.0, read for verification and run from your checkout;
* StreamVoiceAnon is Apache-2.0 and imported from your checkout.

Checkpoints record the licenses of every component (teachers, base model, data). They are marked
`redistributable: false` by default, and aivoice never uploads or publishes weights.
