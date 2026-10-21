# StreamVoiceAnon+: fine-tuning extension → aivoice code

**Status:** FINETUNED_PUBLIC_BASE. Architecture confidence **MEDIUM** for the extension; the base model is
the official code. Weights: **NONE**. The authors' StreamVoiceAnon+ weights were "coming soon" at the
inspected commit; aivoice fine-tunes the public StreamVoiceAnon base, and any resulting weights are named
`aivoice/streamvoiceanon-plus-finetune-v0.1`.

Base: Plachtaa/StreamVoiceAnon @ `2017051` (Apache-2.0; ICASSP 2026, arXiv 2601.13948). Extension paper:
StreamVoiceAnon+ (arXiv 2603.06079), read through search-indexed text only.

**Not a rebuild.** The dual-AR model (`modules/dual_ar_stream.py`), the ARVC wrapper, the ASR-BSQ content
encoder, the firefly codec and the speaker encoders are imported from **your** StreamVoiceAnon checkout.
aivoice adds only the training extension below. No official file is modified or copied.

## Map

| described component | aivoice | how it attaches to the official code |
|---|---|---|
| neutral prompt + emotional source of the same speaker | `train/dataset.build_pairs` (needs `emotion` labels, e.g. ESD) | — |
| semantic and acoustic [SEP] boundary tokens | `StreamVoiceAnonPlus.sep` (2 learned embeddings) | appended after the interleaved prompt |
| prompt as context, loss on the source | prompt + SEP passed as the official `spk_condition_len` prefix | the official forward then places its targets after the prefix, unchanged |
| Slow-AR / Fast-AR losses | official `token_logits` / `codebook_logits`, CE as in `DualARWrapper.forward` | same formulas |
| emotion2vec frame-level teacher (training only) | `train/preprocess.py` (funasr), mean-pooled 50 Hz → 21.5 Hz codec frames | — |
| distillation from Slow-AR hidden states, L_emo = MSE | `EmotionHead` on the Slow-AR output, captured by a forward hook on the official final norm | forward hook (no code change) |
| L_total = L_slowAR + L_fastAR + λ·L_emo | `StreamVoiceAnonPlus.forward` | — |
| teacher and head training-only; inference unchanged | `inference_state_dict()` drops the head; `export` removes it | official streaming path; the prompt prefill gets the two SEP embeddings (P-I2) |

## Assumptions

| id | what | choice |
|---|---|---|
| P-L1 | λ | 1.0 |
| P-H1 | head | Linear(768→512)-GELU-Linear(→768) on the post-norm Slow-AR output |
| P-A1 | teacher alignment | mean pooling of emotion2vec frames onto codec frames |
| P-R1 | loss region | source frames only (prompt is conditioning) |
| P-F1 | fine-tuning | lr 1e-5, betas (0.9, 0.95), wd 0.01, from `dual_ar_delay_0_8.pth`; delays sampled as in the base (0–8) |
| P-I1 | data preparation | requires an encode helper that returns the official training inputs; the base repo's trainer computes them inside its loop, so `prepare` must be run in the StreamVoiceAnon environment (CUDA) |
| P-I2 | inference SEP | the live path must append the two SEP embeddings after the prompt prefill. Until that hook lands in the worker, an exported `-plus` checkpoint is evaluated offline only |

## Ablations

`--set lambda_emo=0` (no distillation), `--set use_sep=false` (no boundary tokens).

## Validation done here

The extension runs against the **official** dual-AR classes built tiny from the checkout (random weights,
CPU). It produces the three losses, and 150 AdamW steps on a fixed batch took the total loss from 7.35 to
1.92. The head is absent from the inference state. A real fine-tune needs the public base checkpoint, an
emotional corpus and CUDA, none of which were available.
