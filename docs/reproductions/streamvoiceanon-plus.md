# StreamVoiceAnon+ (`streamvoiceanon-plus`): fine-tuning extension

**FINETUNED_PUBLIC_BASE · confidence MEDIUM · authors' weights unreleased · nothing trained yet.** Map:
[`SPEC.md`](../../src/aivoice/research_reproductions/streamvoiceanon_plus/SPEC.md).

This is an extension of the official StreamVoiceAnon, not a rebuild. Training pairs a neutral prompt with an
emotional source from the same speaker, inserts learned semantic + acoustic [SEP] tokens between them, and adds
a frame-level emotion2vec distillation loss on the Slow-AR hidden states:
L = L_slowAR + L_fastAR + λ·L_emo. The teacher and the head are training-only. It was tested against the
official dual-AR classes (tiny, random) from the checkout.

## Commands (RTX 4090, StreamVoiceAnon environment)

```bash
export AIVOICE_STREAMVOICEANON_REPO_DIR=~/.cache/aivoice/vendor/StreamVoiceAnon
pip install funasr                                  # emotion2vec teacher (training only)
aivoice reproduce prepare streamvoiceanon-plus --dataset custom:/data/ESD-en --output runs/svplus/prep --device cuda \
    --teacher emotion2vec=/models/emotion2vec_plus_base
aivoice reproduce train streamvoiceanon-plus --stage finetune --dataset runs/svplus/prep --output runs/svplus/ft \
    --device cuda --precision bf16 --batch-size 8 --grad-accum 4 --max-steps 50000 --background
aivoice reproduce resume streamvoiceanon-plus --output runs/svplus/ft --background
aivoice reproduce export streamvoiceanon-plus --checkpoint runs/svplus/ft/checkpoints/step00050000.pt --output svplus.pt
```

ESD utterances must be labelled: the custom scanner takes `<speaker>/<file>`, so add `"emotion"` to each manifest
row (or lay the corpus out per speaker and post-process the manifest). `evaluate` / live use go through the
official engine and need the SEP hook in the prompt prefill (SPEC P-I2), which is not wired yet.
