# Zero-VC (`zerovc-repro`): paper reproduction

**PAPER_REPRODUCTION_WITH_ASSUMPTIONS · confidence MEDIUM (content student and SA module LOW) · no official code or
weights exist · nothing trained yet.** Full map and assumptions: [`SPEC.md`](../../src/aivoice/research_reproductions/zerovc/SPEC.md).

Zero lookahead: every layer is causal and streams with explicit per-layer state, so one 20 ms frame in gives
320 samples out. The speaker anonymiser and the MPD/MSD discriminators are training-only. A test checks that
the inference module never imports the anonymiser. WavLM-large layer 7 → attention pooling gives the timbre
vector, which is computed once per target voice and cached.

Variants: **A** (closest to the paper) distils the causal content student from w2v-BERT 2.0. **B**
(aivoice practical variant, not the paper) trains the same student on HuBERT-base L9 k-means-200 codes.
Ablation: `--set anonymizer.kind=none` trains without anonymisation.

## Commands (RTX 4090)

```bash
# teachers: download yourself (aivoice never downloads them)
hf download facebook/w2v-bert-2.0 --local-dir /models/w2v-bert-2.0
hf download microsoft/wavlm-large  --local-dir /models/wavlm-large

# 1. content student (variant A)
aivoice reproduce prepare zerovc-repro --stage content --dataset libritts:/data/LibriTTS --output runs/zerovc/prep-content \
    --teacher w2v-bert-2.0=/models/w2v-bert-2.0 --device cuda
aivoice reproduce train zerovc-repro --stage content --dataset runs/zerovc/prep-content --output runs/zerovc/content \
    --teacher w2v-bert-2.0=/models/w2v-bert-2.0 --device cuda --precision bf16 --batch-size 32 --workers 8 \
    --max-steps 200000 --checkpoint-every 5000 --eval-every 2500 --compile --background
aivoice reproduce resume zerovc-repro --output runs/zerovc/content --background

# 2. vc stage (frozen content encoder)
aivoice reproduce prepare zerovc-repro --stage vc --dataset libritts:/data/LibriTTS --output runs/zerovc/prep-vc \
    --teacher wavlm-large=/models/wavlm-large --device cuda
aivoice reproduce train zerovc-repro --stage vc --dataset runs/zerovc/prep-vc --output runs/zerovc/vc \
    --content-checkpoint runs/zerovc/content/checkpoints/step00200000.pt --teacher wavlm-large=/models/wavlm-large \
    --device cuda --precision bf16 --batch-size 16 --max-steps 500000 --background
aivoice reproduce resume zerovc-repro --output runs/zerovc/vc --background

# 3. evaluate, export, run
aivoice reproduce export zerovc-repro --checkpoint runs/zerovc/vc/checkpoints/step00500000.pt --output zerovc.pt --install
aivoice reproduce evaluate zerovc-repro --checkpoint zerovc.pt --source me.wav --target target.wav --long-minutes 30 \
    --teacher wavlm-large=/models/wavlm-large --device cuda --output zerovc-eval.json
aivoice live --engine zerovc-repro --voice "Target" --consent-ack
```

Variant B: add `--set content.variant='"B"' --teacher hubert-base=/models/hubert-base-ls960` to prepare and
train (prepare learns the k-means codebook).

PAPER-REPORTED, not reproduced: SS-S 0.171, SS-R 0.521, WER 3.96 %, RTF 0.063 (Xeon 8468V).
