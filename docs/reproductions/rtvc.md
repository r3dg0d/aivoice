# RT-VC (`rtvc`): reconstruction around the official layout

**PARTIAL_OFFICIAL_CODE · confidence MEDIUM · no official weights, training code or license.** Map and
deviations: [`SPEC.md`](../../src/aivoice/research_reproductions/rtvc/SPEC.md).

The official repository (Berkeley-Speech-Group/RT-VC @ `9d9a55d`) has the model classes, but no weights,
training code, streaming loop or **LICENSE**. aivoice re-implements the layout without copying any code, and
`aivoice reproduce verify rtvc --official-checkout DIR` confirms that the pitch tracker, EMA inverter and DDSP
vocoder have **identical parameter shapes and counts** to the official classes (2.53 M / 2.13 M / 7.80 M at the
default sizes).

What aivoice adds:
* a **causal** STFT front-end (the official helpers are centred: 32 ms lookahead);
* exact streaming state for every part (conv buffers, upsampling tails, float64 oscillator phase,
  deterministic noise generator with OLA tail, reverb tail);
* the full training pipeline: YIN pitch/periodicity targets, loudness as the official mean-magnitude
  definition, SPARC EMA pseudo-labels, MFCC/log-mel inputs, gain/noise augmentation, speaker conditioning
  through FiLM, and the DDSP multi-scale spectral loss.

## Commands (RTX 4090)

```bash
pip install speech-articulatory-coding          # SPARC: EMA pseudo-label teacher (training only)
aivoice reproduce prepare rtvc --dataset libritts:/data/LibriTTS --output runs/rtvc/prep --teacher sparc=/models/sparc --device cuda
for s in pitch ema; do
  aivoice reproduce train rtvc --stage $s --dataset runs/rtvc/prep --output runs/rtvc/$s --device cuda \
      --precision bf16 --batch-size 64 --max-steps 100000 --background
done
aivoice reproduce train rtvc --stage vocoder --dataset runs/rtvc/prep --output runs/rtvc/vocoder \
    --teacher wavlm-large=/models/wavlm-large --device cuda --batch-size 32 --max-steps 300000 --background
aivoice reproduce resume rtvc --output runs/rtvc/vocoder --background
aivoice reproduce export rtvc --pitch runs/rtvc/pitch/checkpoints/step00100000.pt --ema runs/rtvc/ema/checkpoints/step00100000.pt \
    --vocoder runs/rtvc/vocoder/checkpoints/step00300000.pt --output rtvc.pt --install
aivoice reproduce evaluate rtvc --checkpoint rtvc.pt --source me.wav --target target.wav --long-minutes 30 --output rtvc-eval.json
aivoice live --engine rtvc --voice "Target" --consent-ack
```

PAPER-REPORTED, not reproduced: 61.4 ms latency on CPU.
