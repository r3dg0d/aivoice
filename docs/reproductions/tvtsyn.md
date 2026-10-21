# TVTSyn (`tvtsyn-repro`, `tvtsyn-aivoice`): paper reproduction checked against official code

**PAPER_REPRODUCTION_WITH_ASSUMPTIONS · confidence MEDIUM (encoder, VQ, timbre HIGH) · official GTM/decoder
weights not inspectable · nothing trained yet.** Map, findings and assumptions:
[`SPEC.md`](../../src/aivoice/research_reproductions/tvtsyn/SPEC.md).

The TVTSyn backbone is included in the official VOSSA release (CC BY-NC 4.0). aivoice verified its numbers
against that code without copying it. The time-varying timbre module (Global Timbre Memory, gate, Slerp) and
the factorised VQ have parameter shapes identical to the official classes (3.77 M and 42 k).

**Finding:** the official encoder has 4 frames of right context in every layer, which is 640 ms of real
lookahead offline; the official streamer approximates it with a 4-frame window. `tvtsyn-repro` streams that
configuration exactly (640 ms). `tvtsyn-aivoice` puts the lookahead in the first layer only (exact 80 ms) and is
**not** the paper's configuration.

## Commands (RTX 4090)

```bash
pip install speechbrain     # x-vector + ECAPA teachers
aivoice reproduce prepare tvtsyn-repro --dataset libritts:/data/LibriTTS --output runs/tvtsyn/prep \
    --teacher hubert-base=/models/hubert-base-ls960 --teacher xvector=/models/spkrec-xvect-voxceleb \
    --teacher ecapa=/models/spkrec-ecapa-voxceleb --device cuda
aivoice reproduce train tvtsyn-repro --stage content --dataset runs/tvtsyn/prep --output runs/tvtsyn/content \
    --device cuda --precision bf16 --batch-size 32 --max-steps 200000 --background
aivoice reproduce train tvtsyn-repro --stage synth --dataset runs/tvtsyn/prep --output runs/tvtsyn/synth \
    --content-checkpoint runs/tvtsyn/content/checkpoints/step00200000.pt --device cuda --batch-size 16 \
    --max-steps 500000 --grad-checkpoint --background
aivoice reproduce resume tvtsyn-repro --output runs/tvtsyn/synth --background
aivoice reproduce export tvtsyn-repro --checkpoint runs/tvtsyn/synth/checkpoints/step00500000.pt --output tvtsyn.pt --install
aivoice reproduce evaluate tvtsyn-repro --checkpoint tvtsyn.pt --source me.wav --target target.wav --output tvtsyn-eval.json
```

For the 80 ms variant use `tvtsyn-aivoice` (preset `aivoice`) in every command; it needs its own content stage.
Ablations: `--set timbre.interp='"lerp"'`, `--set timbre.gate='"fixed"'`.

Live use needs [x-vector ; ECAPA] enrollment inside the worker, which is not wired into `aivoice live` yet.
`evaluate` works offline.
