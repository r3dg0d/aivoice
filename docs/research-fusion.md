# Research fusion

aivoice does not train a new model. It combines published engines with inference-time techniques drawn
from the papers in [research/research-review.md](research/research-review.md). It keeps only the
techniques that measurably help, and only where they apply.

## Pipeline (`fusion/pipeline.py`)

| Stage | Technique | Source | Evidence |
|---|---|---|---|
| Text | Normalization, language detection, `[main\|lang]` code-switch tags for X-Voice | X-Voice | Unit tests |
| Route | Capability filter, then local measurements, then the fixed order | aivoice | Unit tests |
| Reference | Trim, best window, loudness, auto-denoise, identity guard | Cross-Lingual F5-TTS 2 and VoxCPM2 prompt hygiene | Experiment C: retained |
| Duration | Speaking rate measured over VAD speech only, then target duration | Cross-Lingual F5-TTS 2 (adapted), applied to VoiceStar and X-Voice | Experiment F: retained |
| Long-form | Sentence segmentation, pauses, crossfades, loudness matching, per-segment similarity check | VoiceStar (long-form), aivoice | Experiment G: retained |
| Enhance | Click repair and loudness, identity-guarded; `off` by default | aivoice | Experiment G: optional |
| Evaluate | GE2E similarity, acoustic metrics | GE2E | Exact reproduction |

## The duration fusion

VoiceStar has native duration control, but voicestar 0.1.0's default target duration equals the length
of the reference clip, whatever the target text. Cross-Lingual F5-TTS 2 notes that speaking-rate
estimates are thrown off by silence at the edges of the prompt. aivoice combines the two:

1. Measure the reference speaking rate over VAD speech only. Syllables come from the transcript, or are
   counted from acoustic nuclei when there is no transcript.
2. Plan the target duration as the target syllables divided by that rate, plus punctuation pauses.
3. Pass the plan to the engine's native duration input.

Experiment F covers 28 synthetic pairs with 0–2 s of random edge padding:

| Method | Mean absolute duration error (95% CI) |
|---|---|
| Rate over the whole clip | 0.845 (0.70–1.00) |
| VoiceStar 0.1.0 default | 0.820 (0.68–0.95) |
| Silence-aware, with transcript | 0.155 (0.12–0.19) |
| Silence-aware, acoustic | 0.142 |

Against the target's speech span, the signed error is −2.9% (CI −7.6% to +1.9%). The remaining bias
against total length (−13.7%) comes from espeak's own edge silence in the target, not from the plan.

## Not fused, and why

- **Vector-field fusion (PFluxTTS).** `research/flow_matching.py` implements fused-field ODE sampling,
  but it requires two models with the same latent space, frame rate and conditioning. No installed pair
  meets that, and fusing incompatible fields would produce noise. It stays experimental and unused.
- **Learnable speaker encoder (Confucius4-TTS).** This is part of that model's training. It cannot be
  bolted onto another engine.
- **Disentangled timbre and style references (FC-TTS).** There is no public code or weights.
  `--style-reference` is refused rather than faked.
