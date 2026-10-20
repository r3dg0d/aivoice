# F: Silence-aware duration planning

- **Status:** measured
- **What changed:** Measure the reference speaking rate over VAD speech only (not the whole clip) when planning target duration.
- **Why (paper):** Cross-Lingual F5-TTS 2: speaking-rate prediction is thrown off by leading/trailing prompt silence.
- **Implemented as:** research/duration.py plan_duration (transcript and transcript-free acoustic variants).
- **Baseline:** Rate over the whole reference clip; and VoiceStar 0.1.0's default estimate (== reference duration).
- **Data:** 4 synthetic espeak-ng speakers x 7 reference->target Harvard-sentence pairs, uniform 0-2 s lead and trail padding

## Measured

```json
{
  "abs_relative_duration_error": {
    "naive_whole_clip": {
      "mean": 0.8447,
      "ci95": [
        0.7016,
        0.9976
      ],
      "n": 28
    },
    "voicestar_default": {
      "mean": 0.8202,
      "ci95": [
        0.6758,
        0.9543
      ],
      "n": 28
    },
    "silence_aware_transcript": {
      "mean": 0.1549,
      "ci95": [
        0.1208,
        0.1917
      ],
      "n": 28
    },
    "silence_aware_acoustic": {
      "mean": 0.1424,
      "ci95": [
        0.101,
        0.1844
      ],
      "n": 28
    }
  },
  "improvement_naive_minus_silence_aware_transcript": {
    "mean": 0.6898,
    "ci95": [
      0.5257,
      0.8603
    ],
    "n": 28
  },
  "by_total_padding": {
    "pad1-2s": {
      "naive_whole_clip": 0.6697,
      "voicestar_default": 0.6766,
      "silence_aware_transcript": 0.169,
      "silence_aware_acoustic": 0.1608
    },
    "pad<1s": {
      "naive_whole_clip": 0.2301,
      "voicestar_default": 0.1465,
      "silence_aware_transcript": 0.1596,
      "silence_aware_acoustic": 0.2459
    },
    "pad>2s": {
      "naive_whole_clip": 1.1879,
      "voicestar_default": 1.1442,
      "silence_aware_transcript": 0.1386,
      "silence_aware_acoustic": 0.0965
    }
  },
  "silence_aware_transcript_vs_speech_span": {
    "abs_rel_error": {
      "mean": 0.1067,
      "ci95": [
        0.0791,
        0.1345
      ],
      "n": 28
    },
    "signed_rel_error": {
      "mean": -0.0285,
      "ci95": [
        -0.0756,
        0.0189
      ],
      "n": 28
    }
  },
  "silence_aware_transcript_signed_vs_total": {
    "mean": -0.1367,
    "ci95": [
      -0.1823,
      -0.091
    ],
    "n": 28
  },
  "syllable_counter": "espeak-ng IPA vowel nuclei",
  "pairs": 28
}
```

**Decision:** RETAIN (default on for engines with duration control): improvement CI excludes zero

**Performance cost / regression:** adds one VAD pass over the reference (~ms); no synthesis cost

## Limitations

- espeak-ng formant speech with exact, constant speaking rate; natural speech varies more
- measures the duration *plan*, not a model's adherence to it (no duration-controlled model ran here)
- the plan predicts the speech span (with internal pauses); ground-truth totals also contain the synthesizer's own edge silence (~0.3 s for espeak-ng), hence a negative signed error vs total. No edge allowance is added because it is engine-specific and would be tuned on espeak
