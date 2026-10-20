# C: Reference preprocessing (trim, best window, loudness, optional denoise)

- **Status:** measured
- **What changed:** Condition engines on a prepared copy of the reference instead of the raw upload.
- **Why (paper):** Cross-Lingual F5-TTS 2 and VoxCPM2 both trim prompt silence; noisy/quiet prompts degrade zero-shot cloning.
- **Implemented as:** research/reference_quality.py prepare_reference with identity guard; spectral gating denoise optional.
- **Baseline:** raw degraded reference
- **Data:** 4 synthetic speakers x 5 degradations; anchor = a different clean utterance of the same speaker

## Measured

```json
{
  "similarity_to_clean_other_utterance": {
    "silence-pad": {
      "raw": 0.9382,
      "prepared": 0.957,
      "prepared+denoise": 0.9579,
      "prepared+auto (default)": 0.957
    },
    "quiet": {
      "raw": 0.9095,
      "prepared": 0.919,
      "prepared+denoise": 0.9178,
      "prepared+auto (default)": 0.919
    },
    "noise-15dB": {
      "raw": 0.8334,
      "prepared": 0.832,
      "prepared+denoise": 0.9226,
      "prepared+auto (default)": 0.9226
    },
    "clipped": {
      "raw": 0.7886,
      "prepared": 0.793,
      "prepared+denoise": 0.8249,
      "prepared+auto (default)": 0.793
    },
    "combined": {
      "raw": 0.8589,
      "prepared": 0.8913,
      "prepared+denoise": 0.9466,
      "prepared+auto (default)": 0.9466
    }
  },
  "improvement_prepared_minus_raw": {
    "silence-pad": {
      "mean": 0.0188,
      "ci95": [
        0.0091,
        0.0324
      ],
      "n": 4
    },
    "quiet": {
      "mean": 0.0094,
      "ci95": [
        0.0027,
        0.0148
      ],
      "n": 4
    },
    "noise-15dB": {
      "mean": -0.0014,
      "ci95": [
        -0.0037,
        0.0002
      ],
      "n": 4
    },
    "clipped": {
      "mean": 0.0044,
      "ci95": [
        -0.0031,
        0.0145
      ],
      "n": 4
    },
    "combined": {
      "mean": 0.0324,
      "ci95": [
        0.0232,
        0.0416
      ],
      "n": 4
    }
  },
  "improvement_denoise_minus_prepared": {
    "silence-pad": {
      "mean": 0.0009,
      "ci95": [
        -0.0012,
        0.0027
      ],
      "n": 4
    },
    "quiet": {
      "mean": -0.0012,
      "ci95": [
        -0.0047,
        0.0024
      ],
      "n": 4
    },
    "noise-15dB": {
      "mean": 0.0905,
      "ci95": [
        0.0707,
        0.1224
      ],
      "n": 4
    },
    "clipped": {
      "mean": 0.0318,
      "ci95": [
        0.0124,
        0.0502
      ],
      "n": 4
    },
    "combined": {
      "mean": 0.0553,
      "ci95": [
        0.0339,
        0.0849
      ],
      "n": 4
    }
  },
  "overall_prepared_minus_raw": {
    "mean": 0.0127,
    "ci95": [
      0.0067,
      0.0191
    ],
    "n": 20
  },
  "overall_default_auto_minus_raw": {
    "mean": 0.0419,
    "ci95": [
      0.0245,
      0.0608
    ],
    "n": 20
  },
  "overall_denoise_minus_prepared": {
    "mean": 0.0355,
    "ci95": [
      0.0198,
      0.0529
    ],
    "n": 20
  },
  "encoder": "GE2E (Resemblyzer pretrained.pt, LSTM 3x256)",
  "note": "embeddings computed without the encoder's own silence trimming, so preparation effects are visible"
}
```

**Decision:** preparation: RETAIN (default on); denoise: RETAIN as 'auto' (only when estimated SNR < 25 dB, identity-guarded)

**Performance cost / regression:** one analysis + preparation pass at profile creation; no per-utterance cost

## Limitations

- GE2E similarity on formant speech; identity effects on real voices may differ
- measures the conditioning clip, not cloned output (no cloning engine ran here)
