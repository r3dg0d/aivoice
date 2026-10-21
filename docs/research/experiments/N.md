# N: Formant contribution (pitch unchanged)

- **Status:** measured
- **What changed:** Warp only the spectral envelope (formants) by 0.95-0.80.
- **Why (paper):** Formant spacing encodes vocal-tract length, a major cue to body size and sex independent of F0.
- **Implemented as:** SpectralModulator formant_ratio (cepstral true-envelope warp of the excitation-normalised spectrum)
- **Baseline:** ratio 1.0 (STFT round trip only)
- **Data:** 4 synthetic speakers x 3 sentences

## Measured

```json
{
  "ratio 1.0": {
    "identity": {
      "mean": 0.9011,
      "ci95": [
        0.8875,
        0.9153
      ],
      "n": 12
    },
    "identity_unmodified": {
      "mean": 0.9011,
      "ci95": [
        0.8875,
        0.9153
      ],
      "n": 12
    },
    "masc_shift": {
      "mean": -0.0,
      "ci95": [
        -0.0,
        -0.0
      ],
      "n": 12
    },
    "f0_delta_st": {
      "mean": 0.0,
      "ci95": [
        0.0,
        0.0
      ],
      "n": 12
    },
    "vtl_delta_cm": {
      "mean": 0.0,
      "ci95": [
        0.0,
        0.0
      ],
      "n": 12
    },
    "centroid_delta_hz": {
      "mean": 0.0,
      "ci95": [
        0.0,
        0.0
      ],
      "n": 12
    },
    "clicks": {
      "mean": 0.0,
      "ci95": [
        0.0,
        0.0
      ],
      "n": 12
    },
    "voiced_fraction_ratio": {
      "mean": 1.0,
      "ci95": [
        1.0,
        1.0
      ],
      "n": 12
    }
  },
  "ratio 0.95": {
    "identity": {
      "mean": 0.8651,
      "ci95": [
        0.848,
        0.8829
      ],
      "n": 12
    },
    "identity_unmodified": {
      "mean": 0.9011,
      "ci95": [
        0.8875,
        0.9153
      ],
      "n": 12
    },
    "masc_shift": {
      "mean": 0.036,
      "ci95": [
        0.0169,
        0.0559
      ],
      "n": 12
    },
    "f0_delta_st": {
      "mean": -0.0896,
      "ci95": [
        -0.1354,
        -0.046
      ],
      "n": 12
    },
    "vtl_delta_cm": {
      "mean": -0.2675,
      "ci95": [
        -1.1575,
        0.4717
      ],
      "n": 12
    },
    "centroid_delta_hz": {
      "mean": -75.5,
      "ci95": [
        -88.1669,
        -62.0144
      ],
      "n": 12
    },
    "clicks": {
      "mean": 0.0,
      "ci95": [
        0.0,
        0.0
      ],
      "n": 12
    },
    "voiced_fraction_ratio": {
      "mean": 0.9771,
      "ci95": [
        0.957,
        0.9967
      ],
      "n": 12
    }
  },
  "ratio 0.9": {
    "identity": {
      "mean": 0.7917,
      "ci95": [
        0.7754,
        0.8079
      ],
      "n": 12
    },
    "identity_unmodified": {
      "mean": 0.9011,
      "ci95": [
        0.8875,
        0.9153
      ],
      "n": 12
    },
    "masc_shift": {
      "mean": 0.0651,
      "ci95": [
        0.0192,
        0.1115
      ],
      "n": 12
    },
    "f0_delta_st": {
      "mean": -0.1241,
      "ci95": [
        -0.18,
        -0.072
      ],
      "n": 12
    },
    "vtl_delta_cm": {
      "mean": 0.4225,
      "ci95": [
        -0.6718,
        1.3234
      ],
      "n": 12
    },
    "centroid_delta_hz": {
      "mean": -141.9833,
      "ci95": [
        -161.681,
        -118.1827
      ],
      "n": 12
    },
    "clicks": {
      "mean": 0.0,
      "ci95": [
        0.0,
        0.0
      ],
      "n": 12
    },
    "voiced_fraction_ratio": {
      "mean": 0.9928,
      "ci95": [
        0.9761,
        1.0064
      ],
      "n": 12
    }
  },
  "ratio 0.85": {
    "identity": {
      "mean": 0.7083,
      "ci95": [
        0.6896,
        0.7274
      ],
      "n": 12
    },
    "identity_unmodified": {
      "mean": 0.9011,
      "ci95": [
        0.8875,
        0.9153
      ],
      "n": 12
    },
    "masc_shift": {
      "mean": 0.071,
      "ci95": [
        -0.0062,
        0.1485
      ],
      "n": 12
    },
    "f0_delta_st": {
      "mean": -0.1655,
      "ci95": [
        -0.2524,
        -0.0943
      ],
      "n": 12
    },
    "vtl_delta_cm": {
      "mean": 1.2983,
      "ci95": [
        0.1133,
        2.2918
      ],
      "n": 12
    },
    "centroid_delta_hz": {
      "mean": -208.4583,
      "ci95": [
        -235.6929,
        -177.3929
      ],
      "n": 12
    },
    "clicks": {
      "mean": 0.0,
      "ci95": [
        0.0,
        0.0
      ],
      "n": 12
    },
    "voiced_fraction_ratio": {
      "mean": 0.9824,
      "ci95": [
        0.9556,
        1.0104
      ],
      "n": 12
    }
  },
  "ratio 0.8": {
    "identity": {
      "mean": 0.6344,
      "ci95": [
        0.615,
        0.6533
      ],
      "n": 12
    },
    "identity_unmodified": {
      "mean": 0.9011,
      "ci95": [
        0.8875,
        0.9153
      ],
      "n": 12
    },
    "masc_shift": {
      "mean": 0.0702,
      "ci95": [
        -0.0404,
        0.177
      ],
      "n": 12
    },
    "f0_delta_st": {
      "mean": -0.1965,
      "ci95": [
        -0.2889,
        -0.1139
      ],
      "n": 12
    },
    "vtl_delta_cm": {
      "mean": 2.4925,
      "ci95": [
        1.1603,
        3.6692
      ],
      "n": 12
    },
    "centroid_delta_hz": {
      "mean": -248.95,
      "ci95": [
        -277.3125,
        -217.7623
      ],
      "n": 12
    },
    "clicks": {
      "mean": 0.0,
      "ci95": [
        0.0,
        0.0
      ],
      "n": 12
    },
    "voiced_fraction_ratio": {
      "mean": 0.9791,
      "ci95": [
        0.9469,
        1.0195
      ],
      "n": 12
    }
  }
}
```

**Decision:** formant lowering raises apparent VTL monotonically; identity cost grows with the shift - the masculine preset uses 0.90 at full strength (0.934 at the default 0.65)

**Performance cost / regression:** n/a

## Limitations

- LPC formant tracking on formant-synthesised speech; ~5-10 % tracking error is normal
