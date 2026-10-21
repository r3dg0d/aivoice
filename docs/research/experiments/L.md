# L: Modulation identity retention

- **Status:** measured
- **What changed:** Each built-in preset at its default strength.
- **Why (paper):** Modulation must change attributes while the voice stays recognisably the same speaker.
- **Implemented as:** modulation/presets.py; identity guard in `aivoice modulate --file --identity`
- **Baseline:** the unmodified utterance's similarity to the same anchor
- **Data:** 4 synthetic speakers x 2 sentences

## Measured

```json
{
  "natural": {
    "identity": {
      "mean": 0.9031,
      "ci95": [
        0.8908,
        0.9199
      ],
      "n": 8
    },
    "identity_drop": {
      "mean": 0.0,
      "ci95": [
        -0.0,
        0.0
      ],
      "n": 8
    },
    "f0_delta_st": {
      "mean": 0.0,
      "ci95": [
        0.0,
        0.0
      ],
      "n": 8
    },
    "vtl_delta_cm": {
      "mean": 0.0,
      "ci95": [
        0.0,
        0.0
      ],
      "n": 8
    },
    "clicks": {
      "mean": 0.0,
      "ci95": [
        0.0,
        0.0
      ],
      "n": 8
    }
  },
  "masculine-subtle": {
    "identity": {
      "mean": 0.8633,
      "ci95": [
        0.8523,
        0.8779
      ],
      "n": 8
    },
    "identity_drop": {
      "mean": 0.0398,
      "ci95": [
        0.0311,
        0.0485
      ],
      "n": 8
    },
    "f0_delta_st": {
      "mean": -1.3376,
      "ci95": [
        -1.4266,
        -1.2185
      ],
      "n": 8
    },
    "vtl_delta_cm": {
      "mean": -0.3363,
      "ci95": [
        -1.1988,
        0.3842
      ],
      "n": 8
    },
    "clicks": {
      "mean": 0.0,
      "ci95": [
        0.0,
        0.0
      ],
      "n": 8
    }
  },
  "masculine-natural": {
    "identity": {
      "mean": 0.8162,
      "ci95": [
        0.8069,
        0.8286
      ],
      "n": 8
    },
    "identity_drop": {
      "mean": 0.0869,
      "ci95": [
        0.0795,
        0.0946
      ],
      "n": 8
    },
    "f0_delta_st": {
      "mean": -2.5963,
      "ci95": [
        -2.7176,
        -2.4787
      ],
      "n": 8
    },
    "vtl_delta_cm": {
      "mean": -0.1525,
      "ci95": [
        -1.6987,
        1.2642
      ],
      "n": 8
    },
    "clicks": {
      "mean": 0.0,
      "ci95": [
        0.0,
        0.0
      ],
      "n": 8
    }
  },
  "masculine-strong": {
    "identity": {
      "mean": 0.7612,
      "ci95": [
        0.7515,
        0.7744
      ],
      "n": 8
    },
    "identity_drop": {
      "mean": 0.1419,
      "ci95": [
        0.1206,
        0.1613
      ],
      "n": 8
    },
    "f0_delta_st": {
      "mean": -3.8283,
      "ci95": [
        -3.9637,
        -3.637
      ],
      "n": 8
    },
    "vtl_delta_cm": {
      "mean": 0.9112,
      "ci95": [
        -0.8704,
        2.6352
      ],
      "n": 8
    },
    "clicks": {
      "mean": 0.0,
      "ci95": [
        0.0,
        0.0
      ],
      "n": 8
    }
  },
  "feminine-subtle": {
    "identity": {
      "mean": 0.884,
      "ci95": [
        0.8725,
        0.8956
      ],
      "n": 8
    },
    "identity_drop": {
      "mean": 0.0191,
      "ci95": [
        0.0087,
        0.0289
      ],
      "n": 8
    },
    "f0_delta_st": {
      "mean": 1.4326,
      "ci95": [
        1.388,
        1.4867
      ],
      "n": 8
    },
    "vtl_delta_cm": {
      "mean": -0.8338,
      "ci95": [
        -1.4813,
        -0.255
      ],
      "n": 8
    },
    "clicks": {
      "mean": 0.0,
      "ci95": [
        0.0,
        0.0
      ],
      "n": 8
    }
  },
  "feminine-natural": {
    "identity": {
      "mean": 0.8097,
      "ci95": [
        0.7764,
        0.8377
      ],
      "n": 8
    },
    "identity_drop": {
      "mean": 0.0934,
      "ci95": [
        0.0634,
        0.1264
      ],
      "n": 8
    },
    "f0_delta_st": {
      "mean": 2.794,
      "ci95": [
        2.6257,
        3.0509
      ],
      "n": 8
    },
    "vtl_delta_cm": {
      "mean": -0.7775,
      "ci95": [
        -1.7752,
        0.324
      ],
      "n": 8
    },
    "clicks": {
      "mean": 0.0,
      "ci95": [
        0.0,
        0.0
      ],
      "n": 8
    }
  },
  "deep": {
    "identity": {
      "mean": 0.8363,
      "ci95": [
        0.8172,
        0.8572
      ],
      "n": 8
    },
    "identity_drop": {
      "mean": 0.0668,
      "ci95": [
        0.0538,
        0.0816
      ],
      "n": 8
    },
    "f0_delta_st": {
      "mean": -1.7737,
      "ci95": [
        -1.8424,
        -1.7138
      ],
      "n": 8
    },
    "vtl_delta_cm": {
      "mean": -0.5212,
      "ci95": [
        -1.5475,
        0.1901
      ],
      "n": 8
    },
    "clicks": {
      "mean": 0.0,
      "ci95": [
        0.0,
        0.0
      ],
      "n": 8
    }
  },
  "warm": {
    "identity": {
      "mean": 0.9032,
      "ci95": [
        0.8899,
        0.9195
      ],
      "n": 8
    },
    "identity_drop": {
      "mean": -0.0,
      "ci95": [
        -0.0028,
        0.0026
      ],
      "n": 8
    },
    "f0_delta_st": {
      "mean": -0.0418,
      "ci95": [
        -0.0712,
        -0.0137
      ],
      "n": 8
    },
    "vtl_delta_cm": {
      "mean": -0.3725,
      "ci95": [
        -0.8488,
        0.0638
      ],
      "n": 8
    },
    "clicks": {
      "mean": 0.0,
      "ci95": [
        0.0,
        0.0
      ],
      "n": 8
    }
  },
  "bright": {
    "identity": {
      "mean": 0.8873,
      "ci95": [
        0.8731,
        0.9051
      ],
      "n": 8
    },
    "identity_drop": {
      "mean": 0.0159,
      "ci95": [
        0.0129,
        0.0185
      ],
      "n": 8
    },
    "f0_delta_st": {
      "mean": -0.0305,
      "ci95": [
        -0.0668,
        0.0009
      ],
      "n": 8
    },
    "vtl_delta_cm": {
      "mean": -0.3612,
      "ci95": [
        -0.6725,
        -0.0662
      ],
      "n": 8
    },
    "clicks": {
      "mean": 0.0,
      "ci95": [
        0.0,
        0.0
      ],
      "n": 8
    }
  },
  "dark": {
    "identity": {
      "mean": 0.8978,
      "ci95": [
        0.8774,
        0.9202
      ],
      "n": 8
    },
    "identity_drop": {
      "mean": 0.0053,
      "ci95": [
        -0.0054,
        0.0176
      ],
      "n": 8
    },
    "f0_delta_st": {
      "mean": -0.0511,
      "ci95": [
        -0.0823,
        -0.0212
      ],
      "n": 8
    },
    "vtl_delta_cm": {
      "mean": 0.4538,
      "ci95": [
        -0.4012,
        1.3675
      ],
      "n": 8
    },
    "clicks": {
      "mean": 0.0,
      "ci95": [
        0.0,
        0.0
      ],
      "n": 8
    }
  },
  "breathy": {
    "identity": {
      "mean": 0.9029,
      "ci95": [
        0.8897,
        0.9199
      ],
      "n": 8
    },
    "identity_drop": {
      "mean": 0.0002,
      "ci95": [
        -0.0016,
        0.0019
      ],
      "n": 8
    },
    "f0_delta_st": {
      "mean": -0.09,
      "ci95": [
        -0.1725,
        -0.0245
      ],
      "n": 8
    },
    "vtl_delta_cm": {
      "mean": -0.1662,
      "ci95": [
        -0.5876,
        0.2213
      ],
      "n": 8
    },
    "clicks": {
      "mean": 0.0,
      "ci95": [
        0.0,
        0.0
      ],
      "n": 8
    }
  },
  "rough": {
    "identity": {
      "mean": 0.9031,
      "ci95": [
        0.891,
        0.9195
      ],
      "n": 8
    },
    "identity_drop": {
      "mean": 0.0,
      "ci95": [
        -0.002,
        0.0021
      ],
      "n": 8
    },
    "f0_delta_st": {
      "mean": -0.0112,
      "ci95": [
        -0.0308,
        0.0098
      ],
      "n": 8
    },
    "vtl_delta_cm": {
      "mean": -0.0563,
      "ci95": [
        -0.1,
        -0.0213
      ],
      "n": 8
    },
    "clicks": {
      "mean": 0.0,
      "ci95": [
        0.0,
        0.0
      ],
      "n": 8
    }
  },
  "broadcast": {
    "identity": {
      "mean": 0.8646,
      "ci95": [
        0.8529,
        0.8792
      ],
      "n": 8
    },
    "identity_drop": {
      "mean": 0.0385,
      "ci95": [
        0.0308,
        0.0452
      ],
      "n": 8
    },
    "f0_delta_st": {
      "mean": -0.074,
      "ci95": [
        -0.1009,
        -0.0458
      ],
      "n": 8
    },
    "vtl_delta_cm": {
      "mean": -1.4587,
      "ci95": [
        -3.0302,
        -0.2449
      ],
      "n": 8
    },
    "clicks": {
      "mean": 0.25,
      "ci95": [
        0.0,
        0.625
      ],
      "n": 8
    }
  },
  "anonymous": {
    "identity": {
      "mean": 0.8184,
      "ci95": [
        0.79,
        0.8502
      ],
      "n": 8
    },
    "identity_drop": {
      "mean": 0.0847,
      "ci95": [
        0.062,
        0.1061
      ],
      "n": 8
    },
    "f0_delta_st": {
      "mean": -1.7785,
      "ci95": [
        -1.8439,
        -1.7152
      ],
      "n": 8
    },
    "vtl_delta_cm": {
      "mean": 0.385,
      "ci95": [
        -0.6313,
        1.5664
      ],
      "n": 8
    },
    "clicks": {
      "mean": 0.0,
      "ci95": [
        0.0,
        0.0
      ],
      "n": 8
    }
  }
}
```

**Decision:** presets with mean GE2E drop > 0.10 (drift risk): masculine-strong. `--identity FLOOR` backs strength off automatically in file mode.

**Performance cost / regression:** n/a

## Limitations

- GE2E on synthetic speech; thresholds are not perceptual
