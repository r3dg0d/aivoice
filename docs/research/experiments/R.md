# R: Chunk smoothing and phase continuity (DSP path)

- **Status:** measured
- **What changed:** Identity phase locking on/off; 16/32/48 ms analysis windows; capture block size.
- **Why (paper):** Phase-vocoder phasiness and frame-boundary discontinuities are the audible failure of streaming pitch shifting (Laroche & Dolson 1999).
- **Implemented as:** SpectralModulator(phase_locking, window_ms); output independent of capture block size by design
- **Baseline:** phase locking off
- **Data:** 4 synthetic speakers x 2 sentences, -4 st / 0.93 formants

## Measured

```json
{
  "window 16 ms, phase locking on: aperiodicity on voiced frames": {
    "mean": 0.2425,
    "ci95": [
      0.1694,
      0.3185
    ],
    "n": 8
  },
  "window 16 ms, phase locking off: aperiodicity on voiced frames": {
    "mean": 0.1979,
    "ci95": [
      0.1559,
      0.242
    ],
    "n": 8
  },
  "window 32 ms, phase locking on: aperiodicity on voiced frames": {
    "mean": 0.1937,
    "ci95": [
      0.1566,
      0.2333
    ],
    "n": 8
  },
  "window 32 ms, phase locking off: aperiodicity on voiced frames": {
    "mean": 0.198,
    "ci95": [
      0.1625,
      0.2363
    ],
    "n": 8
  },
  "window 48 ms, phase locking on: aperiodicity on voiced frames": {
    "mean": 0.1874,
    "ci95": [
      0.1516,
      0.2239
    ],
    "n": 8
  },
  "window 48 ms, phase locking off: aperiodicity on voiced frames": {
    "mean": 0.2081,
    "ci95": [
      0.1675,
      0.2526
    ],
    "n": 8
  },
  "original: aperiodicity on voiced frames": {
    "mean": 0.08,
    "ci95": [
      0.0691,
      0.0916
    ],
    "n": 8
  },
  "block_size_invariance_max_abs_diff": 0.0
}
```

**Decision:** phase locking RETAIN at 32 ms (0.194 vs 0.198); default window 32 ms (32 ms delay); 16 ms offered as the low-latency preset (16 ms delay), where phase locking is switched off automatically because it measured worse there; block size does not change the output (max diff above). Cross-frame peak matching was also tried and measured worse; not kept.

**Performance cost / regression:** n/a

## Limitations

- YIN aperiodicity is an objective periodicity proxy, not a listening test
- X-VC / RVC / MeanVC2 crossfades could not be compared here (no weights)
