# O: Long-session streaming stability

- **Status:** measured
- **What changed:** 30 minutes of looped rights-free speech (with pauses) through the live per-block path.
- **Why:** memory leaks, latency drift, hidden-state corruption, NaNs, backlog growth and clock drift only show up
  over long sessions.
- **Implemented as:** `aivoice.vc.longsession.run_long_session` (`aivoice research long-session --minutes 30 [--paced]`).
- **Data:** `O-unpaced-dsp.json`, `O-unpaced-worker.json`, `O-paced.json` (2-minute windows).

## Measured

| run | NaN blocks | fallbacks | block p95 first → last | RSS growth | underruns / drops |
|---|---|---|---|---|---|
| DSP masculine, unpaced (after ring-buffer fix) | 0 | 0 | 1.69 → 1.72 ms | +2.0 MiB | n/a |
| echo worker + post-modulation, unpaced (before fix) | 0 | 0 | 1.82 → 3.61 ms | +11.0 MiB | n/a |
| DSP masculine, real-time paced (before fix, under concurrent CPU load) | 0 | 0 | 5.37 → 2.80 ms | +18.5 MiB | 36 / 0 |

**Decision:** stable. The first unpaced run grew 16.3 MiB until the per-chunk timing list hit its cap; timings and
block statistics are now bounded ring buffers with running totals (+2.0 MiB over 30 min afterwards). In the paced run,
underruns (17 of 36 during the first 2 minutes, with three other CPU-heavy jobs running) did not accumulate latency:
quiet-chunk trimming kept buffered output at 30-50 ms throughout.

**Limitations:** no real audio devices (simulated clocks); no neural engine (the worker run uses a no-op echo engine
to exercise the transport, not a model's state).
