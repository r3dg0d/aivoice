# Research methodology

How aivoice decides whether a technique from a paper earns a place in the pipeline.

## Rules

1. **Verify the source first.** A paper counts only if its abstract or code could be read. Every claim
   in `research/catalog.py` names where it came from. Anything that could not be retrieved is marked
   `unverified`, and nothing depends on it.
2. **Use the official implementation when there is one.** Engines call the published package with the
   signatures read from its wheel (voxcpm 2.0.3, x-voice 0.1.1, voicestar 0.1.0). Tests use fakes with
   the same signatures (`tests/fakes/fakepkgs`).
3. **Re-implement only what can be checked.** The GE2E encoder and BS.1770-4 loudness are each
   re-implemented and compared with their reference implementations: cosine 1.0000 against Resemblyzer,
   and agreement with pyloudnorm to 4 decimals.
4. **Adapt rather than imitate.** A training-time idea is only applied at inference time when the
   mechanism carries over. For example, the silence-aware speaking rate from Cross-Lingual F5-TTS 2 is
   used, but its learned rate predictor is not. Each adaptation is labelled as one.
5. **No silent fallbacks.** An option an engine cannot honour raises `UnsupportedOption`. It is dropped
   only with `--allow-unsupported`, and the drop is printed.

## Experiment protocol

- **One change per experiment.** Every experiment has a stated baseline (see `aivoice research experiments`).
- **Rights-free data.** Data is synthetic espeak-ng speakers (`evaluation/synthetic_data.py`), so results
  are reproducible and nobody's voice is used. The limitation is stated in each report: formant speech is
  not natural speech.
- **Statistics.** Results are reported as means with bootstrap 95% confidence intervals (2,000 resamples).
- **Decision rule.** A change is **retained** when its CI excludes zero in the right direction and it does
  not regress identity (GE2E cosine) by more than 0.01. Otherwise it is **rejected** or left **optional**.
- **Reports.** Each experiment writes `docs/research/experiments/<id>.json` and `.md`, including
  limitations and the performance cost.
- **Missing prerequisites.** An experiment whose prerequisites are missing is reported as *not run*, with
  the reason. It is never estimated.

## Metrics

| Metric | Implementation | Notes |
|---|---|---|
| Speaker similarity | GE2E cosine (Resemblyzer weights), optional ECAPA if installed | Evidence, not proof of identity |
| WER / CER | Levenshtein over normalized text; ASR from whichever backend is installed | Not reported when no ASR is available |
| Loudness | BS.1770-4 integrated LUFS | |
| Clicks | Second difference against a robust local scale (threshold 20) | |
| Duration error | Planned vs. actual seconds, measured against the total and against the speech span | |
| Latency | First audio, RTF, wall time | Measured per machine; never copied from papers |
| Memory | Peak RSS and CUDA peak (when torch is present) | |
