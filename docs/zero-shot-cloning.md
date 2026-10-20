# Zero-shot voice cloning

Clone a voice from one reference recording, with no training. Only clone voices you have the right to use.
Every file aivoice writes is labelled as synthetic speech.

## Quick start

```bash
# 1. Create a profile. The reference is analysed, prepared, and a copy of the original is kept.
aivoice clone create "Narrator" --reference narrator.wav --transcript "Exact words in the clip." --i-have-rights

# 2. Synthesize. The engine is chosen automatically from what is installed.
aivoice clone synthesize --voice Narrator --text "Hello from a cloned voice." -o hello.wav

# 3. Compare the installed engines on the same voice and text.
aivoice clone compare --voice Narrator --text "The same sentence for every engine." --output compare/
```

To set an engine up, see [model-backends.md](model-backends.md). `aivoice engines list` shows which
engines will run.

## Commands

| Command | What it does |
|---|---|
| `clone create NAME --reference F` | Analyse the clip, prepare it and store a v2 profile. The analysis covers duration, speech ratio, SNR, bandwidth, clipping and loudness. Preparation trims, picks the best window, normalizes loudness and denoises when needed. Options: `--transcript`, `--language`, `--denoise auto/on/off`, `--max-seconds`, `--no-prepare`, `--publishable` |
| `clone list` / `clone inspect NAME` | Profiles, their analysis, the preparation steps and any warnings |
| `clone preview NAME` | Short sample written into the profile directory |
| `clone synthesize` | `--engine auto/<name>` and `--priority balanced/latency/similarity/naturalness/multilingual/expressiveness`. Content: `--style`, `--language`. Timing and sampling: `--duration`, `--speed`, `--seed`, `--steps`, `--guidance`. Also `--enhance off/auto/high-fidelity`, `--lexicon` and `--allow-unsupported` |
| `clone compare` | Several engines on one voice and text, with similarity, length and RTF for each. TTS and VC are reported separately |
| `clone delete NAME` | Removes the profile and its cached embeddings and prepared audio |
| `clone migrate` | Copies v1 profiles (voices library, experimental voices) into v2. The originals are never modified |
| `clone attest ID --i-have-rights` | Records consent on a migrated profile |
| `clone adapt` | Optional speaker adaptation (see below) |

## What happens on `synthesize`

The full pipeline is described in [research-fusion.md](research-fusion.md):

1. **Text.** Language detection, normalization (numbers and abbreviations), and your lexicon.
2. **Routing.** Engines that cannot do the request are filtered out with a reason. The survivors are ranked
   by your machine's measurements, or by the fixed order.
3. **Reference.** The profile's prepared reference is used; `--no-prepare` at creation keeps the raw one.
4. **Duration.** A silence-aware duration plan is applied for engines with duration control (VoiceStar,
   X-Voice).
5. **Long text.** Text is split at sentence boundaries, joined with crossfades and loudness matching, and
   each segment's similarity is checked.
6. **Enhancement.** Optional. Every step is identity-guarded and is undone if speaker similarity drops.
7. **Output.** A labelled WAV plus `<name>.json` with the engine, options, plan, timings and metrics.

## Unsupported options fail loudly

Every engine publishes its capabilities. Two examples:

- `--style` on VoiceStar raises an error: VoiceStar has no style input.
- `--style-reference` on any engine raises an error: none supports a separate style recording.

Pass `--allow-unsupported` to drop such options with a printed warning instead.

## Speaker adaptation (optional)

Zero-shot is the default. `aivoice clone adapt --voice NAME --dataset DIR --output OUT --i-have-rights` only does
two things:

- validates an authorized dataset (clip durations, SNR, clipping, transcripts, speaker consistency)
- writes VoxCPM LoRA manifests (`train.jsonl` / `val.jsonl`)

It never starts training. Run VoxCPM's own training script yourself if you choose to.

## Limits

- Similarity scores (GE2E cosine) are evidence that two clips sound alike. They are not proof of identity.
- Quality depends on the engine. None of the neural engines could be run where this release was built
  (see [benchmarks.md](benchmarks.md)).
