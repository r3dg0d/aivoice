# Speaker conditioning

How a reference recording becomes the conditioning an engine receives.

## Profile v2

A profile is stored under `~/.local/share/aivoice/speakers/<id>/`. It contains:

- `profile.json`: `profile_version` 2, name, language, transcript, consent, hashes, analysis and provenance
- `original.<ext>`: an untouched copy, unless `--no-keep-original`
- `prepared.wav`: the prepared reference that engines receive
- `prepared.json`: what preparation did, with before and after reports
- `ge2e.npy`: a speaker-verification embedding, used for evaluation only

Engines receive the audio file, not an embedding. Each engine computes its own conditioning:

- VoxCPM2: AudioVAE latents
- X-Voice: mel features
- VoiceStar: Encodec tokens
- MeanVC2: WavLM + ECAPA

aivoice never injects a foreign speaker embedding into a model that was not trained on it.

## Reference analysis

`clone create` reports the following for the clip:

- total and speech duration (VAD) and speech ratio
- estimated SNR, effective bandwidth and clipping ratio
- integrated loudness (BS.1770-4)

It also warns about anything that is known to hurt zero-shot cloning:

- under 3 s of speech, or over 30 s
- more than 0.5 s of silence at either edge
- clipping, or an estimated SNR below 20 dB
- content that looks upsampled from a lower rate
- a DC offset
- parts of the clip that sound like a different speaker

## Preparation (default on)

1. Remove DC offset.
2. Trim leading and trailing silence, keeping pauses inside the clip.
3. Keep the `--max-seconds` window (20 s by default) that contains the most speech.
4. Denoise with spectral gating, only when the estimated SNR is below 25 dB (`--denoise auto`).
5. Normalize to −20 LUFS, peak-limited.

**Identity guard.** Each step is reverted, with the reason recorded, if GE2E similarity between its
output and the original falls below 0.90.

**Evidence.** Experiment C ([research/experiments/C.md](research/experiments/C.md)) covers 4 synthetic
speakers × 5 degradations:

- Preparation alone helps on padded and quiet clips.
- Denoising adds +0.036 cosine on average across all 20 cases (95% CI 0.020–0.053). Most of the gain is
  on noisy clips: at 15 dB SNR, similarity rises from 0.83 to 0.92. Denoising is applied only when a clip
  is measurably noisy, so `auto` is the default.

## Transcripts

| Engine | Uses the reference transcript |
|---|---|
| VoxCPM2 | Optional. With a transcript it uses continuation mode; without one it clones from the reference only |
| X-Voice Stage 2 | No (transcript-free) |
| VoiceStar | Required |
| Fish | Required for reference cloning |
| MeanVC2 | Not applicable (VC) |

The router does not pick an engine that needs a transcript when the profile has none.
