# Audio showcase

The audio in `examples/audio/` is synthetic and rights-free:

- **Voices.** The speakers are espeak-ng formant voices (`synth-a`, `synth-b`) that imitate no one.
- **Labelling.** Every WAV carries a RIFF `ICMT` comment saying it is synthetic.
- **Loudness.** Every file is normalized to −23 LUFS, so A/B comparisons are not biased by level.

Open [docs/demo/index.html](demo/index.html) through GitHub Pages (serving from the repository root), or
run `python -m http.server` in the repository root and visit `/docs/demo/`. The page has an A/B player
and shows each file's measurements.

## What is (and is not) here

This environment could not download any neural cloning model and has no GPU. The showcase therefore
demonstrates the parts of the pipeline that ran: reference preparation, duration planning, long-form
joins and enhancement. The cloning engines are listed in `manifest.json` as `unavailable`, with the
reason. No placeholder audio stands in for them. Run the generator on a machine with engines installed
and their samples are added automatically.

## Samples for synth-a

| Comparison | A | B | Measured |
|---|---|---|---|
| Reference preparation | [raw upload](../examples/audio/synth-a/upload-raw.wav) | [prepared](../examples/audio/synth-a/upload-prepared.wav) | GE2E similarity to the clean reference: 0.816 → 0.833 |
| Duration planning | [naive plan](../examples/audio/synth-a/duration-naive.wav) | [silence-aware plan](../examples/audio/synth-a/duration-silence-aware.wav) | Planned 6.17 s vs 4.03 s; natural length 5.11 s |
| Long-form joins | [hard concatenation](../examples/audio/synth-a/longform-naive.wav) | [aivoice joins](../examples/audio/synth-a/longform-aivoice.wav) | Level jumps and splice clicks vs crossfades and loudness matching |
| Enhancement | [naive long-form](../examples/audio/synth-a/longform-naive.wav) | [`--enhance auto`](../examples/audio/synth-a/longform-naive-enhanced.wav) | Clicks 2 → 0 |

Reference voice: [synth-a](../examples/audio/synth-a/reference.wav),
[synth-b](../examples/audio/synth-b/reference.wav). The synth-b files follow the same layout.

## Regenerating and validating

```bash
python examples/scripts/generate_audio_demos.py --engines installed --output examples/audio/
python examples/scripts/validate_audio_demos.py   # hashes, labels, authorization, loudness spread, links
```

To add a speaker, edit `examples/audio/speakers.json`. Each entry must set `synthesis_authorized` and
`publication_authorized`, and the generator refuses any entry that does not. Do not add recordings of
real people without their documented consent to publication.
