# Identity-preserving masculine voice

```bash
aivoice modulate masculine                      # default strength 65 %
aivoice modulate masculine --strength 40
aivoice modulate masculine --strength 65 --formants -12 --pitch -3   # override single attributes
aivoice modulate masculine --file me.wav -o me_masc.wav --identity 0.80   # never drift below 0.80 similarity
aivoice modulate calibrate me.wav --preset masculine --identity 0.80     # strongest safe strength for YOUR voice
```

The goal is **the same speaker with altered vocal attributes**, not a different man. It is not a pitch-down effect.

## Why not just lower the pitch

Perceived voice sex/masculinity is carried by F0 *and* by formant frequencies, which follow vocal-tract length
(longer tract → lower, more closely spaced formants), with smaller contributions from voice quality and intonation
(Peterson & Barney 1952; Hillenbrand et al. 1995; Fitch & Giedd 1999; Hillenbrand & Clark 2009; Skuk & Schweinberger
2014). These sources could not be opened from the build environment and are cited only as the reason to *test*
formants alongside pitch; every default below comes from aivoice's own measurements.

Lowering F0 alone leaves the formants of a shorter vocal tract in place: the classic "pitched-down" voice that sounds
like a slowed recording rather than a deeper speaker.

## Architecture

One 32 ms STFT (`aivoice.modulation.dsp.SpectralModulator`):

1. **Envelope / excitation split.** An iterative cepstral *true envelope* (Röbel & Rodet 2005), with the lifter
   order tied to the tracked F0, separates the vocal-tract envelope from the harmonic excitation.
2. **Pitch** moves the excitation's harmonics (phase vocoder with identity phase locking, Laroche & Dolson 1999),
   so the formants stay put while F0 changes.
3. **Formants** are moved by warping the envelope in frequency, `env_out(f) = env(f / ratio)`, independently of
   pitch (0.90 at full strength ≈ 10 % lower formants ≈ 1.7 cm longer apparent vocal tract for a 15 cm tract).
4. **Optional** attribute stages in the same STFT (no extra latency): body (~180 Hz), resonance (~700 Hz),
   brightness, spectral tilt, intonation range (scale around a running-median F0), roughness, breathiness.

Default `masculine` at strength *s*: pitch −4·s semitones, formant ratio 0.90^s. Built-ins: `masculine-subtle`
(0.35), `masculine-natural` (0.65), `masculine-strong` (1.0).

## What was measured (experiment M, [research/experiments/M.md](research/experiments/M.md))

Data: four rights-free synthetic espeak-ng speakers (two "male", two "female") × 5 sentences, strength 0.65.
*identity* = GE2E cosine to a different utterance of the same speaker; *masc* = projection of the embedding change on a
male-female direction from the same corpus (a **weak proxy**, not perception).

| Arm | identity (all) | masc (all) | identity (female voices) | masc (female voices) | ΔF0 |
|---|---|---|---|---|---|
| unmodified | 0.886 | 0.000 | 0.863 | 0.000 | 0 st |
| pitch only (−2.6 st) | 0.842 | −0.003 | 0.807 | +0.024 | −2.50 st |
| pitch only, 1.5× (−3.9 st) | 0.826 | −0.003 | 0.793 | +0.048 | −3.68 st |
| **pitch + formants** (default) | 0.794 | **+0.043** | 0.779 | **+0.098** | −2.51 st |
| + body resonance | 0.792 | +0.041 | 0.778 | +0.099 | −2.52 st |
| + narrower intonation, darker | 0.793 | +0.039 | 0.781 | +0.099 | −2.42 st |

Decisions (rule: keep a stage only if its masculinity gain is significant and at least as efficient per unit of
identity lost as simply lowering pitch further):

- **Formants: retained.** +0.047 masculinity (95 % CI 0.026–0.067) for −0.047 identity: efficiency 0.99, versus
  0.013 for a bigger pitch drop. Pitch alone barely moves the proxy on the "male" voices at all.
- **Body resonance: rejected as a default** (−0.003, CI −0.007–0.002).
- **Intonation + brightness: rejected as a default** (−0.002, CI −0.007–0.004).

They stay available as `--body`, `--intonation`, `--brightness` because GE2E may simply be insensitive to them, but
they are not presented as making the voice "more masculine".

Experiment N (formants alone) shows identity falling steadily from ratio 0.95 (−0.04) to 0.80 (−0.27), with the
masculinity proxy saturating around 0.85. Experiment L (identity per preset): masculine-subtle −0.04, natural −0.09,
strong −0.14 GE2E — only *strong* crosses the 0.10 drift flag.

## Using identity measurements

GE2E similarity is a similarity score under one encoder, not proof that a voice is "still you". Use it as a guard:

- `--identity FLOOR` (file mode) bisects the strength down until the output's similarity to the input is ≥ FLOOR.
- `aivoice modulate calibrate SAMPLE` prints similarity at each strength on your own recording, so you can pick the
  live strength that suits your voice (`aivoice modulate preset use masculine --strength 0.55`).

## Limitations

- Measured on formant-synthesis voices; real voices respond differently. Calibrate on your own voice.
- No listening test was possible here; "sounds masculine" is not claimed, only measured acoustic and embedding changes.
- Phase-vocoder pitch shifting raises aperiodicity (YIN aperiodicity 0.06→0.14-0.16 on a 200 Hz voice at −4 st):
  some phasiness remains at large shifts. Small formant + pitch changes preserve identity best.
- Breathiness reduction (feminine → masculine voices are often less breathy) is not implemented: removing aspiration
  noise needs a harmonic/noise decomposition that is not in the DSP engine. The `--breathiness` control only adds.
