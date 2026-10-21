# Voice modulation

Voice modulation changes **attributes of your own voice** — pitch, formants, resonance, voice quality, dynamics —
while trying to keep it recognisably you. There is no target speaker, no model download and no consent prompt (it is
your voice); nothing leaves the machine. It is a different system from voice conversion (your speech → somebody
else's timbre) and from speech-to-speech resynthesis (ASR → TTS). See [streaming-vc.md](streaming-vc.md).

```bash
aivoice modulate preset list
aivoice modulate masculine --strength 65           # live, into aivoice-virtual-mic
aivoice modulate deep --strength 40 --body 3
aivoice modulate warm --test-input clip.wav --sink default --duration 10   # hear it on a file first
aivoice modulate broadcast --file in.wav -o out.wav  # offline, prints the measured attribute changes
aivoice modulate preset save my-voice --from masculine --strength 50 --formants -8
aivoice modulate preset use my-voice                 # default for the widget; applied live if a session runs
aivoice vc set --strength 0.4                        # change a running session without restarting it
aivoice vc control toggle-bypass                     # A/B against your unprocessed voice (delay-matched)
aivoice live --engine xvc --voice V --modulate warm  # post-modulation after a VC engine
```

## The DSP engine

`aivoice.modulation.dsp.SpectralModulator` is a streaming, numpy-only engine: it runs on one CPU core and is also the
fallback of every neural VC session. Everything happens inside one STFT (32 ms Hann window, 8 ms hop), so stages
add no latency:

| Control | Implementation | Range |
|---|---|---|
| `--pitch` | phase-vocoder shift of the *excitation* (spectrum ÷ true envelope), identity phase locking | semitones |
| `--formants` | frequency warp of the cepstral true envelope (independent of pitch) | % (−10 = 10 % lower) |
| `--intonation` | pitch-range scale around a 1.2 s running-median F0 (YIN per frame); 0 = monotone | ×1 = unchanged |
| `--body` / `--resonance` | peaking bands at ~180 Hz / ~700 Hz | dB |
| `--brightness` | high shelf from ~3.5 kHz | dB |
| tilt | spectral slope above 500 Hz (presets only) | dB/octave |
| `--breathiness` | random-phase aspiration noise under the warped envelope, > 1.5 kHz; 1.0 ≈ 8 dB below the voice (measured) | 0-1 |
| `--roughness` | 25-60 Hz amplitude modulation on voiced frames | 0-1 |
| `--denoise` | spectral-floor suppression (slow-rise / fast-fall floor tracker) | 0-1 |
| presets only | de-esser, compressor (−26 dBFS, up to 4:1), saturation, band-limiting, robotic phase flattening, wet/dry | |

A soft limiter (−1 dBFS knee) ends every chain. Bypass keeps the same delay as the processed path, so A/B switching
does not jump in time.

### Measured (this machine, CPU, 48 kHz)

| | value |
|---|---|
| algorithmic delay | 32 ms (window − 1 sample; measured by cross-correlation = 1535 samples); 16 ms with `low-latency` |
| compute | p50 0.8 ms, p95 1.7 ms, p99 2.2 ms per 10 ms block (masculine preset); RTF ≈ 0.10 |
| identity round trip | max error 1e-8 (no change = bit-transparent apart from float rounding) |
| pitch accuracy | within 10 cents of the target on harmonic test signals (−12 … +5 st) |
| block-size invariance | output bit-identical for 7, 384, 480 and 4801-sample blocks |
| 30 min unpaced stream | no NaNs, no fallbacks, p95 flat, memory bounded (see experiment O) |

### Measured trade-offs (experiments L, M, N, R)

- Identity retention per preset (GE2E drop): natural 0.00, warm/breathy/rough ≈ 0.00, bright 0.02, broadcast 0.04,
  masculine-subtle 0.04, deep 0.07, masculine-natural 0.09, feminine-natural 0.09, anonymous 0.08, masculine-strong
  0.14 (flagged as drift risk).
- Masculine: formants matter far more than extra pitch drop; body/intonation/brightness showed no effect on the proxy
  and are not in the default ([masculine-voice.md](masculine-voice.md)).
- Phase locking reduces aperiodicity slightly at 32/48 ms windows and is kept; at 16 ms it does not help.
- Pitch shifting doubles YIN aperiodicity (e.g. 0.08 → 0.19): the main residual artifact of the DSP path.

## Presets

`natural`, `masculine`(-subtle/-natural/-strong), `feminine`(-subtle/-natural), `deep`, `bright`, `dark`, `warm`,
`thin`, `full`, `rough`, `breathy` (identity-preserving); `broadcast`, `radio`, `telephone`, `robotic`, `anonymous`,
`character` (character effects); `low-latency` (utility). `anonymous` is a light disguise, **not** anonymisation:
speaker-verification systems can still link a DSP-shifted voice to its owner. For real anonymisation research see the
experimental `streamvoiceanon` engine.

Strength scales the deltas: semitones and dB linearly, ratios in the log domain; 0 is always the unmodified voice.
User presets: `$XDG_CONFIG_HOME/aivoice/modulation_presets.json`.

## Neural modulation

StyleStream (timbre/accent/emotion style conversion) is integrated as an **experimental** engine: research /
non-commercial license, curated style targets only, ~1 s latency. It is not used for the presets above and could not
be compared with the DSP path here (experiment Q not run).

## Limitations

- Phase-vocoder pitch shifting has audible phasiness at large shifts; formant changes are cleaner than pitch changes.
- Breathiness can only be added; there is no harmonic/noise decomposition to remove it.
- Accent / style transformation is not a DSP operation; it needs a neural style model (StyleStream, experimental).
- All measurements here use synthetic formant voices; calibrate on your own voice (`aivoice modulate calibrate`).
