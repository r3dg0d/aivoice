# Real-time architecture

```
                      ┌──────────────────────────┐
  physical mic ──────▶│ capture (parec / PortAudio)│  48 kHz, 10 ms blocks, bounded drop-oldest queue
                      └────────────┬─────────────┘
          ┌────────────────────────┼──────────────────────────┐
          ▼                        ▼                          ▼
   DIRECT VC                 MODULATION                SPEECH-TO-SPEECH
   (aivoice live/virtualmic  (aivoice modulate)        (aivoice experimental)
    --engine ...)                                      VAD → ASR → stabilizer
   gain → [pre: denoise,     gain → DSP engine         → (LLM) → expression
    pitch/formant norm] →    (pitch, formants,         → TTS
   resample → VC engine →     resonance, quality)
   resample → [post: modul.]
          └────────────────────────┼──────────────────────────┘
                                   ▼
                     soft limiter → player (jitter buffer, drift trim)
                                   ▼
                     aivoice_sink ──monitor──▶ aivoice_mic  ("aivoice-virtual-mic")
                                   ▼
                     Discord / OBS / VRChat / games / browsers (WebRTC)
```

The three systems are separate code paths (`aivoice.vc`, `aivoice.modulation`, `aivoice.realtime`) and separate
commands; they share capture, the limiter/player and the virtual microphone. One session at a time owns the mic.

## Per-block path (`vc/live.py LivePipeline.feed`)

Every capture block goes through the same function live, in `aivoice file --engine`, in `benchmark vc` and in the
long-session test, so measurements describe live behaviour:

1. input gain, NaN/Inf repair;
2. optional pre-stage (`SpectralModulator`: noise suppression, light pitch/formant normalisation);
3. `StreamingResampler` 48 kHz → engine rate (polyphase Kaiser windowed-sinc, chunk-invariant);
4. `engine.process_chunk()` — local (DSP) or a resident worker process (X-VC, MeanVC2, RVC, ...);
5. resample back, short fade-in after an engine switch;
6. optional post-modulation;
7. a FIFO that turns the engine's chunk rhythm into one output block per input block;
8. soft limiter, level meters, timing.

Engines that buffer internally (RVC 150-350 ms blocks, X-VC 60-120 ms steps, StyleStream 600 ms) emit bursts; the FIFO
and the player's 30 ms prebuffer absorb them. A FIFO above 500 ms (an engine emitting much more than it consumed) is
trimmed and logged rather than letting latency grow.

## Persistent models

`WorkerVCEngine` starts `stream_worker.py <handler>` in the engine's own interpreter once (`load`), computes target
conditioning once per voice (`prepare_reference`), resets caches per stream (`start_stream`) and then only sends
`chunk` requests (base64 float32 over JSON lines). Nothing reloads per chunk, per utterance or per callback.
Measured IPC overhead with a no-op worker: **0.13 ms mean, 0.46 ms p99 per 20 ms chunk** (CPU).

## Failure handling

| Failure | Detection | Result |
|---|---|---|
| worker crash (segfault, OOM kill) | pipe closes | next engine in `--fallback` (default `dsp`), then passthrough |
| worker hang / overload | per-chunk timeout (5 s, 10 s for RVC/experimental) | same |
| engine exception | `EngineError` | same |
| primary recovers | supervisor retries after 10 s, up to 3 times | switches back with a 20 ms fade |

The microphone output never stops: a block that fails mid-switch is sent dry. All of it is visible in
`aivoice vc status` (`fallbacks`, `errors`, `restarts`). Covered by `tests/test_vc.py` (crash, timeout, restart).

## Clocks, underruns and drift

Capture and playback run on different clocks. Every playback underrun adds latency permanently, so the runner drops
*quiet* queued chunks while more than prebuffer + 30 ms is buffered (`Player.trim_quiet`): latency stays bounded
without cutting speech. Counters: `underruns`, `dropped_blocks` (capture overflow), `trimmed_ms`.

## Latency budget

| Stage | DSP modulation | VC engine |
|---|---|---|
| capture block accumulation | 10 ms | 10 ms |
| engine algorithmic latency | 32 ms (32 ms window) / 16 ms (`low-latency`) | engine-specific (see `aivoice vc engines`) |
| engine compute | p95 ≈ 1.7 ms per 10 ms block (measured, 1 CPU core) | measure with `benchmark vc` |
| resampling | 0 (DSP runs at 48 kHz) | ≈ 0.75 ms (24-tap filters, both directions) |
| player prebuffer | 30 ms | 30 ms |
| `pacat` latency request | 30 ms | 30 ms |
| PipeWire quantum + application buffer | not controlled by aivoice (typically 5-20 ms) | same |

Glass-to-glass for DSP is therefore **estimated** at roughly 100-130 ms on PipeWire; it was not measured with
real hardware (the build environment has no sound devices). The prebuffer and `pacat` latency can be lowered once
measured stable on your machine.

## Status and control

- `$XDG_RUNTIME_DIR/aivoice/vc.json` — written once a second by the session (engine, voice, input, rates, chunk,
  algorithmic and pipeline latency, block/engine p50/p95/p99, RTF, levels, underruns, drops, fallbacks, GPU via
  `nvidia-smi` every 5 s, worker VRAM every 5 s).
- `vc-control.json` — `aivoice vc set|control` writes parameter changes (preset, strength, overrides, gains, bypass,
  mute); the session applies them within a second without reloading the engine.
- Signals: `SIGUSR1` toggles bypass, `SIGUSR2` toggles mute, `SIGTERM` stops cleanly.

## PipeWire and feedback

The stable `aivoice_sink` + `aivoice_mic` pair (`virtualmic.py`, NixOS module `nixosModules.virtual-mic`) is reused by
every engine. Capturing from `aivoice_mic` or `aivoice_sink.monitor` is refused (`feedback_risk`), and the input picker
in the widget never lists aivoice's own devices. `pipewire-pulse` and PulseAudio both work (`pactl`/`parec`/`pacat`).

## Privacy

Direct VC and modulation are local only: no microphone audio leaves the machine, no telemetry. Remote processing
exists only in speech-to-speech mode and only after `aivoice experimental remote allow`.
