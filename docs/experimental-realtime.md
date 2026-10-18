# Experimental real-time mode (ASR -> LLM -> TTS)

Status: **experimental**. The pipeline, CLI, model manager and audio routing are
implemented and tested; end-to-end quality and latency depend on the models you
install, and this repository's own measurements so far use a scripted ASR and a
local espeak-ng TTS (see [benchmarks.md](benchmarks.md)).

```
microphone ─▶ capture (callback only enqueues) ─▶ VAD + endpointer ─▶ streaming ASR
   ─▶ transcript stabilizer ─▶ [optional LLM] ─▶ expression compiler ─▶ TTS (streamed)
   ─▶ jitter buffer / limiter ─▶ virtual mic (PipeWire/Pulse) ─▶ Discord / OBS / VRChat
```

This mode *re-synthesizes* what you said in another voice. It is not voice
conversion: the existing MeanVC2 `live` / `virtualmic` commands still do that and
are unchanged.

## First run

```bash
pip install -e '.[experimental]'          # sounddevice, soundfile, onnxruntime, moonshine-voice, huggingface_hub, ...
pip install --no-deps silero-vad          # Silero VAD model file only (no torch)
aivoice experimental enable
aivoice models download moonshine-small --yes   # CPU streaming ASR (fetched by moonshine-voice)
# TTS: start a fish-speech server (docs/drama3.md), or fall back to espeak-ng
aivoice experimental tts use fish-server --url http://127.0.0.1:8080 --model s2-pro
aivoice voice add "Me" --audio me.wav --text "exact words in me.wav" --i-have-rights
aivoice voice use "Me"
aivoice experimental start --consent-ack
```

Select **aivoice-virtual-mic** as the input device in your app.

Try it without a microphone: `aivoice experimental start --input clip.wav --output null --consent-ack`.

## Threads and queues

| Thread | Work | Never blocks on |
|---|---|---|
| audio callback | copy block, `put_nowait` (drop-oldest when 4 s behind) | anything |
| `listen` | VAD, ASR partials, stabilizer commits, finalize at endpoint | TTS / LLM |
| `transform` | LLM rewrite (timeout-bounded), expression compile | ASR |
| `synth` | stream TTS chunks into the player | capture |
| `status` | write `$XDG_RUNTIME_DIR/aivoice/engine.json` every 0.5 s | — |

Queues between stages are bounded (8 segments) and drop the oldest segment when
full, counted as `dropped_segments`. A segment that waited longer than
`audio.max_backlog_ms` (default 4 s) before synthesis starts is dropped as
`backlog`, so output never drifts far behind you.

## Utterances and timelines

Each utterance gets an id (`u00042`) and a timeline stamped with
`time.monotonic()`: speech start, first partial, every partial revision, endpoint,
ASR final, every committed segment (LLM start/end, TTS request, first TTS chunk,
first sample played, done, cancellation reason). `aivoice experimental latency`
reports p50/p90/p95 for:

| Metric | Meaning |
|---|---|
| `speech_end_to_first_audio_ms` | estimated end of your speech (endpoint minus the silence hangover) -> first sample out |
| `endpoint_to_first_audio_ms` | VAD endpoint -> first sample out (negative = speculative output started earlier) |
| `stable_to_first_audio_ms` | first committed text -> first sample out |
| `tts_request_to_first_audio_ms` | TTS time to first audio |
| `capture_to_playback_ms` | start of speech -> first sample out |
| `full_utterance_ms` | start of speech -> last sample out |

## Speculative synthesis

With `stabilizer.speculative = true` (default), stable phrases are spoken while
you are still talking; see [latency-optimization.md](latency-optimization.md).
When an LLM rewrite is active, speculation is turned off automatically, because a
rewrite needs the whole sentence.

## Controls

| Action | How |
|---|---|
| start / stop | `aivoice experimental start` (foreground) / `aivoice experimental stop` |
| mute toggle | `aivoice experimental control toggle-mute` (SIGUSR1) |
| interrupt output | `aivoice experimental control interrupt` (SIGUSR2) |
| push-to-talk | `start --ptt`, then `control ptt-down` / `control ptt-up` |
| emergency stop | `control emergency-stop` (flush everything + mute) |

Bind the `control` commands to compositor hotkeys (Hyprland/Sway/GNOME/KDE): they
are just signals to the engine's pid, so they work on Wayland.

Barge-in (`audio.barge_in`): `queue` (default; your next sentence waits for the
previous output, as a voice changer should), `interrupt` (new speech cancels
pending and playing output), `duck` (lower output while you talk).

Echo: output goes to the virtual mic, not your speakers, so the mic does not hear
it. If you enable `--monitor` and listen on speakers, `audio.echo_gate` raises
the VAD threshold while output plays; use headphones anyway.

## Presets

`aivoice experimental preset list|use|save|show|delete`

| Preset | ASR | LLM | Stabilizer | Notes |
|---|---|---|---|---|
| ultra-low-latency | moonshine-small | off | speculative, 2-word commits | 220 ms endpoint hangover, 40 ms prebuffer |
| balanced (default) | auto (balanced) | rule-based hybrid | speculative | |
| expressive | moonshine-medium | expressive | final only | Drama 3 tags, 80 ms prebuffer |
| character | configured | character | final only | uses the voice profile's character prompt |
| accuracy | auto (accurate) | configured | agreement 3, final only | 500 ms hangover |

`preset save NAME` stores the current settings as a custom preset.

## Privacy

Everything is local by default. A remote TTS (a fish-speech server on another
host, or the hosted Fish Audio API) only works after
`aivoice experimental remote allow DEST`. Transcribed text, plus the reference
audio of local voice profiles, is what is sent; microphone audio never is. There is no
automatic fallback from local to cloud.

## Implemented vs. planned

| Implemented | Experimental (works, needs your hardware to validate) | Not implemented |
|---|---|---|
| engine, queues, timelines, controls, presets, CLI | Moonshine and Parakeet with real weights | GUI (the project has none; status is CLI/`--watch`) |
| HF model manager | Drama 3 through fish-speech / SGLang-Omni | CUDA graphs / custom CUDA streams (left to the TTS server) |
| fish-speech + hosted API clients (tested against a protocol fake) | llama.cpp rewriting on your CPU | WebSocket `/v1/tts/live` hosted streaming (HTTP streaming is used) |
| virtual mic (tested against a real PulseAudio daemon) | latency targets in the table below | importing hosted Fish voices into local Drama 3 (no official API) |

Targets from the design brief are aspirational: Verbatim < 300 ms, Balanced < 500 ms,
Expressive < 800 ms, Character < 1000 ms (endpoint -> first audio). See
[benchmarks.md](benchmarks.md) for what has actually been measured.
