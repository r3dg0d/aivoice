# Live voice paths: conversion, modulation, speech-to-speech

aivoice has three real-time systems. They are different tasks and different code, and share only the microphone,
the limiter and the single `aivoice-virtual-mic`:

| | A. Direct voice conversion | B. Voice modulation | C. Speech-to-speech |
|---|---|---|---|
| what | your speech → a **target voice**'s timbre, your words and prosody | **your** voice with altered attributes | your words → transcript → (rewritten) → synthesized voice |
| needs | a target voice (reference audio, or an RVC model) + consent | nothing | ASR + TTS models; optional LLM |
| engines | X-VC, RVC, MeanVC2; experimental StyleStream, StreamVoiceAnon | DSP (`aivoice.modulation`) | Moonshine/Parakeet → llama.cpp → Fish S2-Pro/Drama 3/... |
| command | `aivoice live / virtualmic / file --engine E --voice V` | `aivoice modulate PRESET` | `aivoice experimental start` |
| latency | engine-dependent: ~80-450 ms algorithmic | 32 ms (16 ms low-latency) | utterance-level (≥ 300 ms) |
| docs | [xvc.md](xvc.md), [rvc-live.md](rvc-live.md), [engine-comparison.md](engine-comparison.md) | [voice-modulation.md](voice-modulation.md), [masculine-voice.md](masculine-voice.md) | [experimental-realtime.md](experimental-realtime.md) |

Pipeline details: [realtime-architecture.md](realtime-architecture.md).

## Engines and their real status

`aivoice engines list` (and `aivoice vc engines`) check each engine on *this* machine:

| State | Meaning | Can start |
|---|---|---|
| READY | runtime and weights found | yes |
| INSTALLED | runtime found; needs a per-voice asset (RVC `.pth`) | yes, with such a voice |
| EXPERIMENTAL | runnable but restricted (latency, license, curated targets); never picked by `auto` | only explicitly |
| MISSING_WEIGHTS | runtime present, checkpoints absent | no |
| MISSING_RUNTIME | the engine's interpreter/packages are absent | no |
| UNSUPPORTED_HARDWARE | e.g. CUDA-only engine without a GPU | no |
| RESEARCH_ONLY | paper only, no public implementation | never |
| BLOCKED_CODE | official code/weights incomplete (e.g. no checkpoints, no streaming loop) | never |
| BLOCKED_LICENSE | usable code exists, but its license forbids this use | never |

Investigated architectures and why they are or are not runnable: [engine-comparison.md](engine-comparison.md).

## Commands

```bash
aivoice engines list                                     # every engine, with state
aivoice engines info xvc                                 # capabilities, license, pinned commit, install hint
aivoice live --engine xvc --voice "Target" --consent-ack # shared pipeline, resident model, fallback
aivoice live --engine auto --priority gaming --voice "Target" --consent-ack
aivoice virtualmic --engine xvc --voice "Target" --consent-ack       # the same into aivoice-virtual-mic
aivoice virtualmic --voice "RVC voice" --consent-ack     # unchanged: the voice's own engine/runtime path
aivoice file in.wav --engine meanvc2 --voice "Target" -o out.wav --consent-ack
aivoice vc route --priority similarity --voice "Target"  # what auto would pick, and why
aivoice vc status --watch                                # live latency / RTF / VRAM / fallbacks
aivoice vc control toggle-bypass                         # A/B
aivoice benchmark vc --engines xvc,meanvc2,rvc --voice "Target" --consent-ack
```

`--engine` on `live` selects aivoice's shared pipeline; without it, `live` keeps the previous MeanVC2
`run_rt.py` path and `virtualmic` keeps the voice's saved engine (RVC v2 when it has a `.pth`, from c7d5db8).

## Routing (`--engine auto`)

Priorities: `balanced`, `latency`, `gaming` (≤ 150 ms), `similarity`, `naturalness` (WER/CER as the measurable
proxy), `streaming-quality` (best similarity ≤ 250 ms), `low-vram`, `cpu`. The router filters on what is runnable,
what the voice provides (reference / `.pth`), hardware and the latency budget, then ranks by **this machine's**
`aivoice benchmark vc` results (an engine that is not real time here — RTF ≥ 1 — is never chosen by score). Without
local measurements it uses aivoice's previous order — RVC for trained voices, then MeanVC2, then X-VC — and says it
is not a quality ranking. Paper numbers are never used for ranking. Modulation engines are never picked for VC.

## Fallback

`--fallback dsp` (default): if the engine crashes, hangs or raises, the session switches to the next engine and then
to passthrough; your microphone output does not stop. The primary is retried after 10 s (up to 3 times).
Remote services are never part of a fallback chain.

## Consent

Converting into another person's voice requires `--consent-ack` (and an authorized voice, as before).
Modulating your own voice (`aivoice modulate`) does not ask: it is your voice, and nothing is cloned.
