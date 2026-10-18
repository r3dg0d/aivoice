# Moonshine Voice (streaming ASR on CPU)

Backend names: `moonshine-tiny`, `moonshine-small`, `moonshine-medium`.

## What it is

[Moonshine Voice](https://github.com/moonshine-ai/moonshine) ships a C++ runtime
with Python bindings (`pip install moonshine-voice`, imported as
`moonshine_voice`). Its streaming architectures keep encoder state between calls
and emit line events as they decode. aivoice uses that directly:

| aivoice | moonshine-voice 0.1.5 |
|---|---|
| `load_model()` | `Transcriber(model_path, ModelArch.SMALL_STREAMING, update_interval)` |
| `transcribe_stream(chunk)` | `Stream.add_audio(list[float], 16000)` + `LineStarted` / `LineTextChanged` / `LineCompleted` listeners |
| `finalize_utterance()` | `Stream.stop()` (flushes; remaining lines complete) |
| `transcribe(clip)` | `transcribe_without_streaming()` |

These names were checked against the package source (0.1.5), not assumed.
Capabilities report `native_streaming=True`.

## Models and downloads

The runtime loads `.ort` model files that its own downloader fetches from
`download.moonshine.ai` (`get_model_for_language(lang, arch)`). The Hugging Face
repositories (`UsefulSensors/moonshine-streaming-*`) hold the transformers
checkpoints, a different format, so aivoice does not download those for this
backend. aivoice never lets the runtime download implicitly:

```bash
aivoice models download moonshine-small --yes     # explicit; registers the path
aivoice experimental asr use moonshine-small
```

`asr.allow_model_download = true` permits the runtime to fetch at load time
instead.

Licensing: moonshine-voice prints a notice that non-English models are under the
non-commercial Moonshine Community License; check https://www.moonshine.ai/license
for your language.

## Tuning

* `update_interval` (0.2 s in aivoice) is the floor between decoding passes; the
  runtime stretches it when a pass takes longer than the audio it covers.
* Runs on CPU, leaving the GPU to TTS. Expected footprint (estimates, measure with
  `aivoice experimental asr benchmark`): tiny ~400 MiB, small ~700 MiB, medium ~1.2 GiB RAM.
* `Transcriber.set_keyterms([...])` biases toward names; not yet exposed in the CLI.

## Status

The adapter is unit-tested against a fake module that mirrors the 0.1.5 event
API. It has **not** been run against real Moonshine weights in this repository's
CI environment (the model CDN was unreachable there). Run
`aivoice experimental asr benchmark` on your machine.
