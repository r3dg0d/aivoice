# Fish Audio Drama 3 (and S2-Pro)

## The hardware fact first

Drama 3 (`fishaudio/drama-3`) is a **35B-parameter mixture-of-experts** model
(3B active per token). Fish Audio lists it at **>= 48 GB of GPU memory in FP8** and
does not support CPU, Apple MPS or AMD ROCm. A community INT4 build is reported at
about 25.9 GB, which still exceeds a 24 GB RTX 4090 before KV cache and runtime
overhead. **Drama 3 does not run on a single RTX 4090.** aivoice's planner says
so (`aivoice experimental plan --mock-4090`), and the model manager warns before
downloading it on a machine whose GPU is too small.

What works instead:

| Mode | How | Fits a 4090? |
|---|---|---|
| LOCAL | fish-speech `tools/api_server.py` on this machine with **S2-Pro** (4B) | yes (~12 GB, estimate) |
| LOCAL (Drama 3) | same server with Drama 3 | no: needs a >= 48 GB GPU |
| REMOTE | the same server on a bigger GPU box you control (or SGLang-Omni / vLLM-Omni exposing `/v1/tts`) | n/a |
| FISH AUDIO API | hosted `api.fish.audio`, model header `drama-3` | n/a |
| FALLBACK | espeak-ng / Piper / command (local, not streaming) | yes |

## Running a fish-speech server

From the fish-speech repository (commands from its inference docs):

```bash
# S2-Pro on the local GPU
python tools/api_server.py --listen 127.0.0.1:8080 \
    --llama-checkpoint-path checkpoints/s2-pro \
    --decoder-checkpoint-path checkpoints/s2-pro/codec.pth \
    --decoder-config-name modded_dac_vq

# Drama 3 on a >= 48 GB GPU
python tools/api_server.py --listen 0.0.0.0:8080 --llama-checkpoint-path checkpoints/drama-3 --device cuda:0
```

Fish recommends SGLang-Omni for Drama 3 efficiency. aivoice only needs the HTTP
surface (`/v1/health`, `/v1/tts`, `/v1/references/*`).

Weights can be fetched with aivoice's manager into the shared HF cache:

```bash
aivoice models info fishaudio/s2-pro
aivoice models download fishaudio/s2-pro --yes
```

## Pointing aivoice at it

```bash
aivoice experimental tts use fish-server --url http://127.0.0.1:8080 --model s2-pro
aivoice experimental tts health
aivoice experimental tts say "[laughs] That's actually pretty funny." -o test.wav

# remote Drama 3 box
aivoice experimental remote allow http://gpu-box:8080
aivoice experimental tts use fish-server --url http://gpu-box:8080 --model drama-3

# hosted API
export FISH_API_KEY=...   # or: aivoice credentials set fish-audio (system keyring)
aivoice experimental remote allow fish-api
aivoice experimental tts use fish-api --model drama-3
```

`--model` tells aivoice which tag dialect to emit (Drama 3 paired tags vs S2
bracket cues). With `auto` it asks `/v1/tts/defaults`; if the server does not say,
it uses the S2 subset, which Drama 3 also accepts.

## Request details

* Body: fish-speech `ServeTTSRequest` (`text`, `references[{audio,text}]` or
  `reference_id`, `format`, `streaming`, `chunk_length`, `temperature`, `top_p`,
  `latency`, `use_memory_cache`), sent as msgpack when `ormsgpack`/`msgpack` is
  installed, else JSON with base64 audio.
* Streaming: `streaming: true`, `format: wav`. The server sends a WAV header with
  placeholder sizes, then PCM as it is generated. aivoice decodes incrementally
  and plays the first chunk as soon as `audio.prebuffer_ms` has arrived.
* `use_memory_cache: "on"` lets the server reuse the encoded reference audio
  between requests (lower latency for local voice profiles).
* Hosted API: `POST /v1/tts` with `Authorization: Bearer` and `model: drama-3`
  headers, `format: pcm`, `sample_rate: 44100`, streamed response body (same as
  the official `fish-audio-sdk`). The SDK's WebSocket `/v1/tts/live` is not used.
* Cancellation: aivoice stops reading the response and closes it.
* Warmup: `experimental start` synthesizes "Ready." first, which also fixes the
  output sample rate.

## Fallback

When the configured server fails *before* producing audio, a local engine
(espeak-ng/Piper/command) speaks instead. That fallback is never remote, so a
local setup does not switch to the cloud on its own.

## Status

The clients are tested against a protocol-faithful fake server (tests/fakes) that
streams WAV the way fish-speech does. They have **not** been run against real
Drama 3 or S2-Pro weights from this repository's CI environment. Run
`aivoice experimental benchmark` against your server to get real numbers.
