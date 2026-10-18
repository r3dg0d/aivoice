# LLM speech transformation (optional)

`SpeechIntelligenceEngine` rewrites recognised speech before synthesis. The
pipeline works without it: `verbatim` mode never calls a model, and every
failure falls back to your own words.

## Modes (`aivoice llm style MODE`)

| Mode | Model needed | What happens |
|---|---|---|
| `verbatim` | no | exact recognised text |
| `cleanup` | optional | filler/stutter removal by rule; the model (if loaded) fixes grammar |
| `natural` | yes | conversational flow, same facts |
| `expressive` | yes | **wording kept** (rule cleanup only); the model chooses emotion + delivery |
| `character` | yes | rephrased as the character (`llm character "..."` or the voice profile's) |
| `custom` | yes | your instruction (`llm prompt "..."`) |
| `hybrid` | optional | rule cleanup + emotion/delivery within the budget |

## Setup

```bash
pip install 'aivoice[llm]'                      # llama-cpp-python (CPU build by default)
aivoice models download qwen2.5-1.5b --yes      # Qwen/Qwen2.5-1.5B-Instruct-GGUF, *q4_k_m.gguf
aivoice llm enable --model qwen2.5-1.5b
aivoice llm style natural
aivoice llm intensity 0.5 --creativity 0.3 --timeout-ms 350
aivoice llm test "um so I I don't know what you're doing"
```

Known small models in the registry: `qwen2.5-0.5b`, `qwen2.5-1.5b`, `qwen3-1.7b`
(aivoice appends `/no_think` to keep it from reasoning), and `gemma-3-1b` (gated:
accept Google's terms on the Hub first). Any local GGUF path works too.

The LLM runs on CPU by default (`llm.n_gpu_layers = 0`), leaving the GPU to TTS.
The `Llama` object persists for the whole session; llama.cpp reuses the KV cache
for the unchanged system-prompt prefix between calls.

## Output contract

The model must return JSON, constrained with llama.cpp's JSON-schema grammar:

```json
{"transformed": "Wait, what exactly are you doing?", "emotion": "confused", "delivery": "curious", "confidence": 0.91}
```

This object is internal. Drama 3 never sees it: the expression compiler turns
`emotion`/`delivery` into the backend's own markup.

## Guards

The original text is used instead when:

* the call exceeds `llm.timeout_ms` (generation is cancelled via a stopping
  criterion, so it does not keep the CPU busy);
* the output is not valid JSON or lacks `transformed`;
* the rewrite **drops or invents a number**, **drops a name**, or **introduces
  new names**;
* a cleanup/hybrid/expressive result is much longer than the input;
* the previous call is still being cancelled.

`max_tokens` (96) limits output. Because an LLM needs the whole sentence, the
engine turns off speculative synthesis while a model-backed mode is active.

## Status

The engine, contract, guards and timeouts are unit-tested with a scripted
backend. In this repository's development environment llama-cpp-python had no
prebuilt wheel (a source build was not attempted) and GGUF downloads from the Hub
were blocked, so real-model latency has not been measured here; `aivoice llm test`
prints it on your machine.
