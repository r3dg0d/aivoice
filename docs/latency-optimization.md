# Latency optimization

Measure first: `aivoice experimental benchmark` (components + pipeline) and
`aivoice experimental latency` (live engine). Every number below that is not
labelled as measured is a design target or an estimate.

## Where the time goes (endpoint -> first audio)

```
VAD hangover (min_silence_ms) ─▶ ASR finalize ─▶ [LLM] ─▶ TTS time-to-first-audio ─▶ prebuffer ─▶ device
```

| Stage | Knob | Notes |
|---|---|---|
| Endpoint hangover | `vad.min_silence_ms` (300; 220 in ultra-low-latency) | the largest fixed cost; lower values split sentences at pauses |
| Speculation | `stabilizer.speculative` | speaks stable phrases *before* the endpoint, so the hangover overlaps playback |
| Commit size | `stabilizer.min_commit_words`, `commit_on_punctuation`, `agreement` | smaller = earlier but choppier prosody |
| ASR finalize | backend choice | Moonshine flushes a cached stream; Parakeet re-decodes the utterance |
| LLM | `llm.timeout_ms`, `max_tokens`, model size | hard budget; falls back to your words |
| TTS | `tts.chunk_length` (100-300), `tts.latency=balanced`, server `--compile`/`--half` | server-side; aivoice streams the first chunk immediately |
| Prebuffer | `audio.prebuffer_ms` (60) | jitter safety vs. delay; underruns show in `status` |
| Device | pacat `--latency-msec=30`, 10 ms blocks | PipeWire quantum also matters (`pw-metadata -n settings 0 clock.force-quantum 256`) |

## What aivoice does

* **Never blocks capture.** The audio callback only enqueues; a 4 s drop-oldest
  queue absorbs slow ASR passes.
* **Persistent sessions.** ASR, LLM and the TTS connection are created once;
  `experimental start` warms TTS up (first request is often the slowest).
* **Streaming everywhere it is real.** Moonshine is natively streaming; TTS audio
  is decoded and played chunk by chunk; Parakeet is labelled chunked.
* **CPU/GPU partitioning.** Default placement: VAD + ASR (Moonshine) + LLM on CPU,
  TTS on the GPU (or remote). `aivoice experimental plan` shows the plan.
* **Cancellation.** Obsolete segments (barge-in, backlog) are cancelled before
  playback and their HTTP streams closed.
* **Bounded backlog.** Unstarted output older than `audio.max_backlog_ms` is dropped.
* **Server-side caching.** `use_memory_cache: on` for reference audio.

## What aivoice deliberately leaves to the model server

Pinned memory, CUDA streams, CUDA graphs, KV-cache/prefix reuse and quantization
for Drama 3 / S2-Pro live inside fish-speech / SGLang-Omni / vLLM-Omni; that is
where the tensors are. aivoice does not fake them in the client. For Parakeet,
fp16 on CUDA and `torch.inference_mode()` are used.

## Recipes

* Lowest latency: `experimental preset use ultra-low-latency`, Moonshine small,
  S2-Pro on the local GPU with `--compile`, `tts.chunk_length 100`.
* Natural prosody: `balanced`, `min_commit_words 4`, `prebuffer_ms 80`.
* Remote Drama 3: add the network RTT to TTFA; keep `prebuffer_ms` >= 80 on Wi-Fi.

## Cloning engines in real time (Path B)

- **Selecting an engine.** `aivoice experimental tts use engine:voxcpm2 --voice NAME` puts a cloning
  engine behind the real-time TTS interface.
- **Model loading.** The engine worker loads the model once and stays resident, so only the first
  utterance pays the load cost.
- **Streaming.** VoxCPM2 streams through `generate_streaming`. VoiceStar and X-Voice return whole
  utterances, which the real-time status reports as non-streaming.
- **Memory on a 24 GB card.** VoxCPM2 (~8 GB), Moonshine on CPU and an optional CPU LLM fit together.
  `aivoice experimental plan` shows the budget.
- **Per-utterance cost.** Reference preparation happens once per profile, not per utterance.
  Enhancement is off by default in real time, because it adds hundreds of ms of analysis.
