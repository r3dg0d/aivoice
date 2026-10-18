# NVIDIA Parakeet TDT 0.6B v3

Backend name: `parakeet`. Hub repo: `nvidia/parakeet-tdt-0.6b-v3`.

## Facts (checked)

* 0.6B-parameter FastConformer-TDT; 25 European languages with automatic language
  detection (bg, hr, cs, da, nl, en, et, fi, fr, de, el, hu, it, lv, lt, mt, pl,
  pt, ro, sk, sl, es, sv, ru, uk).
* `ASRModel.transcribe(..., timestamps=True)` returns word/segment timestamps.
* NeMo's `transcribe()` accepts numpy arrays (checked in nemo_toolkit 3.0.0
  source), so aivoice passes audio straight from memory.
* License: CC-BY-4.0 per the model card.

## Streaming: this is *not* native

The v3 checkpoint is a full-context (offline) model. NeMo documents no
cache-aware streaming mode for it. aivoice therefore:

1. implements only whole-clip `transcribe()` in `ParakeetBackend`
   (`native_streaming=False`), and
2. streams it through `ChunkedStreamingAdapter`: every `interval_ms` (480 ms) of new
   audio it re-decodes the utterance so far (window capped at 20 s) and reports
   the result as a partial.

Every partial costs a full re-decode, and its text can change anywhere, so all
partials go through the transcript stabilizer before anything is spoken. The
interval stretches when a decode takes longer than the audio it covers.
`aivoice experimental status` shows this backend as `chunked re-decoding, not
native streaming`.

## Setup

```bash
# CUDA torch build first (see pytorch.org for your CUDA), then:
pip install 'nemo_toolkit[asr]'
aivoice models download nvidia/parakeet-tdt-0.6b-v3 --yes   # ~2.5 GB .nemo, pinned by commit
aivoice models verify nvidia/parakeet-tdt-0.6b-v3
aivoice models load parakeet                                  # load + warmup test
aivoice experimental asr use parakeet
```

When the snapshot is registered, the backend restores the local `.nemo` with
`ASRModel.restore_from`; otherwise `from_pretrained` resolves through the HF cache.

Security: a `.nemo` file is a tar archive with pickled state. Only load it from
publishers you trust (NVIDIA's own repo here).

## Resources

Estimates (measure them yourself): ~3.5 GB VRAM in fp16 on CUDA, ~5 GB RAM on CPU
(slow). On a 24 GB RTX 4090 it fits next to S2-Pro, but if the TTS server needs
the memory, use Moonshine on CPU instead (`aivoice experimental plan`).

## Status

Unit-tested with a fake NeMo model object. Not run against real weights in this
repository's CI environment (no GPU, Hub unreachable there).
