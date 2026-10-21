# X-VC in aivoice

X-VC ([arXiv:2604.12456](https://arxiv.org/abs/2604.12456), SJTU X-LANCE) converts speech in one step inside the
latent space of a pretrained neural codec. aivoice runs the **official code** (`github.com/Jerrister/X-VC`, MIT,
pinned at commit `49df8c5`) in its own interpreter, with the model resident for the whole session.

> **Status here:** integrated and unit-tested, **not run**. The build environment could not reach Hugging Face
> (weights) and has no GPU, so no X-VC audio, latency or VRAM figure in this repository comes from a real run.
> `aivoice engines info xvc` shows the live status on your machine; `aivoice benchmark vc` measures it.
> X-VC is **not** the `auto` default until local benchmarks put it ahead (see [engine-comparison.md](engine-comparison.md)).

## What the model does (from the code)

`bins/infer_utils.py` at the pinned commit:

| Stage | Module | Input → output |
|---|---|---|
| target conditions (once per voice) | `precompute_conditions` | target wav → ERes2Net utterance embedding (192-d) + target mel (frame-level) |
| semantic path | `WhisperVQEncoderWrapper` (GLM-4-Voice tokenizer) → `semantic_adapter` | source chunk → speech tokens → embeddings |
| acoustic path | SAC `acoustic_encoder` → `acoustic_quantizer` | source chunk → codec latents |
| conversion | `prenet` (speaker-conditioned) → `acoustic_converter` (frame condition + adaptive norm) | latents in target timbre |
| decode | `acoustic_decoder` (codec wave generator) | 16 kHz waveform |

Streaming (`run_streaming`) recomputes a fixed **2.4 s** window every `current` ms:
`window = [i·current − history, i·current + current + smooth + future)`, keeps `[history, history+current)` and
crossfades `smooth` ms against the previous window's tail. aivoice's `SlidingWindowStreamer`
(`src/aivoice/vc/stream_worker.py`) cuts the same windows from a ring buffer as microphone audio arrives; a unit
test checks it is sample-identical to upstream's loop for irregular capture blocks.

## Latency (by construction)

| `--mode` | current / future / smooth | algorithmic latency | provenance |
|---|---|---|---|
| `best-quality` | 120 / 100 / 20 ms | **240 ms** | upstream `scripts/batch_infer_seedtts_stream.sh` |
| `balanced` | 80 / 80 / 20 ms | 180 ms | aivoice choice, **not validated by the authors** |
| `lowest-latency` | 60 / 60 / 20 ms | 140 ms | aivoice choice, **not validated by the authors** |

Add compute per step (a Whisper-large-v3-size encoder over 2.4 s every `current` ms is the dominant cost),
resampling (~1 ms) and device buffers. 240 ms is above the 200 ms "undesirable for gaming" line; whether the shorter
settings keep quality must be measured (`aivoice benchmark vc --engines xvc --quality lowest-latency`).

## Install

```bash
aivoice models install xvc --yes   # pinned git checkout + HF weights + (with `pip install modelscope`) ERes2Net
python3.10 -m venv ~/.local/share/aivoice/xvc-venv
~/.local/share/aivoice/xvc-venv/bin/pip install -r ~/.cache/aivoice/vendor/X-VC/requirements.txt
export AIVOICE_XVC_PYTHON=~/.local/share/aivoice/xvc-venv/bin/python
aivoice engines info xvc            # READY once runtime + weights are found
```

Upstream pins `torch==2.5.1` and `transformers==4.44.1`, which is why X-VC gets its own venv. Paths can be overridden
with `settings engines.xvc.{repo_dir,ckpt,glm_tokenizer_dir,eres2net_dir,device,dtype}`.

**Pickle safety.** Upstream loads `xvc.pt` with `torch.load(weights_only=False)`. aivoice's worker forces
`weights_only=True`; if the checkpoint needs pickle code it refuses with an explanation. Only set
`aivoice experimental config set engines.xvc.trust_pickle true` for a checkpoint whose source you trust.

## Use

```bash
aivoice live --engine xvc --voice "Narrator" --consent-ack
aivoice virtualmic --engine xvc --voice "Narrator" --consent-ack   # same, into aivoice-virtual-mic
aivoice file input.wav --engine xvc --voice "Narrator" -o out.wav --consent-ack
aivoice benchmark vc --engines xvc,meanvc2,rvc --voice "Narrator" --consent-ack
```

Capabilities advertised (`aivoice engines info xvc`): zero-shot, reference required, 16 kHz, streaming. Cross-lingual
conversion is reported in the paper but is listed as *not verified*; there is no pitch control in the X-VC code.

## RTX 4090 notes (to verify locally)

- Start with `--mode best-quality` (the authors' setting) and compare `balanced` in `benchmark vc`.
- `engines.xvc.dtype bf16` casts the generator to bfloat16; it is **not** benchmarked here - keep it only if the
  benchmark shows lower p95 at unchanged similarity/WER.
- `torch.compile`, CUDA graphs and TensorRT are not wired up for X-VC: the 2.4 s recompute dominates, and none of them
  could be measured here.
