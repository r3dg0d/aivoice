# Live RVC v1/v2

RVC voices (a `.pth` trained on one speaker, optionally a FAISS `.index`) convert live with the model kept loaded.
This is not zero-shot: each voice needs its own trained model. RVC has no paper; it is a community project
(RVC-Project WebUI, rvc-python, Applio).

## Two ways to run it

| | `aivoice virtualmic --voice V` (default for RVC voices) | `--engine rvc --unified` / `aivoice live --engine rvc` |
|---|---|---|
| process | `rvc/rt_worker.py` inside the rvc-python venv, with its own sounddevice I/O | same `RvcRealtime` block processor, driven by aivoice's shared pipeline |
| routing | PipeWire metadata pins capture → your mic, playback → `aivoice_sink` | aivoice captures and plays into `aivoice_sink` itself |
| fallback | none (process exit = silence) | DSP modulation / passthrough when the worker dies |
| status | log lines | `aivoice vc status`, QuickShell metrics |
| post-modulation | no | `--modulate warm` etc. |

Both share the block processor (block + extra left context + RMVPE F0 + HuBERT features blended with the index +
`net_g`, then SOLA alignment and a sin² crossfade), introduced in commit c7d5db8.

## Settings

Per voice in `voices/<id>/rt.json` (`aivoice voices rt NAME --pitch 12 --index-rate 0.6`), or per run with
`--pitch` / `--index-rate`.

| `--mode` | block | extra context | crossfade | algorithmic latency |
|---|---|---|---|---|
| lowest-latency | 150 ms | 400 ms | 40 ms | ~200 ms |
| balanced | 250 ms | 600 ms | 50 ms | ~310 ms |
| best-quality | 350 ms | 1000 ms | 80 ms | ~440 ms |

Latency = block + crossfade + 10 ms SOLA search, before compute and device buffers. Shorter blocks raise the
per-block overhead (HuBERT and RMVPE run on `block + extra` every block); measure with
`aivoice benchmark vc --engines rvc --voice V --quality lowest-latency`.

## Setup

`scripts/aivoice-runtime/build-rvc-venv.sh` builds `~/.local/share/aivoice/rvc-venv` (Python 3.10, torch 2.4.1 cu121,
rvc-python). Override with `AIVOICE_RVC_PYTHON`. `aivoice engines list` reports `rvc INSTALLED` once the runtime is
found; a voice is usable when it has a `.pth` (`aivoice voices rt NAME` shows `rvc_model`).

## Security

An RVC `.pth` is a Python pickle and rvc-python loads it with code execution. Only use models you trust; aivoice
prints this before synthesis and never downloads models silently. (Applio's realtime path loads `.pth` with
`weights_only=True`; switching aivoice's RVC runtime to it would remove that risk and is a candidate for a later
release - it was read, not integrated.)

## Not measured here

No RVC model or rvc-python runtime was available in the build environment, so RVC latency, VRAM and similarity in
this repository are unmeasured. On the owner's machine the c7d5db8 path was run by the user.
