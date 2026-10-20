# Model backends

`aivoice engines capabilities` prints this table from the code, so it cannot drift from what the
engines actually accept:

| Engine | TTS | VC | Zero-shot | No transcript | Cross-lingual | Style prompt | Duration | Streaming | Adapt | Rate |
|---|---|---|---|---|---|---|---|---|---|---|
| voxcpm2 | yes | - | yes | yes | yes | yes | - | yes | yes (LoRA) | 48 kHz |
| xvoice | yes | - | yes | yes | yes | - | yes | - | - | 24 kHz |
| voicestar | yes | - | yes | - (needs transcript) | - | - | yes | - | - | 16 kHz |
| meanvc2 | - | yes | yes | yes | yes | - | - | yes | - | 16 kHz |
| fish-server | yes | - | yes | - | yes | - | - | yes | - | 44 kHz |
| fish-api | yes | - | yes | - | yes | yes | - | yes | - | 44 kHz |
| espeak | yes | - | - | - | - | - | - | - | - | 22 kHz |

No engine supports a separate style *reference* recording, so `--style-reference` is refused everywhere.

## Default preference order

Measurements on your machine (`aivoice benchmark research`) take precedence. Without them the router
picks the first available engine in this order:

`voxcpm2 > xvoice > voicestar > fish-server > fish-api > meanvc2 > espeak`

`--priority latency|similarity|naturalness|multilingual|expressiveness` re-ranks using the measured
metric. Remote engines (`fish-api`, remote `fish-server`) are used only with `--consent-ack`.

## Installing an engine

Each engine lives in its own virtualenv, because their torch and transformers pins conflict:

```bash
python -m venv ~/.venvs/voxcpm && ~/.venvs/voxcpm/bin/pip install voxcpm==2.0.3
aivoice models download voxcpm2            # openbmb/VoxCPM2, commit-pinned, asks before downloading
aivoice experimental config set engines.voxcpm2.python ~/.venvs/voxcpm/bin/python
# or: export AIVOICE_VOXCPM2_PYTHON=~/.venvs/voxcpm/bin/python

python -m venv ~/.venvs/voicestar && ~/.venvs/voicestar/bin/pip install voicestar==0.1.0
aivoice models download voicestar          # pyp1/VoiceStar (weights CC-BY-4.0)
aivoice experimental config set engines.voicestar.python ~/.venvs/voicestar/bin/python

python -m venv ~/.venvs/xvoice && ~/.venvs/xvoice/bin/pip install x-voice==0.1.1
aivoice models download xvoice             # XRXRX/X-Voice
aivoice experimental config set engines.xvoice.python ~/.venvs/xvoice/bin/python

aivoice engines list                       # READY column shows what will run
```

To use weights stored somewhere else, set `engines.<engine>.model_path`.

## Notes per engine

- **VoxCPM2.**
  - Clones from `reference_wav_path` alone; a transcript is optional and enables continuation mode.
  - `--style` becomes VoxCPM's `(description)text` prefix.
  - The official README reports about 8 GB VRAM and RTF ≈ 0.3 on an RTX 4090. aivoice has not measured
    that.
- **X-Voice.**
  - Transcript-free Stage-2 cloning with its own speaking-rate predictor.
  - aivoice calls its CLI, so the model is reloaded on every call. It is suitable for offline
    synthesis, not real time.
  - Code-switched text is tagged `[main|lang]` by `research/phonetics.py`.
- **VoiceStar.**
  - Needs a reference transcript (otherwise the package downloads Whisper itself).
  - Its default target duration equals the reference length. aivoice passes a silence-aware plan instead
    (see [research-fusion.md](research-fusion.md)).
- **MeanVC2.** Voice conversion only; the existing real-time path is unchanged.
- **Fish.** The existing S2-Pro / Drama 3 clients. Drama 3 needs ≥ 48 GB of GPU memory.

## RTX 4090 (24 GB) guidance

- **What the router enforces.** The router refuses an engine whose estimated VRAM (`est_vram_mib`) is
  more than the free memory it detects.
- **Which engines fit.** VoxCPM2 (~8 GB), VoiceStar (~5 GB) and X-Voice (~3 GB) each fit alongside
  Moonshine ASR. Drama 3 does not fit.
- **What was not measured.** None of these numbers was measured by aivoice in this environment, which
  has no GPU. Run `aivoice benchmark research` to record your own. The router then uses those
  measurements.
