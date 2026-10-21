# RT-VC: Real-Time Zero-Shot Voice Conversion with Speech Articulatory Coding (aivoice reconstruction)

* registry name: `rtvc` · source status **PARTIAL_OFFICIAL_CODE** · architecture confidence **MEDIUM**
* weights trained with this code: `aivoice/rtvc-paper-repro-v0.1` (never official)
* official code: https://github.com/Berkeley-Speech-Group/RT-VC (none (no LICENSE file: all rights reserved by default))

Layout: `config.py` (presets, every value marked VERIFIED / PAPER / ASSUMPTION), `model/` (encoder, timbre,
decoder, discriminator, losses), `train/` (dataset, preprocess, trainer recipe, checkpoint), `streaming/`
(state, buffers, inference), `eval/`. The paper-to-module map is in [SPEC.md](SPEC.md). Commands and
measured status: [docs/reproductions/rtvc.md](../../../../docs/reproductions/rtvc.md).
