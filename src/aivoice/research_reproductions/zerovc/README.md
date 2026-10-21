# Zero-VC: Zero-Lookahead Streaming Voice Conversion via Speaker Anonymization (aivoice reproduction)

* registry name: `zerovc-repro` · source status **PAPER_REPRODUCTION_WITH_ASSUMPTIONS** · architecture confidence **MEDIUM**
* weights trained with this code: `aivoice/zerovc-paper-repro-v0.1` (never official)
* official code: none (n/a)

Layout: `config.py` (presets, every value marked VERIFIED / PAPER / ASSUMPTION), `model/` (encoder, timbre,
decoder, discriminator, losses), `train/` (dataset, preprocess, trainer recipe, checkpoint), `streaming/`
(state, buffers, inference), `eval/`. The paper-to-module map is in [SPEC.md](SPEC.md). Commands and
measured status: [docs/reproductions/zerovc.md](../../../../docs/reproductions/zerovc.md).
