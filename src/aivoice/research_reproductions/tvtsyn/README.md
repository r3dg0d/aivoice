# TVTSyn: time-varying timbre streaming voice conversion (aivoice reproduction)

* registry name: `tvtsyn-repro` · source status **PAPER_REPRODUCTION_WITH_ASSUMPTIONS** · architecture confidence **MEDIUM**
* weights trained with this code: `aivoice/tvtsyn-paper-repro-v0.1` (never official)
* official code: https://github.com/Morris88826/VOSSA (TVTSyn backbone inside the VOSSA release) (CC BY-NC 4.0)

Layout: `config.py` (presets, every value marked VERIFIED / PAPER / ASSUMPTION), `model/` (encoder, timbre,
decoder, discriminator, losses), `train/` (dataset, preprocess, trainer recipe, checkpoint), `streaming/`
(state, buffers, inference), `eval/`. The paper-to-module map is in [SPEC.md](SPEC.md). Commands and
measured status: [docs/reproductions/tvtsyn.md](../../../../docs/reproductions/tvtsyn.md).
