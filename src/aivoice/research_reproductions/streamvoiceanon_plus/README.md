# StreamVoiceAnon+: emotion-preserving streaming anonymisation (aivoice fine-tuning extension)

* registry name: `streamvoiceanon-plus` · source status **FINETUNED_PUBLIC_BASE** · architecture confidence **MEDIUM**
* weights trained with this code: `aivoice/streamvoiceanon-plus-finetune-v0.1` (never official)
* official code: https://github.com/Plachtaa/StreamVoiceAnon (base model; StreamVoiceAnon+ weights unreleased) (Apache-2.0 (code); weights per the Plachta/StreamVoiceAnon model card)

Layout: `config.py` (presets, every value marked VERIFIED / PAPER / ASSUMPTION), `model/` (encoder, timbre,
decoder, discriminator, losses), `train/` (dataset, preprocess, trainer recipe, checkpoint), `streaming/`
(state, buffers, inference), `eval/`. The paper-to-module map is in [SPEC.md](SPEC.md). Commands and
measured status: [docs/reproductions/streamvoiceanon-plus.md](../../../../docs/reproductions/streamvoiceanon-plus.md).
