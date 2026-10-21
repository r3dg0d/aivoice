"""StreamVoiceAnon+ provenance record."""

from ..common.provenance import (
    ArchConfidence,
    Assumption,
    ReproductionInfo,
    SourceStatus,
)

INFO = ReproductionInfo(
    key="streamvoiceanon-plus",
    title="StreamVoiceAnon+: emotion-preserving streaming anonymisation (aivoice fine-tuning extension)",
    paper_key="streamvoiceanon-plus",
    source_status=SourceStatus.FINETUNED_PUBLIC_BASE,
    arch_confidence=ArchConfidence.MEDIUM,
    model_name="aivoice/streamvoiceanon-plus-finetune-v0.1",
    official_code="https://github.com/Plachtaa/StreamVoiceAnon (base model; StreamVoiceAnon+ weights unreleased)",
    official_commit="201705182c045298225071481e7cd59d537e935e",
    official_license="Apache-2.0 (code); weights per the Plachta/StreamVoiceAnon model card",
    redistributable_weights=False,
    sample_rate=44100,
    hop_samples=2048,
    lookahead_samples=2 * 2048,
    variants={},
    assumptions=[
        Assumption("lambda", "lambda_emo = 1.0", "value not available to aivoice", ArchConfidence.LOW),
        Assumption("distillation head", "2-layer MLP (512 hidden) on the Slow-AR output (post-norm)",
                   "head shape and layer not available", ArchConfidence.LOW),
        Assumption("SEP tokens", "two learned embeddings appended after the neutral prompt (semantic, acoustic)",
                   "matches the described semantic/acoustic boundary tokens; exact placement unverified"),
        Assumption("teacher alignment", "emotion2vec frames (50 Hz) mean-pooled to codec frames (21.5 Hz)",
                   "frame alignment not specified"),
        Assumption("loss region", "Slow/Fast-AR losses on the source only (prompt + SEP passed as prefix)",
                   "prompt tokens are conditioning, as in prompted inference"),
        Assumption("fine-tuning lr", "1e-5 from the public base", "not specified"),
    ],
    notes="Extension, not a rebuild: the official dual-AR model, encoders and decoder are imported from your "
          "StreamVoiceAnon checkout. The emotion teacher and head are training only; inference is the official path.",
)
