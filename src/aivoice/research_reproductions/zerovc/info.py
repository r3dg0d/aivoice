"""Zero-VC provenance record."""

from ..common.provenance import (
    ArchConfidence,
    Assumption,
    ReproductionInfo,
    SourceStatus,
)

INFO = ReproductionInfo(
    key="zerovc-repro",
    title="Zero-VC: Zero-Lookahead Streaming Voice Conversion via Speaker Anonymization (aivoice reproduction)",
    paper_key="zerovc",
    source_status=SourceStatus.PAPER_REPRODUCTION_WITH_ASSUMPTIONS,
    arch_confidence=ArchConfidence.MEDIUM,
    model_name="aivoice/zerovc-paper-repro-v0.1",
    official_code=None,
    official_license=None,
    redistributable_weights=False,
    weights_license_notes="teachers: WavLM-large (MIT), w2v-BERT 2.0 (MIT), HuBERT-base (Apache-2.0); data decides",
    sample_rate=16000,
    hop_samples=320,
    lookahead_samples=0,
    variants={"A": "content student distilled from w2v-BERT 2.0 (closest to the paper)",
              "B": "aivoice practical variant: content student trained on HuBERT-base L9 k-means-200 codes (NOT the paper)"},
    assumptions=[
        Assumption("SA module", "McAdams LPC warping (alpha 0.7-0.9) or user-supplied anonymised copies",
                   "the paper's SA system is not identified in the text available to aivoice"),
        Assumption("content student", "causal SEANet CNN (strides 2,4,5,8) + 8-layer windowed causal transformer",
                   "paper says 'distilled streaming w2v-bert-2.0' without the student architecture",
                   ArchConfidence.LOW),
        Assumption("w2v-BERT 2.0 layer", "hidden_states[16]", "layer not given", ArchConfidence.LOW),
        Assumption("decoder", "HiFi-GAN V1 widths, upsample (8,5,4,2), causal convs, global timbre added per stage",
                   "paper: 'mainly based on HiFi-GAN with critical modifications for zero-lookahead streaming'"),
        Assumption("timbre embedding", "256-d after attention pooling (hidden 256)", "dimension not given"),
        Assumption("optimiser / losses", "HiFi-GAN: AdamW 2e-4 (0.8, 0.99), mel 45, fm 2, adv 1",
                   "paper defers to the HiFi-GAN adversarial framework"),
        Assumption("training crop / reference", "2 s source crop, 3 s reference from another utterance of the speaker",
                   "not stated"),
    ],
    notes="Paper-reported numbers (search-indexed text, not re-measured): SS-S 0.171, SS-R 0.521, WER 3.96 %, "
          "RTF 0.063 on an Intel Xeon Platinum 8468V CPU; trained on LibriTTS (~460 h after removing < 4 s).",
)


def model_name(variant: str, stage: str) -> str:
    base = "aivoice/zerovc-paper-repro-v0.1" if variant == "A" else "aivoice/zerovc-variantB-v0.1"
    return base if stage == "vc" else f"{base}-content"
