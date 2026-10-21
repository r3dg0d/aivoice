"""TVTSyn provenance record."""

from ..common.provenance import ArchConfidence, Assumption, ReproductionInfo, SourceStatus

INFO = ReproductionInfo(
    key="tvtsyn-repro",
    title="TVTSyn: time-varying timbre streaming voice conversion (aivoice reproduction)",
    paper_key="tvtsyn",
    source_status=SourceStatus.PAPER_REPRODUCTION_WITH_ASSUMPTIONS,
    arch_confidence=ArchConfidence.MEDIUM,  # content encoder / VQ / timbre: HIGH (verified); decoder: MEDIUM
    model_name="aivoice/tvtsyn-paper-repro-v0.1",
    official_code="https://github.com/Morris88826/VOSSA (TVTSyn backbone inside the VOSSA release)",
    official_commit="5dd1a5c51a1da531b0d127ff48aea007cc96ed34",
    official_license="CC BY-NC 4.0",
    redistributable_weights=False,
    weights_license_notes="aivoice code is independent (Apache-2.0); weights trained with it inherit the data "
                          "licenses; do not mix with VOSSA (non-commercial) checkpoints",
    sample_rate=16000,
    hop_samples=320,
    lookahead_samples=32 * 320,
    variants={"repro": "official configuration: right_context 4 in all 8 encoder layers (exact streaming = 640 ms)",
              "aivoice": "tvtsyn-aivoice: lookahead 4 in the first layer only (exact streaming = 80 ms); NOT the paper"},
    assumptions=[
        Assumption("encoder lookahead", "per-layer right_context 4 as in the official config (T-S1)",
                   "official streaming step buffers only 4 frames, which matches offline only approximately"),
        Assumption("conditional LayerNorm", "per-frame channel LayerNorm (T-D3)",
                   "official GroupNorm(1, C) normalises over time too (non-causal)"),
        Assumption("waveform decoder", "causal SEANet decoder 64 filters, ratios 8,5,4,2", "width not verified"),
        Assumption("discriminator", "HiFi-GAN MPD + MSD (T-L2)", "not verified"),
        Assumption("pitch / energy targets", "token-averaged log-F0 (YIN) and log frame RMS", "units not verified"),
        Assumption("timbre reference", "x-vector + ECAPA of another utterance of the same speaker", "not verified"),
    ],
    notes="PAPER-REPORTED: < 80 ms GPU latency. Verified against VOSSA code: encoder layout and sizes, factorised "
          "VQ (4096 x 8, L2, commitment 0.15), GTM (48 tokens x 128, MLP 256 + priors), RoPE-on-query cross attention, "
          "gate MLP, Slerp, 704-d global embedding, frame synthesizer stages 3/3/2 with conditional LN, causal pitch "
          "/ energy predictors (k 7) with token-averaged targets.",
)


def model_name(variant: str, stage: str) -> str:
    base = "aivoice/tvtsyn-paper-repro-v0.1" if variant == "repro" else "aivoice/tvtsyn-aivoice-v0.1"
    return base if stage == "synth" else f"{base}-content"
