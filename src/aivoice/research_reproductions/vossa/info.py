"""VOSSA provenance: official code and official weights exist, so aivoice does not reproduce it."""

from ..common.provenance import ArchConfidence, ReproductionInfo, SourceStatus

INFO = ReproductionInfo(
    key="vossa-repro",
    title="VOSSA: Voiceprint Optimization for Streaming Speech Architectures (official release)",
    paper_key="vossa",
    source_status=SourceStatus.OFFICIAL_CODE,
    arch_confidence=ArchConfidence.HIGH,
    model_name="morris88826/VOSSA (official weights)",
    official_code="https://github.com/Morris88826/VOSSA",
    official_commit="5dd1a5c51a1da531b0d127ff48aea007cc96ed34",
    official_license="CC BY-NC 4.0 (code and checkpoints)",
    code_license="n/a (official code is used from your checkout, not copied)",
    redistributable_weights=False,
    sample_rate=16000, hop_samples=320, lookahead_samples=4 * 320,
    notes="Interspeech 2026 (Best Student Paper). PAPER-REPORTED: RTF ~0.25, ~73 ms end-to-end latency. Live engine "
          "`vossa` wraps the official VossaStreamer in a worker. Retraining uses the official Lightning CLI.",
)
