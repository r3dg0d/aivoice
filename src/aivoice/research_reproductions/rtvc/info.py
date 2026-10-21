"""RT-VC provenance record."""

from ..common.provenance import (
    ArchConfidence,
    Assumption,
    ReproductionInfo,
    SourceStatus,
)

INFO = ReproductionInfo(
    key="rtvc",
    title="RT-VC: Real-Time Zero-Shot Voice Conversion with Speech Articulatory Coding (aivoice reconstruction)",
    paper_key="rtvc",
    source_status=SourceStatus.PARTIAL_OFFICIAL_CODE,
    arch_confidence=ArchConfidence.MEDIUM,
    model_name="aivoice/rtvc-paper-repro-v0.1",
    official_code="https://github.com/Berkeley-Speech-Group/RT-VC",
    official_commit="9d9a55d0fb6cd53ebba108a0bab3380b1fc3c066",
    official_license="none (no LICENSE file: all rights reserved by default)",
    redistributable_weights=False,
    weights_license_notes="official repo has no license and no weights; aivoice re-implements the published layout "
                          "(no code copied) and never vendors the official files",
    sample_rate=16000,
    hop_samples=80,
    lookahead_samples=0,
    variants={"causal-frontend": "left-padded STFT front-end (zero lookahead); official helpers are centred"},
    assumptions=[
        Assumption("front-end", "causal STFT (R-F1); log-mel input to the pitch tracker (R-F3)",
                   "official centred STFT has 32 ms lookahead; linear mel power is badly scaled"),
        Assumption("f0 upsampling", "causal Hann OLA as in the official realtime path, also in training (R-F2)",
                   "official offline path interpolates linearly with lookahead"),
        Assumption("EMA pseudo-labels", "SPARC articulatory inversion (Cho et al.), 50 Hz -> 200 Hz linear",
                   "the repo has no training code; RT-VC builds on speech articulatory coding", ArchConfidence.MEDIUM),
        Assumption("pitch / periodicity targets", "aivoice YIN (or user-supplied tracks, e.g. CREPE)",
                   "teacher not in the repo", ArchConfidence.LOW),
        Assumption("tracker widths, nbins, f-range", "hidden 256, 256 bins, 50-800 Hz, base 440", "not in the repo"),
        Assumption("source median in streaming", "from enrolling the user's own voice (R-S1)",
                   "official computes it over the whole source file"),
        Assumption("augmentation", "random gain +-6 dB and white noise 20-50 dB SNR on tracker inputs (R-A1)",
                   "not in the repo"),
    ],
    notes="PAPER-REPORTED (abstract): 61.4 ms latency on CPU, 13.3 % lower latency than SOTA streaming VC. "
          "Official components reconstructed: MFCC/mel front-end parameters, causal conv layouts, EMA inverter, "
          "pitch tracker heads + decoding, speaker encoder (WavLM-large CNN + periodicity-weighted pooling), "
          "DDSP vocoder (FiLM heads, sin/cos harmonic banks, LTV-FIR noise, speaker reverb).",
)
