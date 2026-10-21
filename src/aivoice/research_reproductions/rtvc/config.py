"""RT-VC configuration.

Values marked VERIFIED were read from the official repository (Berkeley-Speech-Group/RT-VC @ 9d9a55d,
constructor defaults and inference helpers). The repository ships no training configs, so sizes it
does not fix are marked ASSUMPTION (SPEC.md R-*).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

SAMPLE_RATE = 16000  # VERIFIED
FRAME = 80           # VERIFIED: framesize 80 -> f_model = 200 Hz


@dataclass
class FrontEndConfig:
    n_fft: int = 1024    # VERIFIED (MFCC melkwargs / MelSpectrogram)
    n_mels: int = 128    # VERIFIED
    n_mfcc: int = 24     # VERIFIED (EMA inverter input)
    causal: bool = True  # aivoice deviation R-F1: official helpers use a centred STFT (512-sample lookahead)


@dataclass
class PitchConfig:
    hidden_dim: int = 256      # ASSUMPTION
    kernel_size: int = 3       # ASSUMPTION
    nbins: int = 256           # ASSUMPTION
    fmin: float = 50.0         # ASSUMPTION
    fmax: float = 800.0        # ASSUMPTION
    f_base: float = 440.0      # ASSUMPTION
    forward_expansion: int = 1  # ASSUMPTION
    n_midlayers: int = 1       # ASSUMPTION
    loudness_scale: float = 8.0  # VERIFIED (scale_factor = 8)
    label_sigma_bins: float = 1.5  # ASSUMPTION: Gaussian-blurred pitch targets for the BCE head


@dataclass
class EMAConfig:
    hidden_dim: int = 256  # ASSUMPTION
    kernel_size: int = 3   # ASSUMPTION
    channels: int = 12     # VERIFIED (12 EMA channels)
    teacher: str = "sparc"  # ASSUMPTION R-T1: SPARC articulatory inversion as pseudo-label teacher


@dataclass
class SpeakerConfig:
    hidden_dim: int = 512          # VERIFIED in_channels 512 (WavLM CNN); width ASSUMPTION
    spkr_emb_dim: int = 128        # VERIFIED (Vocoder default)
    dilations: tuple[int, ...] = (1, 2, 4, 8)  # ASSUMPTION
    nstacks: int = 2               # ASSUMPTION
    use_std: bool = True           # ASSUMPTION
    forward_expansion: int = 1
    n_midlayers: int = 1
    freeze_wavlm_cnn: bool = True  # ASSUMPTION
    wavlm_cnn: str = "wavlm-large"  # VERIFIED (WavLMModel.from_pretrained("microsoft/wavlm-large").feature_extractor)


@dataclass
class VocoderConfig:
    hidden_dim: int = 256        # VERIFIED (default)
    nharmonics: int = 80         # VERIFIED
    nbands: int = 257            # VERIFIED
    attenuate: float = 0.01      # VERIFIED
    kernel_size: int = 3         # VERIFIED
    n_blocks: int = 3            # VERIFIED
    forward_expansion: int = 4   # VERIFIED
    reverb_len: int = 1025       # VERIFIED
    f0_norm: float = 500.0       # VERIFIED (f0 / 500)


@dataclass
class LossConfig:
    fft_sizes: tuple[int, ...] = (2048, 1024, 512, 256, 128, 64)  # DDSP multi-scale spectral loss (ASSUMPTION sizes)
    mss: float = 1.0
    mel: float = 0.0             # optional extra mel L1 (off: not in the DDSP recipe)
    pitch_bce: float = 1.0
    periodicity_bce: float = 1.0
    loudness_l1: float = 1.0
    ema_l1: float = 1.0


@dataclass
class AugmentConfig:  # ASSUMPTION R-A1: input robustness augmentation for the causal trackers
    gain_db: float = 6.0
    noise_snr_db: tuple[float, float] = (20.0, 50.0)
    prob: float = 0.5


@dataclass
class RTVCConfig:
    stage: str = "vocoder"       # pitch | ema | vocoder
    segment_s: float = 1.0
    reference_s: float = 3.0
    lr: float = 1e-3                 # ASSUMPTION: DDSP default learning rate
    betas: tuple[float, float] = (0.9, 0.999)
    frontend: FrontEndConfig = field(default_factory=FrontEndConfig)
    pitch: PitchConfig = field(default_factory=PitchConfig)
    ema: EMAConfig = field(default_factory=EMAConfig)
    speaker: SpeakerConfig = field(default_factory=SpeakerConfig)
    vocoder: VocoderConfig = field(default_factory=VocoderConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    test_standin: bool = False


def paper(stage: str = "vocoder") -> RTVCConfig:
    return RTVCConfig(stage=stage)


def tiny(stage: str = "vocoder") -> RTVCConfig:
    c = RTVCConfig(stage=stage, segment_s=0.5, reference_s=0.5)
    c.pitch = replace(c.pitch, hidden_dim=32, nbins=64)
    c.ema = replace(c.ema, hidden_dim=32)
    c.speaker = replace(c.speaker, hidden_dim=32, spkr_emb_dim=16, dilations=(1, 2), nstacks=1)
    c.vocoder = replace(c.vocoder, hidden_dim=32, nharmonics=20, nbands=65, forward_expansion=2, reverb_len=129)
    c.test_standin = True
    return c


PRESETS = {"paper": paper, "tiny": tiny}

_SUB = {"frontend": FrontEndConfig, "pitch": PitchConfig, "ema": EMAConfig, "speaker": SpeakerConfig,
        "vocoder": VocoderConfig, "loss": LossConfig, "augment": AugmentConfig}


def from_dict(d: dict[str, Any]) -> RTVCConfig:
    c = RTVCConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items() if k not in _SUB})
    for k, cls in _SUB.items():
        if k in d:
            setattr(c, k, cls(**{kk: tuple(vv) if isinstance(vv, list) else vv for kk, vv in d[k].items()}))
    return c
