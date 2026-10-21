"""Provenance labels for paper reproductions.

Every reproduction, checkpoint and number in aivoice carries these labels so nobody can mistake a
reproduction for an official release, self-trained weights for official weights, a paper number for
a local measurement, or an untrained network for a usable one.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class SourceStatus(str, Enum):
    OFFICIAL_CODE = "OFFICIAL_CODE"                      # the authors' code, run as published
    PARTIAL_OFFICIAL_CODE = "PARTIAL_OFFICIAL_CODE"      # authors' code for some parts, aivoice code for the rest
    PAPER_REPRODUCTION = "PAPER_REPRODUCTION"            # aivoice code written from the paper, no gaps that matter
    PAPER_REPRODUCTION_WITH_ASSUMPTIONS = "PAPER_REPRODUCTION_WITH_ASSUMPTIONS"  # paper leaves choices open
    FINETUNED_PUBLIC_BASE = "FINETUNED_PUBLIC_BASE"      # an extension fine-tuned from public weights
    UNTRAINED = "UNTRAINED"
    BLOCKED = "BLOCKED"
    EXPERIMENTAL = "EXPERIMENTAL"


class WeightsStatus(str, Enum):
    OFFICIAL = "OFFICIAL"                                         # released by the authors
    RETRAINED_PAPER_REPRODUCTION = "RETRAINED_PAPER_REPRODUCTION"  # trained by an aivoice user with this code
    FINETUNED_FROM_PUBLIC_BASE = "FINETUNED_FROM_PUBLIC_BASE"
    RANDOM_INIT = "RANDOM_INIT"                                    # never usable for conversion
    NONE = "NONE"


class ArchConfidence(str, Enum):
    HIGH = "HIGH"      # verified against official code or fully specified in the paper
    MEDIUM = "MEDIUM"  # paper specifies the structure, some sizes/details are assumptions
    LOW = "LOW"        # paper only sketches it


class ModelProvenance(str, Enum):
    """Categories shown by `aivoice models info` (and the research panel)."""

    OFFICIAL = "official"
    PAPER_REPRODUCTION = "paper_reproduction"
    AIVOICE_VARIANT = "aivoice_variant"
    LOCALLY_TRAINED = "locally_trained"
    IMPORTED = "imported"


PAPER_REPORTED = "PAPER-REPORTED"
MEASURED = "MEASURED BY AIVOICE"


def paper_reported(value: Any, source: str, **extra: Any) -> dict[str, Any]:
    """A number from a paper: kept, but never mixed with measurements (and never used for routing)."""
    return {"value": value, "label": PAPER_REPORTED, "source": source, **extra}


def measured(value: Any, **extra: Any) -> dict[str, Any]:
    return {"value": value, "label": MEASURED, **extra}


@dataclass
class Assumption:
    """A choice the paper does not pin down. Shown in SPEC.md and `aivoice reproduce info`."""

    what: str
    choice: str
    why: str
    confidence: ArchConfidence = ArchConfidence.MEDIUM


@dataclass
class ReproductionInfo:
    key: str                          # registry name, e.g. "zerovc-repro"
    title: str
    paper_key: str                    # research.catalog key
    source_status: SourceStatus
    arch_confidence: ArchConfidence
    model_name: str                   # name given to weights trained with this code
    official_code: str | None = None  # URL of the authors' code, if any
    official_commit: str | None = None
    official_license: str | None = None
    code_license: str = "Apache-2.0"  # license of the aivoice code in this package
    redistributable_weights: bool = False  # can weights trained with this code be shared? (also depends on data)
    weights_license_notes: str = ""
    sample_rate: int = 16000
    hop_samples: int = 320
    lookahead_samples: int = 0        # algorithmic lookahead of the streaming path, by construction
    variants: dict[str, str] = field(default_factory=dict)
    assumptions: list[Assumption] = field(default_factory=list)
    blocked_by: list[str] = field(default_factory=list)
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("source_status", "arch_confidence"):
            d[k] = getattr(self, k).value
        for a in d["assumptions"]:
            a["confidence"] = a["confidence"].value if isinstance(a["confidence"], Enum) else a["confidence"]
        return d


def weights_label(weights_status: WeightsStatus | str) -> str:
    """One-line label for a checkpoint's weights; RANDOM_INIT is spelled out as unusable."""
    ws = WeightsStatus(weights_status)
    return {
        WeightsStatus.OFFICIAL: "official weights (released by the paper's authors)",
        WeightsStatus.RETRAINED_PAPER_REPRODUCTION: "aivoice paper-reproduction weights (NOT official)",
        WeightsStatus.FINETUNED_FROM_PUBLIC_BASE: "fine-tuned from public weights by an aivoice user (NOT official)",
        WeightsStatus.RANDOM_INIT: "UNTRAINED (random initialisation): not usable for conversion",
        WeightsStatus.NONE: "no weights",
    }[ws]


__all__ = ["ArchConfidence", "Assumption", "MEASURED", "ModelProvenance", "PAPER_REPORTED", "ReproductionInfo",
           "SourceStatus", "WeightsStatus", "measured", "paper_reported", "weights_label"]
