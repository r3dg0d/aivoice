"""Registry of paper reproductions (what `aivoice reproduce`, the VC engine list and the research panel use).

Every entry is EXPERIMENTAL and never chosen by ``--engine auto``. A live engine for a reproduction is
only startable with a *trained* checkpoint (RANDOM_INIT / stand-in checkpoints are refused).
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any

from .common.provenance import ModelProvenance

# registry name -> (module, extra config preset or None)
ENTRIES: dict[str, tuple[str, str | None]] = {
    "rtvc": ("aivoice.research_reproductions.rtvc", None),
    "zerovc-repro": ("aivoice.research_reproductions.zerovc", None),
    "tvtsyn-repro": ("aivoice.research_reproductions.tvtsyn", None),
    "tvtsyn-aivoice": ("aivoice.research_reproductions.tvtsyn", "aivoice"),
    "streamvoiceanon-plus": ("aivoice.research_reproductions.streamvoiceanon_plus", None),
    "vossa-repro": ("aivoice.research_reproductions.vossa", None),
}
ALIASES = {"zerovc": "zerovc-repro", "tvtsyn": "tvtsyn-repro", "rt-vc": "rtvc", "vossa": "vossa-repro",
           "svanon-plus": "streamvoiceanon-plus", "streamvoiceanon+": "streamvoiceanon-plus"}

PROVENANCE = {
    "rtvc": ModelProvenance.PAPER_REPRODUCTION, "zerovc-repro": ModelProvenance.PAPER_REPRODUCTION,
    "tvtsyn-repro": ModelProvenance.PAPER_REPRODUCTION, "tvtsyn-aivoice": ModelProvenance.AIVOICE_VARIANT,
    "streamvoiceanon-plus": ModelProvenance.PAPER_REPRODUCTION, "vossa-repro": ModelProvenance.OFFICIAL,
}


def resolve(name: str) -> str:
    n = name.lower()
    n = ALIASES.get(n, n)
    if n not in ENTRIES:
        raise KeyError(f"unknown reproduction {name!r}; known: {', '.join(ENTRIES)}")
    return n


def get(name: str):
    """The Reproduction object for a registry name."""
    key = resolve(name)
    mod = importlib.import_module(ENTRIES[key][0])
    return mod.REPRODUCTION


def default_preset(name: str) -> str:
    return ENTRIES[resolve(name)][1] or "paper"


def repro_home() -> Path:
    from aivoice.paths import cache_home

    return Path(os.environ.get("AIVOICE_REPRO_HOME") or cache_home() / "reproductions")


def checkpoint_for(name: str) -> Path | None:
    """The inference checkpoint a live engine would use: AIVOICE_<NAME>_CHECKPOINT or <repro_home>/<name>/model.pt."""
    key = resolve(name)
    env = os.environ.get(f"AIVOICE_{key.upper().replace('-', '_')}_CHECKPOINT")
    p = Path(env) if env else repro_home() / key / "model.pt"
    return p if p.exists() else None


def summary(name: str) -> dict[str, Any]:
    key = resolve(name)
    r = get(key)
    info = r.info.as_dict()
    return {"name": key, "provenance": PROVENANCE[key].value, "title": info["title"],
            "source_status": info["source_status"], "arch_confidence": info["arch_confidence"],
            "model_name": info["model_name"] if key != "tvtsyn-aivoice" else "aivoice/tvtsyn-aivoice-v0.1",
            "official_code": info["official_code"], "official_license": info["official_license"],
            "stages": list(r.stages), "checkpoint": str(checkpoint_for(key) or "") or None}


__all__ = ["ALIASES", "ENTRIES", "PROVENANCE", "checkpoint_for", "default_preset", "get", "repro_home", "resolve",
           "summary"]
