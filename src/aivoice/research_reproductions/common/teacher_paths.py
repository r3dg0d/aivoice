"""Where teacher snapshots live (torch-free, so status / listing work without torch)."""

from __future__ import annotations

import os
from pathlib import Path

# name -> (Hugging Face id, transformers class, license of the weights)
KNOWN = {
    "wavlm-large": ("microsoft/wavlm-large", "WavLMModel", "MIT"),
    "wavlm-base-plus": ("microsoft/wavlm-base-plus", "WavLMModel", "MIT"),
    "hubert-base": ("facebook/hubert-base-ls960", "HubertModel", "Apache-2.0 (transformers port; fairseq MIT)"),
    "w2v-bert-2.0": ("facebook/w2v-bert-2.0", "Wav2Vec2BertModel", "MIT"),
    "emotion2vec": ("emotion2vec/emotion2vec_plus_base", "funasr", "emotion2vec model license (see its card)"),
    "ecapa": ("speechbrain/spkrec-ecapa-voxceleb", "speechbrain", "Apache-2.0"),
    "xvector": ("speechbrain/spkrec-xvect-voxceleb", "speechbrain", "Apache-2.0"),
}


class TeacherUnavailable(RuntimeError):
    pass


def teacher_dir(name: str, explicit: str | Path | None = None) -> Path:
    d = explicit or os.environ.get(f"AIVOICE_TEACHER_{name.upper().replace('-', '_').replace('.', '_')}")
    if not d:
        raise TeacherUnavailable(
            f"teacher {name!r} ({KNOWN.get(name, ('?',))[0]}) not configured: download it yourself "
            f"(e.g. `hf download {KNOWN.get(name, ('?',))[0]} --local-dir DIR`) and pass --teacher {name}=DIR")
    p = Path(d).expanduser()
    if not (p / "config.json").exists():
        raise TeacherUnavailable(f"teacher {name!r}: {p} has no config.json")
    return p


__all__ = ["KNOWN", "TeacherUnavailable", "teacher_dir"]
