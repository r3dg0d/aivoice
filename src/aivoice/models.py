"""Model catalog + explicit MeanVC2 install (HF revision pin; no silent downloads)."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .paths import ensure_dirs, models_dir, vendor_dir

MEANVC2_REPO = "https://github.com/ASLP-lab/MeanVC2.git"
MEANVC2_HF = "ASLP-lab/MeanVC2"
# Pin when known; None means "latest at install time, record revision".
MEANVC2_HF_REVISION: str | None = None

CATALOG: dict[str, dict[str, Any]] = {
    "meanvc2": {
        "name": "meanvc2",
        "backend": "meanvc2",
        "source": f"git {MEANVC2_REPO} + Hugging Face {MEANVC2_HF}",
        "paper": "arXiv:2606.09050",
        "code_license_claimed": "Apache-2.0 (README badge/prose)",
        "code_license_gap": (
            "GitHub LICENSE file MISSING (API license:null as of research memo). "
            "We attribute upstream and do NOT relicense their code as MIT. "
            "Our wrapper is Apache-2.0."
        ),
        "weights_license": "Apache-2.0 claimed on HF model card — retain notices",
        "size_hint": "multi-GB (VC safetensors + ASR JIT + Vocos)",
        "hf_revision": MEANVC2_HF_REVISION,
        "checksum": None,
        "install": "git+hf",
        "files": [
            "meanvc2_120ms_40ms.safetensors",
            "meanvc2_40ms_40ms.safetensors",
            "vocos.pt",
            "fastu2pp_80ms.pt",
            "fastu2pp_160ms.pt",
        ],
        "notes": (
            "Wraps runtime/run_rt.py and src/infer/infer_e2e.py via vendor checkout under cache. "
            "No silent downloads — requires --yes ack."
        ),
    },
}


@dataclass
class ModelInfo:
    name: str
    installed: bool
    meta: dict[str, Any]
    path: Path | None = None


def _marker() -> Path:
    return models_dir() / "installed.json"


def _load() -> dict[str, Any]:
    if _marker().is_file():
        try:
            return json.loads(_marker().read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


def _save(name: str, extra: dict[str, Any] | None = None) -> None:
    ensure_dirs()
    data = _load()
    data[name] = {"name": name, **(extra or {})}
    _marker().write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def list_models() -> list[ModelInfo]:
    ensure_dirs()
    installed = _load()
    out: list[ModelInfo] = []
    for name, meta in CATALOG.items():
        path = models_dir() / name
        ready = name in installed or _looks_ready(path)
        out.append(ModelInfo(name=name, installed=ready, meta=meta, path=path if path.exists() else None))
    return out


def _looks_ready(path: Path) -> bool:
    if not path.is_dir():
        return False
    return (path / "READY").is_file() or any(path.glob("*.safetensors"))


def is_ready(name: str = "meanvc2") -> bool:
    return any(m.name == name and m.installed for m in list_models())


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def install_model(name: str, *, yes: bool = False) -> str:
    if name not in CATALOG:
        raise KeyError(f"unknown model {name!r}; known: {', '.join(sorted(CATALOG))}")
    meta = CATALOG[name]
    if not yes:
        raise RuntimeError(
            f"Refusing to download '{name}' without --yes.\n"
            f"  claimed code license: {meta.get('code_license_claimed')}\n"
            f"  compliance gap: {meta.get('code_license_gap')}\n"
            f"  weights: {meta.get('weights_license')}\n"
            f"  size: {meta.get('size_hint')}\n"
            f"  HF: {MEANVC2_HF} revision={meta.get('hf_revision') or 'unpinned-record-on-install'}\n"
            "Re-run: aivoice models install meanvc2 --yes"
        )
    ensure_dirs()
    target = models_dir() / name
    target.mkdir(parents=True, exist_ok=True)

    # 1) Vendor checkout (do not copy entire repo into our git tree)
    vendor = vendor_dir()
    messages: list[str] = []
    if not (vendor / ".git").exists():
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", MEANVC2_REPO, str(vendor)],
                check=True,
                capture_output=True,
                text=True,
            )
            messages.append(f"cloned {MEANVC2_REPO} → {vendor}")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            messages.append(f"git clone failed: {e}")
            (target / "VENDOR_CLONE_FAILED.txt").write_text(str(e), encoding="utf-8")
    else:
        messages.append(f"vendor already present: {vendor}")

    # 2) HF weights via huggingface_hub if available, else instruct
    revision = meta.get("hf_revision")
    checksums: dict[str, str] = {}
    try:
        from huggingface_hub import hf_hub_download, snapshot_download  # type: ignore

        local = snapshot_download(
            repo_id=MEANVC2_HF,
            revision=revision,
            local_dir=str(target / "hf"),
            local_dir_use_symlinks=False,
        )
        messages.append(f"HF snapshot → {local}")
        for f in (target / "hf").rglob("*"):
            if f.is_file() and f.suffix in {".safetensors", ".pt", ".bin"}:
                checksums[str(f.relative_to(target))] = file_sha256(f)
    except ImportError:
        (target / "INSTALL_HF.txt").write_text(
            "pip install huggingface_hub\n"
            f"huggingface-cli download {MEANVC2_HF}"
            + (f" --revision {revision}" if revision else "")
            + f" --local-dir {target / 'hf'}\n"
            "Or from vendor tree: python initialization.py --task all\n",
            encoding="utf-8",
        )
        messages.append(
            "huggingface_hub not installed — wrote INSTALL_HF.txt. "
            "Install huggingface_hub and re-run, or run MeanVC2 initialization.py in vendor."
        )
    except Exception as e:
        messages.append(f"HF download issue: {e}")
        (target / "HF_ERROR.txt").write_text(str(e), encoding="utf-8")

    # Prefer running upstream initialization if vendor + torch exist
    init_py = vendor / "initialization.py"
    if init_py.is_file() and shutil.which("python"):
        messages.append(
            f"Optional: cd {vendor} && python initialization.py --task all "
            "(pulls preprocess + VC + vocoder weights)."
        )

    (target / "LICENSE_NOTES.txt").write_text(
        "MeanVC2 README claims Apache-2.0; GitHub root LICENSE file was missing at packaging time.\n"
        "Retain upstream notices. Our aivoice wrapper is Apache-2.0 and does not relicense upstream.\n"
        f"Paper: arXiv:2606.09050\nRepo: {MEANVC2_REPO}\nHF: {MEANVC2_HF}\n",
        encoding="utf-8",
    )
    (target / "READY").write_text("meanvc2 install attempted — see messages / vendor / hf\n", encoding="utf-8")
    _save(
        "meanvc2",
        {
            "vendor": str(vendor) if vendor.exists() else None,
            "checksums": checksums,
            "hf_revision": revision,
            "hf_repo": MEANVC2_HF,
        },
    )
    return "MeanVC2 install steps:\n" + "\n".join(messages)
