"""Model catalog + explicit MeanVC2 install (HF revision pin; no silent downloads)."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .paths import ensure_dirs, models_dir, vendor_dir

MEANVC2_REPO = "https://github.com/ASLP-lab/MeanVC2.git"
MEANVC2_COMMIT = "13acf84c1bf135ea5edad9c245b345289b06b33e"  # main, 2026-09-13
MEANVC2_HF = "ASLP-lab/MeanVC2"
MEANVC2_HF_REVISION: str | None = "39cdd19522fe896c227da691314d9a0e3b995486"  # 2026-08-03
S3PRL_CKPTS_REVISION = "8cad0b370e7e35f8d56951d95d2be036ea85510c"

_HF = f"https://huggingface.co/{MEANVC2_HF}/resolve/{MEANVC2_HF_REVISION}"
# Files the upstream runtime (runtime/run_rt.py) opens, relative to the vendor checkout.
# sha256 = Hugging Face LFS digest at the pinned revision; None = recorded at first verified download.
MEANVC2_FILES: list[dict[str, Any]] = [
    {"path": "ckpts/pretrained_models/meanvc2_120ms_40ms.safetensors", "url": f"{_HF}/meanvc2_120ms_40ms.safetensors",
     "size": 70839448, "sha256": "04ea21a4fb55400469e6004ae842c3e7f059f759e92946fb95123418cad5d818", "format": "safetensors"},
    {"path": "ckpts/pretrained_models/meanvc2_40ms_40ms.safetensors", "url": f"{_HF}/meanvc2_40ms_40ms.safetensors",
     "size": 70839448, "sha256": "01caafec9a3991a5514df9412952d24ae3370406358a01e20e03df41f2f5d515", "format": "safetensors"},
    {"path": "ckpts/vocos/vocos.pt", "url": f"{_HF}/vocos.pt", "size": 33223674,
     "sha256": "df1f2ba9f7ac35c96832579421ee1ed913a68c3a1d2f8a6536739f902f194a93", "format": "torchscript"},
    {"path": "preprocess/ckpts/fastu2pp_80ms.pt", "url": f"{_HF}/fastu2pp_80ms.pt", "size": 246292087,
     "sha256": "4338739bd13e0f7718276373e9547ce1f9aa02ec0867452860df9e837f90e4d2", "format": "torchscript"},
    {"path": "preprocess/ckpts/fastu2pp_160ms.pt", "url": f"{_HF}/fastu2pp_160ms.pt", "size": 246275643,
     "sha256": "9a372c1b7178f564a15da1cd36901d7d0c9081a2fabf53cb6cc0d971deca4928", "format": "torchscript"},
    # WavLM-Large + ECAPA-TDNN speaker verification model; upstream's own Google Drive link (initialization.py)
    {"path": "preprocess/ckpts/wavlm_large_finetune.pth",
     "url": "https://drive.usercontent.google.com/download?id=1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP&export=download&confirm=t",
     "size": 1301926579, "sha256": None, "format": "torch-zip"},
]
# Only its ``cfg`` dict is needed (wavlm_large_cfg.pt); upstream's microsoft/unilm release URL returns 404
# (checked 2026-10-22), so take s3prl's hosted copy at a pinned revision and discard it after extraction.
WAVLM_LARGE = {"path": "wavlm_large.pt",
               "url": f"https://huggingface.co/s3prl/converted_ckpts/resolve/{S3PRL_CKPTS_REVISION}/wavlm_large.pt",
               "size": 1261965425, "sha256": "6fb4b3c3e6aa567f0a997b30855859cb81528ee8078802af439f7b2da0bf100f"}

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
        "size_hint": "2.0 GB on disk (VC 2x71 MB, ASR 2x246 MB, Vocos 33 MB, speaker model 1.3 GB); 1.26 GB extra "
                     "transient download for the WavLM config",
        "hf_revision": MEANVC2_HF_REVISION,
        "checksum": "HF LFS SHA-256 pinned per file; speaker model digest recorded on first download",
        "install": "pinned git checkout + pinned HF revision + Drive (upstream link), staged and verified",
        "files": [f["path"] for f in MEANVC2_FILES] + ["preprocess/ckpts/wavlm_large_cfg.pt"],
        "notes": (
            "Wraps runtime/run_rt.py and src/infer/infer_e2e.py via vendor checkout under cache. "
            "No silent downloads — requires --yes ack."
        ),
    },
}


XVC_REPO, XVC_COMMIT = "https://github.com/Jerrister/X-VC", "49df8c591eafc48b096e466d96f9839f9c0dd739"
CATALOG["xvc"] = {
    "name": "xvc",
    "backend": "xvc",
    "source": f"git {XVC_REPO}@{XVC_COMMIT[:7]} + HF chenxie95/X-VC, zai-org/glm-4-voice-tokenizer + ModelScope ERes2Net",
    "paper": "arXiv:2604.12456",
    "code_license_claimed": "MIT (LICENSE file present at the pinned commit)",
    "code_license_gap": "none for code; check the weights' model cards (not reachable from the build environment)",
    "weights_license": "see huggingface.co/chenxie95/X-VC and the GLM-4-Voice tokenizer / ERes2Net model cards",
    "size_hint": "~2-3 GB (GLM-4-Voice tokenizer is a Whisper-large-v3 encoder with VQ; xvc.pt; ERes2Net)",
    "hf_revision": None,
    "checksum": None,
    "install": "git+hf+modelscope",
    "files": ["xvc.pt", "glm-4-voice-tokenizer/", "speech_eres2net_sv_en_voxceleb_16k/"],
    "notes": ("Runs in its own Python 3.10 venv (torch 2.5.1, transformers 4.44.1 pinned upstream): "
              "set AIVOICE_XVC_PYTHON. xvc.pt is loaded with weights_only=True unless engines.xvc.trust_pickle=true."),
}


RVC_HF = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/e6d0c1a17da07c33557852f9dfa2bd44cc75737d"
# Official RVC-Project base models (HF lj1995/VoiceConversionWebUI @ e6d0c1a, LFS SHA-256). rvc-python would
# otherwise fetch them unverified from a third-party mirror on first use.
RVC_BASE_FILES = [
    {"path": "hubert_base.pt", "url": f"{RVC_HF}/hubert_base.pt", "size": 189507909,
     "sha256": "f54b40fd2802423a5643779c4861af1e9ee9c1564dc9d32f54f20b5ffba7db96", "format": "torch-zip"},
    {"path": "rmvpe.pt", "url": f"{RVC_HF}/rmvpe.pt", "size": 181184272,
     "sha256": "6d62215f4306e3ca278246188607209f09af3dc77ed4232efdd069798c4ec193", "format": "torch-zip"},
]
RVC_PRETRAINED_G48K = {"path": "f0G48k.pth", "url": f"{RVC_HF}/pretrained_v2/f0G48k.pth", "size": 75465569,
                       "sha256": "b5d51f589cc3632d4eae36a315b4179397695042edc01d15312e1bddc2b764a4", "format": "torch-zip"}
RVC_PRETRAINED_V1_G40K = {"path": "v1-f0G40k.pth", "url": f"{RVC_HF}/pretrained/f0G40k.pth", "size": 72909665,
                          "sha256": "9115654aeef1995f7dd3c6fc4140bebbef0ca9760bed798105a2380a34299831", "format": "torch-zip"}
CATALOG["rvc-base"] = {
    "name": "rvc-base", "backend": "rvc", "source": "HF lj1995/VoiceConversionWebUI @ e6d0c1a (RVC-Project official)",
    "paper": "RVC (no paper)", "code_license_claimed": "MIT (RVC-Project)", "code_license_gap": "none",
    "weights_license": "MIT + the repo's added research-use / user-responsibility clause",
    "size_hint": "371 MB (hubert_base.pt 190 MB, rmvpe.pt 181 MB)", "hf_revision": "e6d0c1a17da07c33557852f9dfa2bd44cc75737d",
    "checksum": "HF LFS SHA-256 pinned", "install": "into the rvc-python package's base_model/ of the RVC venv",
    "files": [f["path"] for f in RVC_BASE_FILES],
    "notes": "Feature extractor (HuBERT) and pitch model (RMVPE) every RVC voice uses. Placed where rvc-python looks, so "
             "it never downloads them itself.",
}
CATALOG["rvc-testvoice"] = {
    "name": "rvc-testvoice", "backend": "rvc", "source": "HF lj1995/VoiceConversionWebUI pretrained_v2/f0G48k.pth @ e6d0c1a",
    "paper": "RVC (no paper)", "code_license_claimed": "MIT (RVC-Project)", "code_license_gap": "none",
    "weights_license": "MIT + the repo's added research-use / user-responsibility clause",
    "size_hint": "148 MB download -> two ~55 MB fp16 voices (v2 48k, v1 40k)", "hf_revision": "e6d0c1a17da07c33557852f9dfa2bd44cc75737d",
    "checksum": "HF LFS SHA-256 pinned", "install": "converted to RVC inference format and added as voice 'rvc-pretrained-base'",
    "files": ["rvc-pretrained-base.pth", "rvc-pretrained-base-v1.pth"],
    "notes": "TEST ASSETS, not cloned voices: RVC's own v2 48 kHz and v1 40 kHz pretrained generators (the starting point for training), "
             "saved like RVC saves a trained model. It exercises the full live RVC path (HuBERT, RMVPE, net_g, SOLA) "
             "for latency and stability tests; its timbre is a generic speaker, not anyone in particular.",
}

# RVC model configs (RVC-Project configs/v2/48k.json, configs/v1/40k.json), in the order RVC stores them in a voice .pth.
RVC_V2_48K_CONFIG = [1025, 36, 192, 192, 768, 2, 6, 3, 0, "1", [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                     [12, 10, 2, 2], 512, [24, 20, 4, 4], 109, 256, 48000]
RVC_V1_40K_CONFIG = [1025, 32, 192, 192, 768, 2, 6, 3, 0, "1", [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                     [10, 10, 2, 2], 512, [16, 16, 4, 4], 109, 256, 40000]


def rvc_python_dir(python: str | None = None) -> Path:
    """rvc_python package dir inside the RVC venv (where it looks for base_model/)."""
    py = python or os.environ.get("AIVOICE_RVC_PYTHON") or str(Path.home() / ".local/share/aivoice/rvc-venv/bin/python")
    r = subprocess.run([py, "-c", "import importlib.util,os;print(os.path.dirname(importlib.util.find_spec('rvc_python').origin))"],
                       capture_output=True, text=True, timeout=60, check=False)
    if r.returncode != 0 or not r.stdout.strip():
        raise RuntimeError(f"rvc-python not importable by {py}; build the RVC venv first "
                           "(scripts/aivoice-runtime/build-rvc-venv.sh)")
    return Path(r.stdout.strip())


def _install_rvc_base(progress=None) -> str:
    from .model_fetch import check_format, fetch, sha256

    base = rvc_python_dir() / "base_model"
    msgs, files = [], {}
    for f in RVC_BASE_FILES:
        dest = base / f["path"]
        if dest.is_file() and dest.stat().st_size == f["size"] and sha256(dest) == f["sha256"]:
            msgs.append(f"ok (already verified): {dest}")
        else:
            fetch(f, dest, progress)
            msgs.append(f"{f['path']}: {check_format(dest, f['format'])}")
        files[f["path"]] = {"sha256": f["sha256"], "size": f["size"], "url": f["url"], "dest": str(dest)}
    _save("rvc-base", {"files": files, "dir": str(base)})
    return "RVC base models installed and verified:\n" + "\n".join(msgs)


def make_rvc_inference_model(pretrained: Path, out: Path, *, version: str = "v2") -> Path:
    """Save an RVC training generator checkpoint the way RVC saves a finished model (fp16, no posterior encoder)."""
    import torch

    ck = torch.load(str(pretrained), map_location="cpu", weights_only=True)
    sd = ck.get("model", ck)
    weight = {k: v.half() for k, v in sd.items() if "enc_q" not in k}
    if not any(k.startswith("dec.") for k in weight) or "emb_g.weight" not in weight:
        raise RuntimeError(f"{pretrained} does not look like an RVC generator")
    width = weight["enc_p.emb_phone.weight"].shape[-1]
    if width != (768 if version == "v2" else 256):
        raise RuntimeError(f"{pretrained}: phone embedding is {width}-dim, not an RVC {version} generator")
    cfg, sr, src = (RVC_V2_48K_CONFIG, "48k", "pretrained_v2/f0G48k") if version == "v2" else (RVC_V1_40K_CONFIG, "40k", "pretrained/f0G40k")
    torch.save({"weight": weight, "config": cfg, "info": f"aivoice test voice from RVC {src}", "sr": sr, "f0": 1,
                "version": version}, str(out))
    return out


def _install_rvc_testvoice(progress=None) -> str:
    from .model_fetch import fetch, sha256

    target = models_dir() / "rvc-testvoice"
    target.mkdir(parents=True, exist_ok=True)
    files, lines = {}, []
    for spec, name, ver in ((RVC_PRETRAINED_G48K, "rvc-pretrained-base.pth", "v2"),
                            (RVC_PRETRAINED_V1_G40K, "rvc-pretrained-base-v1.pth", "v1")):
        src = fetch(spec, target / ".staging" / spec["path"], progress)
        out = make_rvc_inference_model(src, target / name, version=ver)
        files[out.name] = {"sha256": sha256(out), "size": out.stat().st_size, "version": ver, "derived_from": spec}
        lines.append(f"  RVC {ver}: {out}\n    aivoice voices import-rvc {out} --name {out.stem} --no-synthesize")
    shutil.rmtree(target / ".staging", ignore_errors=True)
    _save("rvc-testvoice", {"files": files})
    return ("RVC test voices written (RVC's generic pretrained generators, for engine tests only):\n" + "\n".join(lines))


TVTSYN_HF = "warisqr007/TVTSyn"
TVTSYN_REVISION = "e37063244a413f834fe976c1ba356d2ef37ece6c"
TVTSYN_REPO, TVTSYN_COMMIT = "https://github.com/PSI-TAMU/TVTSyn", "9aa67c0272620caaf43170fe98ec48438e1624e0"
_TVT = f"https://huggingface.co/{TVTSYN_HF}/resolve/{TVTSYN_REVISION}"
TVTSYN_FILES = [
    {"path": "config.json", "url": f"{_TVT}/config.json", "size": 47577,
     "sha256": "de7b6e3ddfe4cf0b6f6604c636cacd634a4d31b5a3d99777499cafb7be6aaa03"},
    {"path": "model.safetensors", "url": f"{_TVT}/model.safetensors", "size": 359528880,
     "sha256": "0ae428fc384bbb0f10e97bef403813fb3e51ec4ea82307a25ee77820355d0f61", "format": "safetensors"},
]
CATALOG["tvtsyn"] = {
    "name": "tvtsyn", "backend": "tvtsyn", "source": f"HF {TVTSYN_HF} @ {TVTSYN_REVISION[:7]} + {TVTSYN_REPO} @ {TVTSYN_COMMIT[:7]}",
    "paper": "arXiv:2602.09389 (ICLR 2026)", "code_license_claimed": "CC BY-NC 4.0 (LICENSE in repo)", "code_license_gap": "none",
    "weights_license": "CC BY-NC 4.0: NON-COMMERCIAL use only", "size_hint": "360 MB + SpeechBrain x-vector/ECAPA (~100 MB)",
    "hf_revision": TVTSYN_REVISION, "checksum": "HF LFS SHA-256 pinned", "install": "pinned files + own Python venv (torch<2.9)",
    "files": [f["path"] for f in TVTSYN_FILES],
    "notes": "Official TVTSyn streaming VC (content-synchronous time-varying timbre). 100 ms steps with 80 ms lookahead, "
             "16 kHz. Runs in its own venv (AIVOICE_TVTSYN_PYTHON, default ~/.local/share/aivoice/tvtsyn-venv): "
             "pip install -e '<TVTSyn checkout>[pytorch,speaker]'.",
}


def tvtsyn_dir() -> Path:
    return models_dir() / "tvtsyn"


def _install_tvtsyn(progress=None) -> str:
    """Pinned TVTSyn bundle into models/tvtsyn/bundle, then the pinned SpeechBrain encoders via the engine venv."""
    from .model_fetch import check_format, fetch

    bundle = tvtsyn_dir() / "bundle"
    msgs = []
    for f in TVTSYN_FILES:
        dest = fetch(f, bundle / f["path"], progress)
        msgs.append(f"{f['path']}: " + (check_format(dest, f["format"]) if f.get("format") else "sha256 ok"))
    py = os.environ.get("AIVOICE_TVTSYN_PYTHON") or str(Path.home() / ".local/share/aivoice/tvtsyn-venv/bin/python")
    if not Path(py).exists():
        raise RuntimeError(f"TVTSyn runtime missing: {py}. Create a Python 3.10-3.12 venv with torch<2.9 and "
                           f"pip install -e '<{TVTSYN_REPO} checkout @ {TVTSYN_COMMIT[:7]}>[pytorch,speaker]'")
    code = ("import sys,json;from tvtsyn_inference.speaker import SpeakerEncoder;"
            "c=json.load(open(sys.argv[1]));SpeakerEncoder(device='cpu',cache_dir=sys.argv[2],models=c['speaker_models']);print('ok')")
    r = subprocess.run([py, "-c", code, str(bundle / "config.json"), str(tvtsyn_dir() / "speaker")],
                       capture_output=True, text=True, timeout=900, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"SpeechBrain speaker encoders failed to download/load: {(r.stderr or r.stdout)[-800:]}")
    msgs.append("SpeechBrain x-vector + ECAPA encoders (revisions pinned in config.json) cached")
    _save("tvtsyn", {"files": {f["path"]: {"sha256": f["sha256"], "size": f["size"], "url": f["url"]} for f in TVTSYN_FILES},
                     "hf_revision": TVTSYN_REVISION, "code_commit": TVTSYN_COMMIT, "dir": str(tvtsyn_dir())})
    return "TVTSyn installed and verified (CC BY-NC 4.0, non-commercial):\n" + "\n".join(msgs)


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
    tmp = _marker().with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, _marker())


def list_models() -> list[ModelInfo]:
    ensure_dirs()
    installed = _load()
    out: list[ModelInfo] = []
    for name, meta in CATALOG.items():
        path = models_dir() / name
        if name == "meanvc2":
            ready = meanvc2_files_present()
        elif name == "tvtsyn":
            ready = all((tvtsyn_dir() / "bundle" / f["path"]).is_file() for f in TVTSYN_FILES)
        elif name in ("rvc-base", "rvc-testvoice"):
            ready = all(Path(f.get("dest") or path / k).is_file() for k, f in (installed.get(name, {}).get("files") or {"_": {"dest": "/nonexistent"}}).items())
        else:
            ready = _looks_ready(path)
        out.append(ModelInfo(name=name, installed=ready, meta=meta, path=path if path.exists() else None))
    return out


def _looks_ready(path: Path) -> bool:
    if not path.is_dir():
        return False
    return (path / "READY").is_file() or any(path.glob("*.safetensors"))


def meanvc2_files_present() -> bool:
    """Every file run_rt.py opens is present with its expected size (integrity: ``models verify``)."""
    v = vendor_dir()
    return (v / "runtime" / "run_rt.py").is_file() and (v / "preprocess/ckpts/wavlm_large_cfg.pt").is_file() and all(
        (v / f["path"]).is_file() and (v / f["path"]).stat().st_size == f["size"] for f in MEANVC2_FILES)


def _check_files(rows: list[tuple[Path, int | None, str | None, str, str | None]]) -> tuple[bool, list[dict[str, Any]]]:
    """(path, size, sha256, trust, format) -> verified rows. Hashes every file; existence alone is not evidence."""
    from .model_fetch import FetchError, check_format, sha256

    out, ok = [], True
    for path, size, want, trust, fmt in rows:
        row: dict[str, Any] = {"path": str(path), "trust": trust}
        if not path.is_file():
            row.update(ok=False, reason="missing")
        elif size is not None and path.stat().st_size != size:
            row.update(ok=False, reason=f"size {path.stat().st_size} != {size}")
        elif want is None:
            row.update(ok=False, reason="no digest recorded; reinstall")
        else:
            got = sha256(path)
            row.update(ok=got == want, sha256=got, reason=None if got == want else f"SHA-256 mismatch (expected {want})")
            if row["ok"] and fmt:
                try:
                    row["format"] = check_format(path, fmt)
                except FetchError as e:
                    row.update(ok=False, reason=str(e))
        ok &= bool(row["ok"])
        out.append(row)
    return ok, out


def verify_model(name: str) -> dict[str, Any]:
    """Re-hash installed files against pinned or recorded digests. A READY marker is not evidence."""
    rec = _load().get(name, {}).get("files", {})
    if name == "meanvc2":
        rows = [(vendor_dir() / f["path"], f["size"], f["sha256"] or (rec.get(f["path"]) or {}).get("sha256"),
                 "hf-lfs-pinned" if f["sha256"] else "recorded", f["format"]) for f in MEANVC2_FILES]
        ok, out = _check_files(rows)
        cfg = vendor_dir() / "preprocess/ckpts/wavlm_large_cfg.pt"
        out.append({"path": str(cfg), "ok": cfg.is_file(), "trust": "derived (s3prl wavlm_large.pt cfg)"})
        ok &= cfg.is_file()
        head = subprocess.run(["git", "-C", str(vendor_dir()), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=False).stdout.strip()
        return {"name": name, "ok": ok and head == MEANVC2_COMMIT, "vendor_commit": head, "expected_commit": MEANVC2_COMMIT,
                "files": out}
    if name == "tvtsyn":
        ok, out = _check_files([(tvtsyn_dir() / "bundle" / f["path"], f["size"], f["sha256"], "hf-lfs-pinned", f.get("format"))
                                for f in TVTSYN_FILES])
        return {"name": name, "ok": ok, "files": out}
    if name == "rvc-base":
        base = rvc_python_dir() / "base_model"
        ok, out = _check_files([(base / f["path"], f["size"], f["sha256"], "hf-lfs-pinned", f["format"]) for f in RVC_BASE_FILES])
        return {"name": name, "ok": ok, "files": out}
    if name == "rvc-testvoice":
        d = models_dir() / "rvc-testvoice"
        ok, out = _check_files([(d / n, r.get("size"), r.get("sha256"), f"recorded (derived from pinned RVC {r.get('version', 'v2')} "
                                 "pretrained generator)", "torch-zip") for n, r in rec.items()] or
                               [(d / "rvc-pretrained-base.pth", None, None, "recorded", "torch-zip")])
        return {"name": name, "ok": ok, "files": out}
    raise KeyError(f"verify supports meanvc2, tvtsyn, rvc-base, rvc-testvoice (got {name!r})")


def is_ready(name: str = "meanvc2") -> bool:
    return any(m.name == name and m.installed for m in list_models())


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _install_xvc(yes: bool) -> str:
    meta = CATALOG["xvc"]
    if not yes:
        raise RuntimeError(
            "Refusing to download 'xvc' without --yes.\n"
            f"  code: {meta['code_license_claimed']}\n  weights: {meta['weights_license']}\n  size: {meta['size_hint']}\n"
            f"  pinned code commit: {XVC_COMMIT}\nRe-run: aivoice models install xvc --yes"
        )
    from .paths import cache_home

    ensure_dirs()
    msgs: list[str] = []
    repo = cache_home() / "vendor" / "X-VC"
    target = models_dir() / "xvc"
    target.mkdir(parents=True, exist_ok=True)
    if not (repo / ".git").exists():
        try:
            subprocess.run(["git", "clone", XVC_REPO, str(repo)], check=True, capture_output=True, text=True)
            msgs.append(f"cloned {XVC_REPO} -> {repo}")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            msgs.append(f"git clone failed: {e}")
    if (repo / ".git").exists():
        r = subprocess.run(["git", "-C", str(repo), "checkout", "-q", XVC_COMMIT], capture_output=True, text=True, check=False)
        msgs.append(f"checked out {XVC_COMMIT[:7]}" if r.returncode == 0 else f"checkout failed: {r.stderr.strip()}")
    checksums: dict[str, str] = {}
    try:
        from huggingface_hub import snapshot_download  # type: ignore

        snapshot_download(repo_id="chenxie95/X-VC", local_dir=str(target), allow_patterns=["*.pt", "*.md", "*.json", "*.yaml"])
        snapshot_download(repo_id="zai-org/glm-4-voice-tokenizer", local_dir=str(target / "glm-4-voice-tokenizer"))
        for f in target.rglob("*"):
            if f.is_file() and f.suffix in {".pt", ".safetensors", ".bin"}:
                checksums[str(f.relative_to(target))] = file_sha256(f)
        msgs.append(f"Hugging Face weights -> {target}")
    except ImportError:
        msgs.append("huggingface_hub not installed: pip install 'aivoice[hf]' and re-run")
    except Exception as e:  # noqa: BLE001
        msgs.append(f"Hugging Face download failed: {e}")
    eres = target / "speech_eres2net_sv_en_voxceleb_16k"
    try:
        from modelscope import snapshot_download as ms_download  # type: ignore

        ms_download("iic/speech_eres2net_sv_en_voxceleb_16k", local_dir=str(eres))
        msgs.append(f"ERes2Net (ModelScope) -> {eres}")
    except ImportError:
        msgs.append(f"ERes2Net is on ModelScope: pip install modelscope and re-run, or download "
                    f"iic/speech_eres2net_sv_en_voxceleb_16k into {eres}")
    except Exception as e:  # noqa: BLE001
        msgs.append(f"ModelScope download failed: {e}")
    _save("xvc", {"vendor": str(repo), "commit": XVC_COMMIT, "checksums": checksums})
    msgs.append("Runtime: python3.10 -m venv ~/.local/share/aivoice/xvc-venv && pip install -r "
                f"{repo}/requirements.txt; then export AIVOICE_XVC_PYTHON=~/.local/share/aivoice/xvc-venv/bin/python")
    msgs.append("Check: aivoice engines info xvc")
    return "X-VC install steps:\n" + "\n".join(msgs)


def install_model(name: str, *, yes: bool = False, progress=None) -> str:
    if name == "xvc":
        return _install_xvc(yes)
    if name in ("rvc-base", "rvc-testvoice", "tvtsyn"):
        if not yes:
            m = CATALOG[name]
            raise RuntimeError(f"Refusing to download '{name}' without --yes.\n  weights: {m['weights_license']}\n"
                               f"  size: {m['size_hint']}\n  source: {m['source']}\nRe-run: aivoice models install {name} --yes")
        if name == "tvtsyn":
            return _install_tvtsyn(progress)
        return _install_rvc_base(progress) if name == "rvc-base" else _install_rvc_testvoice(progress)
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
    return _install_meanvc2(progress=progress)


def _install_meanvc2(progress=None) -> str:
    from .model_fetch import FetchError, check_format, fetch, sha256

    ensure_dirs()
    vendor = vendor_dir()
    msgs: list[str] = []
    if not (vendor / ".git").exists():
        r = subprocess.run(["git", "clone", "--quiet", MEANVC2_REPO, str(vendor)], capture_output=True, text=True, check=False)
        if r.returncode != 0:
            raise RuntimeError(f"git clone {MEANVC2_REPO} failed: {r.stderr.strip()}")
        msgs.append(f"cloned {MEANVC2_REPO} -> {vendor}")
    head = subprocess.run(["git", "-C", str(vendor), "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    if head != MEANVC2_COMMIT:
        subprocess.run(["git", "-C", str(vendor), "fetch", "--quiet", "origin", MEANVC2_COMMIT], capture_output=True, check=False)
        r = subprocess.run(["git", "-C", str(vendor), "checkout", "--quiet", MEANVC2_COMMIT], capture_output=True, text=True, check=False)
        if r.returncode != 0:
            raise RuntimeError(f"cannot check out pinned MeanVC2 commit {MEANVC2_COMMIT}: {r.stderr.strip()}")
    msgs.append(f"vendor at {MEANVC2_COMMIT[:7]}")

    stage = models_dir() / ".staging" / "meanvc2"
    stage.mkdir(parents=True, exist_ok=True)
    recorded = _load().get("meanvc2", {}).get("files", {})
    files: dict[str, dict[str, Any]] = {}
    try:
        for f in MEANVC2_FILES:
            live = vendor / f["path"]
            spec = dict(f)
            spec["sha256"] = f["sha256"] or (recorded.get(f["path"]) or {}).get("sha256")
            if live.is_file() and live.stat().st_size == f["size"] and spec["sha256"] and sha256(live) == spec["sha256"]:
                msgs.append(f"ok (already verified): {f['path']}")
                files[f["path"]] = {"sha256": spec["sha256"], "size": f["size"], "url": f["url"]}
                continue
            got = fetch(spec, stage / Path(f["path"]).name, progress)
            msgs.append(f"{f['path']}: {check_format(got, f['format'])}")
            digest = spec["sha256"] or sha256(got)
            live.parent.mkdir(parents=True, exist_ok=True)
            os.replace(got, live)
            files[f["path"]] = {"sha256": digest, "size": f["size"], "url": f["url"],
                                "trust": "hf-lfs-pinned" if f["sha256"] else "recorded-first-download"}
        cfg = vendor / "preprocess/ckpts/wavlm_large_cfg.pt"
        if not cfg.is_file():
            import torch

            big = fetch(WAVLM_LARGE, stage / WAVLM_LARGE["path"], progress)
            ck = torch.load(str(big), map_location="cpu", weights_only=True)
            cfg_dict = ck.get("cfg") or ck.get("config")
            if not isinstance(cfg_dict, dict):
                raise RuntimeError("wavlm_large.pt has no cfg dict")
            torch.save(cfg_dict, str(cfg))
            big.unlink()
            msgs.append(f"extracted {cfg.name} from s3prl wavlm_large.pt @ {S3PRL_CKPTS_REVISION[:7]} (then deleted it)")
    except FetchError as e:
        raise RuntimeError(f"{e}\nPartial downloads are kept under {stage}; re-run to resume.") from e
    shutil.rmtree(stage, ignore_errors=True)
    (models_dir() / "meanvc2").mkdir(parents=True, exist_ok=True)
    (models_dir() / "meanvc2" / "LICENSE_NOTES.txt").write_text(
        "MeanVC2 README claims Apache-2.0; the GitHub repository has no LICENSE file (checked 2026-10-22).\n"
        "Retain upstream notices. aivoice is Apache-2.0 and does not relicense upstream code or weights.\n"
        "wavlm_large_finetune.pth: upstream Google Drive link, no license stated.\n"
        f"Paper: arXiv:2606.09050\nRepo: {MEANVC2_REPO} @ {MEANVC2_COMMIT}\nHF: {MEANVC2_HF} @ {MEANVC2_HF_REVISION}\n",
        encoding="utf-8")
    _save("meanvc2", {"vendor": str(vendor), "commit": MEANVC2_COMMIT, "hf_repo": MEANVC2_HF,
                      "hf_revision": MEANVC2_HF_REVISION, "files": files})
    if not meanvc2_files_present():
        raise RuntimeError("MeanVC2 files incomplete after install")
    return "MeanVC2 installed and verified:\n" + "\n".join(msgs)
