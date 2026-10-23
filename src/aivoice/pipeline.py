"""Live / file voice conversion orchestration."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from .backend.factory import create_backend
from .metrics import MetricsTracker, format_metrics
from .paths import vendor_dir
from .presets import Mode, get_mode


def convert_file(
    source: Path,
    reference: Path,
    output: Path,
    *,
    mode: str = "balanced",
    device: str = "cuda",
    backend: str | None = None,
) -> None:
    be = create_backend(backend, mode=mode, device=device)
    be.set_reference(reference)
    be.convert_file(source, output)


def run_live_subprocess(
    reference: Path,
    *,
    mode: str = "balanced",
    device: str = "cuda",
) -> int:
    """Spawn MeanVC2 runtime/run_rt.py --mode realtime when vendor present."""
    vendor = vendor_dir()
    rt = vendor / "runtime" / "run_rt.py"
    if not rt.is_file():
        raise RuntimeError(
            f"Missing {rt}. Install: aivoice models install meanvc2 --yes"
        )
    m: Mode = get_mode(mode)
    model_flag = "40ms" if m.meanvc2_model == "40ms" else "120ms"
    cmd = [
        sys.executable,
        str(rt),
        "--mode",
        "realtime",
        "--model",
        model_flag,
        # run_rt.py only reads CLI flags (MEANVC2_TARGET_WAV/AIVOICE_DEVICE env are ignored upstream)
        "--target-spk",
        str(reference),
        "--device",
        device,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(vendor) + os.pathsep + env.get("PYTHONPATH", "")
    env["MEANVC2_TARGET_WAV"] = str(reference)
    env["AIVOICE_DEVICE"] = device
    env.setdefault("PYTHONWARNINGS", "ignore")
    print(f"spawning: {' '.join(cmd)}", flush=True)
    print(f"reference={reference} mode={mode} device={device}", flush=True)
    proc = subprocess.Popen(cmd, cwd=str(vendor), env=env)
    _route_to_virtual_sink(proc)
    try:
        return proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        return proc.wait()


RVC_BLOCKS = {  # mode -> (block s, extra context s, crossfade s, lookahead s); see docs/benchmarks/rvc-presets.md
    # lowest-latency: 100 ms block + 40 ms right context. Measured on 6 LibriSpeech speakers (RTX 4090):
    # 180 ms algorithmic and WER 8.9 % vs the old (0.15, 0.4, 0.04): 200 ms and 11.6 % (better on 5/6 speakers).
    "lowest-latency": (0.10, 0.5, 0.03, 0.04),
    "balanced": (0.25, 0.6, 0.05, 0.0),
    "best-quality": (0.35, 1.0, 0.08, 0.0),
}
RT_DEFAULTS = {"engine": "rvc", "pitch": 0.0, "index_rate": 0.5, "threshold": -50.0, "protect": 0.33}


def voice_rvc_files(voice) -> tuple[Path | None, Path | None]:
    """Locate a voice's RVC .pth and (optional) .index: voice dir copies first, then the package."""
    d = voice.dir()
    rvc = (voice.extra or {}).get("rvc") or {}
    pth = d / "source_rvc.pth"
    if not pth.is_file():
        c = rvc.get("checkpoint")
        pth = Path(c) if c and Path(c).is_file() else None
    idx = d / "source_rvc.index"
    if not idx.is_file():
        i = rvc.get("index")
        idx = Path(i) if i and Path(i).is_file() else None
        if idx is None and rvc.get("root") and Path(rvc["root"]).is_dir():
            found = sorted(Path(rvc["root"]).rglob("*.index"), key=lambda q: ("added" not in q.name, q.name))
            idx = found[0] if found else None
    return pth, idx


def voice_rt_settings(voice) -> dict:
    """Per-voice realtime settings (voices/<id>/rt.json). RVC is the default whenever a .pth exists."""
    import json

    cfg = dict(RT_DEFAULTS)
    f = voice.dir() / "rt.json"
    if f.is_file():
        try:
            cfg.update(json.loads(f.read_text()))
        except Exception:
            pass
    if cfg.get("engine") == "rvc" and voice_rvc_files(voice)[0] is None:
        cfg["engine"] = "meanvc2"
    return cfg


def save_voice_rt_settings(voice, **kw) -> dict:
    import json

    f = voice.dir() / "rt.json"
    cur = {}
    if f.is_file():
        try:
            cur = json.loads(f.read_text())
        except Exception:
            cur = {}
    cur.update({k: v for k, v in kw.items() if v is not None})
    f.write_text(json.dumps(cur, indent=2) + "\n")
    return voice_rt_settings(voice)


def run_live_rvc(voice, *, mode: str = "balanced", device: str = "cuda", settings: dict | None = None) -> int:
    """Real-time RVC v2 (rvc/rt_worker.py inside the rvc-python venv), routed like MeanVC2."""
    cfg = settings or voice_rt_settings(voice)
    pth, idx = voice_rvc_files(voice)
    if pth is None:
        raise RuntimeError(f"voice {voice.display_name} has no RVC .pth; use --engine meanvc2")
    py = os.environ.get("AIVOICE_RVC_PYTHON") or str(Path.home() / ".local/share/aivoice/rvc-venv/bin/python")
    if not Path(py).exists():
        raise RuntimeError(f"RVC realtime runtime missing: {py} (set AIVOICE_RVC_PYTHON)")
    block, extra, xfade, lookahead = RVC_BLOCKS.get(mode, RVC_BLOCKS["balanced"])
    worker = Path(__file__).parent / "rvc" / "rt_worker.py"
    cmd = [py, str(worker), "--model", str(pth), "--pitch", str(cfg["pitch"]),
           "--index-rate", str(cfg["index_rate"]), "--protect", str(cfg["protect"]),
           "--threshold", str(cfg["threshold"]), "--block", str(block), "--extra", str(extra),
           "--crossfade", str(xfade), "--lookahead", str(lookahead), "--device", device]
    if idx is not None:
        cmd += ["--index", str(idx)]
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.setdefault("PYTHONWARNINGS", "ignore")
    rundir = Path.home() / ".cache" / "aivoice" / "rvc-rt"
    rundir.mkdir(parents=True, exist_ok=True)
    print(f"spawning: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(cmd, cwd=str(rundir), env=env)
    _route_to_virtual_sink(proc)
    try:
        return proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        return proc.wait()


def _input_source() -> str | None:
    """Real mic to capture: $AIVOICE_INPUT, ~/.config/aivoice/input-source, else the default
    source (ignoring aivoice's own virtual devices / monitors to avoid a feedback loop)."""
    import shutil

    v = os.environ.get("AIVOICE_INPUT", "").strip()
    cfg = Path.home() / ".config" / "aivoice" / "input-source"
    if not v and cfg.is_file():
        v = cfg.read_text().strip()
    if v:
        return v
    pactl = shutil.which("pactl")
    if pactl:
        d = subprocess.run([pactl, "get-default-source"], capture_output=True, text=True).stdout.strip()
        if d and not d.startswith("aivoice") and not d.endswith(".monitor"):
            return d
    return None


def _route_to_virtual_sink(proc, sink: str = "aivoice_sink") -> None:
    """run_rt.py opens ALSA 'default' for both directions. Pin its playback node to the aivoice
    sink and its capture node to the user's real mic, and keep re-asserting (stream-restore /
    WirePlumber can relink) for as long as the process runs."""
    import json
    import shutil
    import threading

    pw_dump, pw_meta, pactl = shutil.which("pw-dump"), shutil.which("pw-metadata"), shutil.which("pactl")
    if not (pw_dump and pw_meta):
        print("[aivoice] warning: pw-dump/pw-metadata missing; output not routed to the virtual mic", flush=True)
        return
    src = _input_source()
    if src and pactl:
        info = subprocess.run([pactl, "list", "sources"], capture_output=True, text=True).stdout
        blk = info.split(f"Name: {src}\n", 1)
        if len(blk) == 2 and "Mute: yes" in blk[1].split("\nSource #", 1)[0]:
            print(f"[aivoice] warning: input {src} is MUTED - unmuting", flush=True)
            subprocess.run([pactl, "set-source-mute", src, "0"], capture_output=True)
    print(f"[aivoice] input source: {src or '(default)'}", flush=True)

    def worker() -> None:
        done: dict[int, str] = {}
        while proc.poll() is None:
            try:
                objs = json.loads(subprocess.run([pw_dump], capture_output=True, text=True, timeout=10).stdout or "[]")
            except Exception:
                objs = []
            clients = {
                o["id"] for o in objs
                if o.get("type", "").endswith("Client")
                and str((o.get("info") or {}).get("props", {}).get("application.process.id")) == str(proc.pid)
            }
            for o in objs:
                props = (o.get("info") or {}).get("props", {})
                if not (o.get("type", "").endswith("Node") and props.get("client.id") in clients):
                    continue
                mc = props.get("media.class")
                tgt = sink if mc == "Stream/Output/Audio" else src if mc == "Stream/Input/Audio" else None
                if tgt and done.get(o["id"]) != tgt:
                    subprocess.run([pw_meta, str(o["id"]), "target.object", tgt], capture_output=True)
                    done[o["id"]] = tgt
                    print(f"[aivoice] pinned {mc} (node {o['id']}) -> {tgt}", flush=True)
            time.sleep(2 if done else 0.5)

    threading.Thread(target=worker, daemon=True).start()


def benchmark_overhead(*, frames: int = 50, mode: str = "balanced") -> str:
    import numpy as np

    m = get_mode(mode)
    mt = MetricsTracker(sample_rate=m.sample_rate)
    chunk = np.zeros(int(m.sample_rate * m.chunk_ms / 1000), dtype=np.float32)
    for _ in range(frames):
        t0 = time.perf_counter()
        # simulate buffer copy overhead only
        _ = chunk.copy()
        ms = (time.perf_counter() - t0) * 1000
        audio_ms = float(m.chunk_ms)
        metrics = mt.record(process_ms=ms, audio_ms=audio_ms, buffer_depth=m.buffer_chunks)
    return format_metrics(metrics) + "\n(Note: full VC RTF needs models install + CUDA preferred)"
