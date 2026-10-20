"""Real-time RVC v2 voice conversion worker (runs inside the rvc-python venv, NOT aivoice's venv).

Follows RVC-Project's realtime GUI design: fixed block + extra left context, RMVPE f0, HuBERT
features blended with the FAISS index, net_g inference, then SOLA alignment + crossfade between
blocks. Audio I/O is sounddevice on the ALSA default devices; aivoice pins the streams
(capture -> real mic, playback -> aivoice_sink) through PipeWire metadata.

Usage: python rt_worker.py --model x.pth [--index x.index] [--pitch 12] [--index-rate 0.5]
       [--block 0.25] [--extra 0.6] [--crossfade 0.05] [--threshold -50] [--input-wav f --output-wav g]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--index", default="")
    ap.add_argument("--pitch", type=float, default=0.0, help="transpose in semitones")
    ap.add_argument("--index-rate", type=float, default=0.5)
    ap.add_argument("--protect", type=float, default=0.33)
    ap.add_argument("--block", type=float, default=0.25, help="block seconds")
    ap.add_argument("--extra", type=float, default=0.6, help="extra left context seconds")
    ap.add_argument("--crossfade", type=float, default=0.05)
    ap.add_argument("--threshold", type=float, default=-50.0, help="noise gate dBFS")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sr", type=int, default=48000)
    ap.add_argument("--input-wav", default=None, help="offline test: stream this wav through the RT loop")
    ap.add_argument("--output-wav", default=None)
    a = ap.parse_args()

    import logging
    import warnings

    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)  # fairseq/rvc_python INFO+WARNING chatter
    import faiss
    import torch
    import torch.nn.functional as F
    import torchaudio.functional as AF
    from rvc_python.infer import RVCInference

    dev = "cuda:0" if a.device == "cuda" and torch.cuda.is_available() else "cpu:0"
    print("RVC v2 Real-Time Voice Conversion", flush=True)
    print(f"  Device:     {'cuda' if dev.startswith('cuda') else 'cpu'}"
          f"{' (' + torch.cuda.get_device_name(0) + ')' if dev.startswith('cuda') else ''}", flush=True)
    print(f"  Model:      {a.model}", flush=True)
    print(f"  Index:      {a.index or '(none)'}  rate={a.index_rate}", flush=True)
    print(f"  Pitch:      {a.pitch:+g} st   block={a.block}s extra={a.extra}s xfade={a.crossfade}s gate={a.threshold}dB",
          flush=True)

    rvc = RVCInference(device=dev)
    rvc.load_model(a.model, version="v2", index_path=a.index)
    vc = rvc.vc
    from rvc_python.modules.vc.utils import load_hubert

    if vc.hubert_model is None:
        vc.hubert_model = load_hubert(vc.config, vc.lib_dir)
    hubert, net_g, pipe = vc.hubert_model, vc.net_g, vc.pipeline
    tgt_sr, if_f0, version = vc.tgt_sr, vc.if_f0, vc.version
    is_half = vc.config.is_half
    print(f"  fp16:       {is_half}   model sr={tgt_sr}  f0={bool(if_f0)}  version={version}", flush=True)

    index = big_npy = None
    if a.index and os.path.exists(a.index) and a.index_rate > 0:
        index = faiss.read_index(a.index)
        big_npy = index.reconstruct_n(0, index.ntotal)

    SR = a.sr
    zc = SR // 100  # 10 ms grid
    block = int(round(a.block * SR / zc)) * zc
    xfade = int(round(a.crossfade * SR / zc)) * zc
    sola_search = zc
    extra = int(round(a.extra * SR / zc)) * zc
    total = extra + xfade + sola_search + block
    buf = np.zeros(total, dtype=np.float32)
    sola_buf = np.zeros(xfade, dtype=np.float32)
    fade_in = (np.sin(0.5 * np.pi * np.linspace(0, 1, xfade)) ** 2).astype(np.float32)
    fade_out = 1 - fade_in
    gate = 10 ** (a.threshold / 20)
    tdev = torch.device(dev)
    stats = {"n": 0, "over": 0}

    sid = torch.tensor([0], device=tdev).long()

    def convert(x48: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(x48).to(tdev)
        x16 = AF.resample(x, SR, 16000)
        x16 = AF.highpass_biquad(x16, 16000, 48.0)
        x16n = x16.float().cpu().numpy()
        p_len = x16n.shape[0] // 160
        pitch = pitchf = None
        if if_f0:
            coarse, f0 = pipe.get_f0("rt", x16n, p_len, a.pitch, "rmvpe", 3)
            pitch = torch.tensor(coarse[:p_len], device=tdev).unsqueeze(0).long()
            pitchf = torch.tensor(f0[:p_len], device=tdev).unsqueeze(0).float()
        out = pipe.vc(hubert, net_g, sid, x16n, pitch, pitchf, [0, 0, 0], index, big_npy,
                      a.index_rate, version, a.protect)
        y = torch.from_numpy(out).to(tdev)
        if tgt_sr != SR:
            y = AF.resample(y, tgt_sr, SR)
        return y.cpu().numpy().astype(np.float32)

    def process(inp: np.ndarray) -> np.ndarray:
        nonlocal buf, sola_buf
        t0 = time.perf_counter()
        buf = np.concatenate([buf[block:], inp])
        rms = float(np.sqrt(np.mean(inp ** 2)) + 1e-9)
        if rms < gate:
            out = np.zeros(block, dtype=np.float32)
            sola_buf[:] = 0
        else:
            y = convert(buf)
            need = block + sola_search + xfade
            y = y[-need:] if y.shape[0] >= need else np.pad(y, (need - y.shape[0], 0))
            # SOLA: best alignment of the new head against the previous tail
            head = y[: xfade + sola_search]
            if np.any(sola_buf):
                cor = np.correlate(head, sola_buf, "valid")
                nrm = np.sqrt(np.convolve(head ** 2, np.ones(xfade), "valid") + 1e-8)
                off = int(np.argmax(cor / nrm))
            else:
                off = 0
            seg = y[off: off + block + xfade]
            seg = seg.copy()
            seg[:xfade] = seg[:xfade] * fade_in + sola_buf * fade_out
            out = seg[:block]
            sola_buf = seg[block: block + xfade].copy()
            if sola_buf.shape[0] < xfade:
                sola_buf = np.pad(sola_buf, (0, xfade - sola_buf.shape[0]))
        ms = (time.perf_counter() - t0) * 1000
        stats["n"] += 1
        if ms > a.block * 1000:
            stats["over"] += 1
        if stats["n"] % 10 == 0:
            print(f"[{stats['n']}] {ms:.1f}ms / {a.block * 1000:.0f}ms "
                  f"{'OK' if ms < a.block * 1000 else 'SLOW'}  over={stats['over']} gate={'open' if rms >= gate else 'shut'}",
                  flush=True)
        return np.clip(out, -1, 1)

    # warm-up (cudnn/rmvpe/hubert)
    for _ in range(2):
        process((np.random.randn(block) * 0.05).astype(np.float32))
    stats.update(n=0, over=0)
    buf[:] = 0
    sola_buf[:] = 0

    if a.input_wav:
        import soundfile as sf

        x, sr = sf.read(a.input_wav, dtype="float32", always_2d=True)
        x = x.mean(1)
        if sr != SR:
            x = AF.resample(torch.from_numpy(x), sr, SR).numpy()
        x = np.pad(x, (0, (-len(x)) % block + block))
        outs = [process(x[i: i + block]) for i in range(0, len(x), block)]
        sf.write(a.output_wav or "rt_out.wav", np.concatenate(outs), SR)
        print(f"wrote {a.output_wav}  chunks={stats['n']} over={stats['over']}", flush=True)
        return 0

    import queue
    import sounddevice as sd

    print("[Stream] Running (sounddevice, 48 kHz mono). Ctrl+C to stop.", flush=True)
    xr = 0

    def cb(indata, outdata, frames, t, status):
        nonlocal xr
        if status:
            xr += 1
            if xr % 20 == 1:
                print(f"[Stream] status: {status} (x{xr})", flush=True)
        outdata[:, 0] = process(indata[:, 0].copy())

    with sd.Stream(samplerate=SR, blocksize=block, channels=1, dtype="float32", latency="low", callback=cb):
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
