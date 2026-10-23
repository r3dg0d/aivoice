"""Real-time RVC v2 voice conversion worker (runs inside the rvc-python venv, NOT aivoice's venv).

Follows RVC-Project's realtime GUI design: fixed block + extra left context, RMVPE f0, HuBERT
features blended with the FAISS index, net_g inference, then SOLA alignment + crossfade between
blocks. Audio I/O is sounddevice on the ALSA default devices; aivoice pins the streams
(capture -> real mic, playback -> aivoice_sink) through PipeWire metadata.

The block processor is :class:`RvcRealtime`, so the same code also serves aivoice's unified
streaming worker (vc/stream_worker.py ``rvc`` handler) without this file's own audio I/O.
It must keep importing nothing from aivoice.

Usage: python rt_worker.py --model x.pth [--index x.index] [--pitch 12] [--index-rate 0.5]
       [--block 0.25] [--extra 0.6] [--crossfade 0.05] [--threshold -50] [--input-wav f --output-wav g]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np


class RvcRealtime:
    """Persistent RVC model + streaming state. ``process(block)`` takes exactly ``self.block`` samples at ``sr``."""

    def __init__(self, *, model: str, index: str = "", pitch: float = 0.0, index_rate: float = 0.5, protect: float = 0.33,
                 block: float = 0.25, extra: float = 0.6, crossfade: float = 0.05, threshold: float = -50.0,
                 device: str = "cuda", sr: int = 48000, log: bool = True, lookahead: float = 0.0) -> None:
        import logging
        import warnings

        warnings.filterwarnings("ignore")
        logging.disable(logging.WARNING)  # fairseq/rvc_python INFO+WARNING chatter
        import faiss
        import torch
        import torchaudio.functional as AF
        from rvc_python.infer import RVCInference

        self.torch, self.AF = torch, AF
        self.dev = "cuda:0" if device == "cuda" and torch.cuda.is_available() else "cpu:0"
        self.pitch, self.index_rate, self.protect = float(pitch), float(index_rate), float(protect)
        self.block_s = float(block)
        self.log = log
        rvc = RVCInference(device=self.dev)
        rvc.load_model(model, version="v2", index_path=index)
        vc = rvc.vc
        from rvc_python.modules.vc.utils import load_hubert

        if vc.hubert_model is None:
            vc.hubert_model = load_hubert(vc.config, vc.lib_dir)
        self.hubert, self.net_g, self.pipe = vc.hubert_model, vc.net_g, vc.pipeline
        self.tgt_sr, self.if_f0, self.version = vc.tgt_sr, vc.if_f0, vc.version
        self.is_half = vc.config.is_half

        self.index = self.big_npy = None
        if index and os.path.exists(index) and index_rate > 0:
            self.index = faiss.read_index(index)
            self.big_npy = self.index.reconstruct_n(0, self.index.ntotal)

        self.SR = sr
        zc = sr // 100  # 10 ms grid
        self.block = int(round(block * sr / zc)) * zc
        self.xfade = int(round(crossfade * sr / zc)) * zc
        self.sola_search = zc
        self.extra = int(round(extra * sr / zc)) * zc
        # Right context: the emitted segment ends `lookahead` before the newest sample, so HuBERT/net_g see a
        # little future instead of converting the window's edge (adds exactly `lookahead` latency).
        self.lookahead = int(round(lookahead * sr / zc)) * zc
        self.total = self.extra + self.xfade + self.sola_search + self.block + self.lookahead
        self.fade_in = (np.sin(0.5 * np.pi * np.linspace(0, 1, self.xfade)) ** 2).astype(np.float32)
        self.fade_out = 1 - self.fade_in
        self.gate = 10 ** (threshold / 20)
        self.tdev = torch.device(self.dev)
        self.sid = torch.tensor([0], device=self.tdev).long()
        self.stats = {"n": 0, "over": 0}
        self.reset()

    def reset(self) -> None:
        self.buf = np.zeros(self.total, dtype=np.float32)
        self.sola_buf = np.zeros(self.xfade, dtype=np.float32)

    def convert(self, x48: np.ndarray) -> np.ndarray:
        torch, AF = self.torch, self.AF
        x = torch.from_numpy(x48).to(self.tdev)
        x16 = AF.resample(x, self.SR, 16000)
        x16 = AF.highpass_biquad(x16, 16000, 48.0)
        x16n = x16.float().cpu().numpy()
        p_len = x16n.shape[0] // 160
        pitch = pitchf = None
        if self.if_f0:
            coarse, f0 = self.pipe.get_f0("rt", x16n, p_len, self.pitch, "rmvpe", 3)
            pitch = torch.tensor(coarse[:p_len], device=self.tdev).unsqueeze(0).long()
            pitchf = torch.tensor(f0[:p_len], device=self.tdev).unsqueeze(0).float()
        out = self.pipe.vc(self.hubert, self.net_g, self.sid, x16n, pitch, pitchf, [0, 0, 0], self.index, self.big_npy,
                           self.index_rate, self.version, self.protect)
        y = torch.from_numpy(out).to(self.tdev)
        if self.tgt_sr != self.SR:
            y = AF.resample(y, self.tgt_sr, self.SR)
        return y.cpu().numpy().astype(np.float32)

    def process(self, inp: np.ndarray) -> np.ndarray:
        block, xfade, sola_search = self.block, self.xfade, self.sola_search
        t0 = time.perf_counter()
        self.buf = np.concatenate([self.buf[block:], inp])
        # gate on the audio that is about to be emitted (with lookahead that is not the newest block)
        gate_src = self.buf[-(block + self.lookahead):-self.lookahead] if self.lookahead else inp
        rms = float(np.sqrt(np.mean(gate_src ** 2)) + 1e-9)
        if rms < self.gate:
            out = np.zeros(block, dtype=np.float32)
            self.sola_buf[:] = 0
        else:
            y = self.convert(self.buf)
            if self.lookahead:
                y = y[: y.shape[0] - self.lookahead]
            need = block + sola_search + xfade
            y = y[-need:] if y.shape[0] >= need else np.pad(y, (need - y.shape[0], 0))
            # SOLA: best alignment of the new head against the previous tail
            head = y[: xfade + sola_search]
            if np.any(self.sola_buf):
                cor = np.correlate(head, self.sola_buf, "valid")
                nrm = np.sqrt(np.convolve(head ** 2, np.ones(xfade), "valid") + 1e-8)
                off = int(np.argmax(cor / nrm))
            else:
                off = 0
            seg = y[off: off + block + xfade].copy()
            seg[:xfade] = seg[:xfade] * self.fade_in + self.sola_buf * self.fade_out
            out = seg[:block]
            self.sola_buf = seg[block: block + xfade].copy()
            if self.sola_buf.shape[0] < xfade:
                self.sola_buf = np.pad(self.sola_buf, (0, xfade - self.sola_buf.shape[0]))
        ms = (time.perf_counter() - t0) * 1000
        self.stats["n"] += 1
        if ms > self.block_s * 1000:
            self.stats["over"] += 1
        if self.log and self.stats["n"] % 10 == 0:
            print(f"[{self.stats['n']}] {ms:.1f}ms / {self.block_s * 1000:.0f}ms "
                  f"{'OK' if ms < self.block_s * 1000 else 'SLOW'}  over={self.stats['over']} gate={'open' if rms >= self.gate else 'shut'}",
                  flush=True)
        return np.clip(out, -1, 1)

    def warmup(self) -> None:
        # warm-up (cudnn/rmvpe/hubert)
        for _ in range(2):
            self.process((np.random.randn(self.block) * 0.05).astype(np.float32))
        self.stats.update(n=0, over=0)
        self.reset()


def _import_sounddevice():
    """sounddevice, finding libportaudio on LD_LIBRARY_PATH when ctypes cannot (NixOS). Mirrors
    aivoice/audio/portaudio.py: this file must not import aivoice."""
    import ctypes.util

    try:
        import sounddevice

        return sounddevice
    except OSError:
        lib = os.environ.get("AIVOICE_PORTAUDIO") or next(
            (os.path.join(d, n) for d in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if d
             for n in ("libportaudio.so.2", "libportaudio.so") if os.path.isfile(os.path.join(d, n))), None)
        if lib is None:
            raise
        orig = ctypes.util.find_library
        ctypes.util.find_library = lambda name: lib if name == "portaudio" else orig(name)
        try:
            import sounddevice

            return sounddevice
        finally:
            ctypes.util.find_library = orig


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
    ap.add_argument("--lookahead", type=float, default=0.0, help="right context seconds (adds latency, improves edges)")
    ap.add_argument("--threshold", type=float, default=-50.0, help="noise gate dBFS")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--sr", type=int, default=48000)
    ap.add_argument("--input-wav", default=None, help="offline test: stream this wav through the RT loop")
    ap.add_argument("--output-wav", default=None)
    a = ap.parse_args()

    rt = RvcRealtime(model=a.model, index=a.index, pitch=a.pitch, index_rate=a.index_rate, protect=a.protect,
                     block=a.block, extra=a.extra, crossfade=a.crossfade, threshold=a.threshold, device=a.device, sr=a.sr,
                     lookahead=a.lookahead)
    torch = rt.torch
    print("RVC v2 Real-Time Voice Conversion", flush=True)
    print(f"  Device:     {'cuda' if rt.dev.startswith('cuda') else 'cpu'}"
          f"{' (' + torch.cuda.get_device_name(0) + ')' if rt.dev.startswith('cuda') else ''}", flush=True)
    print(f"  Model:      {a.model}", flush=True)
    print(f"  Index:      {a.index or '(none)'}  rate={a.index_rate}", flush=True)
    print(f"  Pitch:      {a.pitch:+g} st   block={a.block}s extra={a.extra}s xfade={a.crossfade}s gate={a.threshold}dB",
          flush=True)
    print(f"  fp16:       {rt.is_half}   model sr={rt.tgt_sr}  f0={bool(rt.if_f0)}  version={rt.version}", flush=True)
    rt.warmup()
    block, SR = rt.block, rt.SR

    if a.input_wav:
        import soundfile as sf

        x, sr = sf.read(a.input_wav, dtype="float32", always_2d=True)
        x = x.mean(1)
        if sr != SR:
            x = rt.AF.resample(torch.from_numpy(x), sr, SR).numpy()
        x = np.pad(x, (0, (-len(x)) % block + block))
        outs = [rt.process(x[i: i + block]) for i in range(0, len(x), block)]
        sf.write(a.output_wav or "rt_out.wav", np.concatenate(outs), SR)
        print(f"wrote {a.output_wav}  chunks={rt.stats['n']} over={rt.stats['over']}", flush=True)
        return 0

    sd = _import_sounddevice()

    print("[Stream] Running (sounddevice, 48 kHz mono). Ctrl+C to stop.", flush=True)
    xr = 0

    def cb(indata, outdata, frames, t, status):
        nonlocal xr
        if status:
            xr += 1
            if xr % 20 == 1:
                print(f"[Stream] status: {status} (x{xr})", flush=True)
        outdata[:, 0] = rt.process(indata[:, 0].copy())

    with sd.Stream(samplerate=SR, blocksize=block, channels=1, dtype="float32", latency="low", callback=cb):
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
