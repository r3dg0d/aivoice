import argparse
import os
import sys
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from _log import log  # noqa: E402

p = argparse.ArgumentParser()
for a in ("-p", "--srp_ckpt_file", "-v", "-r", "-t", "-o", "-w", "--gen_lang", "--speed", "--fix_duration",
          "--nfe_step", "--cfg_strength", "--device"):
    p.add_argument(a)
for f in ("--auto_detect_lang", "--normalize_text", "--layered"):
    p.add_argument(f, action="store_true")
args = p.parse_args()
log("xvoice", **vars(args))
sec = float(args.fix_duration) if args.fix_duration else 1.0
x = (0.1 * np.sin(2 * np.pi * 200 * np.arange(int(24000 * sec)) / 24000) * 32767).astype("<i2")
os.makedirs(args.o, exist_ok=True)
with wave.open(os.path.join(args.o, args.w), "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(24000)
    w.writeframes(x.tobytes())
