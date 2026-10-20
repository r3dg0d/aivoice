"""Standalone engine worker: runs inside an engine's own Python environment.

aivoice starts ``<engine python> -u worker_main.py <handler>`` and exchanges
JSON lines over stdin/stdout. This file must import nothing from aivoice, so an
engine can live in an isolated venv with its own pinned dependencies (X-Voice,
VoxCPM2 and VoiceStar each pin conflicting versions).

Requests:  {"id": n, "op": "load"|"synthesize"|"ping"|"shutdown", ...}
Responses: {"id": n, "ok": true, ...} or {"id": n, "ok": false, "error": "..."}
Streaming: zero or more {"id": n, "chunk": "<base64 float32 LE>", "sample_rate": sr}
           before the final {"id": n, "ok": true, "done": true, ...}.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
import traceback
import wave

import numpy as np

OUT = sys.stdout
sys.stdout = sys.stderr  # libraries that print must not corrupt the protocol stream


def send(obj):
    OUT.write(json.dumps(obj) + "\n")
    OUT.flush()


def write_wav(path, audio, sr):
    x = np.clip(np.asarray(audio, dtype=np.float32).reshape(-1), -1.0, 1.0)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes((x * 32767).astype("<i2").tobytes())


def seed_everything(seed):
    if seed is None:
        return
    import random

    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


class VoxCPMHandler:
    """openbmb/VoxCPM2 via the official `voxcpm` package (API checked against voxcpm 2.0.3)."""

    def __init__(self):
        self.model = None
        self.sample_rate = None

    def load(self, req):
        from voxcpm import VoxCPM

        self.model = VoxCPM(
            voxcpm_model_path=req["model_path"],
            enable_denoiser=False,  # aivoice prepares references itself; ZipEnhancer would download
            zipenhancer_model_path=None,
            optimize=bool(req.get("optimize", True)),
            device=req.get("device"),
            lora_weights_path=req.get("lora_path"),
        )
        self.sample_rate = int(self.model.tts_model.sample_rate)
        return {"sample_rate": self.sample_rate}

    def synthesize(self, req, emit):
        text = req["text"]
        if req.get("style_prompt"):
            # VoxCPM2 documented format: "(description)text"
            text = f"({req['style_prompt']}){text}"
        kw = dict(
            text=text,
            reference_wav_path=req.get("reference_wav"),
            cfg_value=float(req.get("guidance") or 2.0),
            inference_timesteps=int(req.get("steps") or 10),
            normalize=bool(req.get("normalize", False)),
        )
        if req.get("prompt_wav") and req.get("prompt_text"):
            kw["prompt_wav_path"] = req["prompt_wav"]
            kw["prompt_text"] = req["prompt_text"]
        seed_everything(req.get("seed"))
        if req.get("stream"):
            for chunk in self.model.generate_streaming(**kw):
                emit(np.asarray(chunk, dtype=np.float32), self.sample_rate)
            return {"sample_rate": self.sample_rate}
        audio = self.model.generate(**kw)
        return {"audio": np.asarray(audio, dtype=np.float32), "sample_rate": self.sample_rate}


class VoiceStarHandler:
    """VoiceStar via the `voicestar` package (API checked against voicestar 0.1.0)."""

    def __init__(self):
        self.tts = None

    def load(self, req):
        from voicestar import VoiceStar

        self.tts = VoiceStar(
            model_name=req.get("model_name", "VoiceStar_840M_30s"),
            device=req.get("device"),
            top_k=int(req.get("top_k", 10)),
            temperature=float(req.get("temperature", 1.0)),
            repeat_prompt=int(req.get("repeat_prompt", 1)),
        )
        return {"sample_rate": 16000}

    def synthesize(self, req, emit):
        if not req.get("prompt_text"):
            # VoiceStar would otherwise download and run Whisper large-v3-turbo implicitly.
            raise ValueError("VoiceStar needs the reference transcript")
        seed_everything(req.get("seed"))
        seg = self.tts.generate(
            reference_speech=req["reference_wav"],
            text=req["text"],
            reference_text=req["prompt_text"],
            target_duration=req.get("target_duration"),
        )
        audio = seg.waveform.detach().float().cpu().numpy().reshape(-1)
        return {"audio": audio, "sample_rate": int(seg.sample_rate)}


class ToneHandler:
    """Protocol self-test: tones whose length follows the text. No model."""

    def load(self, req):
        return {"sample_rate": 16000}

    def synthesize(self, req, emit):
        sr = 16000
        n = int(sr * max(0.2, len(req["text"].split()) * 0.3))
        t = np.arange(n) / sr
        audio = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        if req.get("stream"):
            for i in range(0, n, sr // 10):
                emit(audio[i : i + sr // 10], sr)
            return {"sample_rate": sr}
        return {"audio": audio, "sample_rate": sr}


HANDLERS = {"voxcpm": VoxCPMHandler, "voicestar": VoiceStarHandler, "tone": ToneHandler}


def main(argv):
    name = argv[1] if len(argv) > 1 else ""
    if name not in HANDLERS:
        send({"id": 0, "ok": False, "error": f"unknown handler {name!r}"})
        return 2
    h = HANDLERS[name]()
    send({"id": 0, "ok": True, "ready": True, "pid": os.getpid()})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            send({"id": -1, "ok": False, "error": f"bad request: {e}"})
            continue
        rid = req.get("id")
        op = req.get("op")
        t0 = time.perf_counter()
        try:
            if op == "ping":
                send({"id": rid, "ok": True})
            elif op == "shutdown":
                send({"id": rid, "ok": True})
                return 0
            elif op == "load":
                out = h.load(req)
                send({"id": rid, "ok": True, "load_ms": (time.perf_counter() - t0) * 1000, **out})
            elif op == "synthesize":
                first = [None]

                def emit(chunk, sr):
                    if first[0] is None:
                        first[0] = (time.perf_counter() - t0) * 1000
                    send({"id": rid, "chunk": base64.b64encode(np.asarray(chunk, "<f4").tobytes()).decode(), "sample_rate": sr})

                out = h.synthesize(req, emit)
                resp = {"id": rid, "ok": True, "done": True, "sample_rate": out["sample_rate"],
                        "total_ms": (time.perf_counter() - t0) * 1000, "first_ms": first[0]}
                if "audio" in out:
                    path = req["out_path"]
                    write_wav(path, out["audio"], out["sample_rate"])
                    resp["path"] = path
                send(resp)
            else:
                send({"id": rid, "ok": False, "error": f"unknown op {op!r}"})
        except Exception as e:  # noqa: BLE001
            send({"id": rid, "ok": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-2000:]})
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
