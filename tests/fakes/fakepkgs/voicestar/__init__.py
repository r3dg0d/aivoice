import numpy as np

from _log import log


class _Wave:
    def __init__(self, a):
        self.a = a

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class _Seg:
    def __init__(self, a):
        self.waveform = _Wave(a)
        self.sample_rate = 16000


class VoiceStar:
    def __init__(self, model_name="VoiceStar_840M_30s", device=None, top_k=10, top_p=1.0, temperature=1.0, repeat_prompt=1):
        log("init", model=model_name, device=device)

    def generate(self, reference_speech, text, reference_text=None, target_duration=None, output_path=None):
        log("generate", reference=reference_speech, text=text, reference_text=reference_text, target_duration=target_duration)
        n = int(16000 * (target_duration or 1.0))
        return _Seg((0.1 * np.sin(2 * np.pi * 150 * np.arange(n) / 16000)).astype(np.float32)[None, :])
