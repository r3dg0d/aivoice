import numpy as np

from _log import log


class _TTS:
    sample_rate = 48000


class VoxCPM:
    def __init__(self, voxcpm_model_path, zipenhancer_model_path="iic/speech_zipenhancer_ans_multiloss_16k_base",
                 enable_denoiser=True, optimize=True, device=None, lora_config=None, lora_weights_path=None):
        log("init", path=voxcpm_model_path, zip=zipenhancer_model_path, denoiser=enable_denoiser, optimize=optimize,
            device=device, lora=lora_weights_path)
        self.tts_model = _TTS()

    def _audio(self, text):
        n = int(48000 * max(0.3, 0.25 * len(text.split())))
        return (0.1 * np.sin(2 * np.pi * 180 * np.arange(n) / 48000)).astype(np.float32)

    def generate(self, text, prompt_wav_path=None, prompt_text=None, reference_wav_path=None, cfg_value=2.0,
                 inference_timesteps=10, min_len=2, max_len=4096, normalize=False, denoise=False, **kw):
        log("generate", text=text, prompt_wav=prompt_wav_path, prompt_text=prompt_text, reference=reference_wav_path,
            cfg=cfg_value, steps=inference_timesteps)
        if (prompt_wav_path is None) != (prompt_text is None):
            raise ValueError("prompt_wav_path and prompt_text must both be provided or both be None")
        return self._audio(text)

    def generate_streaming(self, text, **kw):
        log("generate_streaming", text=text, reference=kw.get("reference_wav_path"))
        a = self._audio(text)
        for i in range(0, a.size, 4800):
            yield a[i:i + 4800]
