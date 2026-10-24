"""RT-VC DDSP vocoder with exact streaming state.

control encoder (causal conv on [f0/500, loudness, periodicity, EMA12]) -> FiLM heads (speaker embedding)
-> harmonic oscillator (sin + cos banks, softmax harmonic distribution, anti-alias mask) + LTV-FIR filtered
noise (overlap-add) -> speaker-dependent reverb (causal FIR from the embedding).

Streaming state: conv buffers, Hann-upsampling tails (frame-rate controls -> audio rate), oscillator
phase (float64), noise generator and OLA tail, reverb tail. Upsampling is the causal overlap-add form
used by the official ``realtime_forward`` for every control *including f0* (the official offline path
interpolates f0 linearly with half-frame lookahead; aivoice trains and runs with the causal form, R-F2).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ...common.streaming_nn import State
from ..config import FRAME, SAMPLE_RATE, VocoderConfig
from .convnet import CausalConvNet, mlp


class LinearFiLM(nn.Module):
    def __init__(self, cin: int, cout: int, cond: int) -> None:
        super().__init__()
        self.lin = nn.Linear(cin, cout)
        self.film = nn.Linear(cond, 2 * cout)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        a, b = self.film(g)[:, None].chunk(2, -1)
        return a * self.lin(x) + b


class MLPFiLM(nn.Module):
    def __init__(self, cin: int, hidden: int, cout: int, cond: int) -> None:
        super().__init__()
        self.f = LinearFiLM(cin, hidden, cond)
        self.rest = nn.Sequential(nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, cout))

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return self.rest(self.f(x, g))


def scaled_sigmoid(x: torch.Tensor) -> torch.Tensor:  # DDSP
    return 2.0 * torch.sigmoid(x) ** math.log(10) + 1e-7


class DDSPVocoder(nn.Module):
    def __init__(self, cfg: VocoderConfig, spkr_emb_dim: int, ema_channels: int = 12, sr: int = SAMPLE_RATE,
                 frame: int = FRAME) -> None:
        super().__init__()
        self.cfg, self.sr, self.F = cfg, sr, frame
        self.enc = CausalConvNet(3 + ema_channels, cfg.hidden_dim, cfg.hidden_dim, cfg.kernel_size, cfg.n_blocks)
        h = cfg.hidden_dim * cfg.forward_expansion
        self.head_amp = MLPFiLM(cfg.hidden_dim, h, 2 * (cfg.nharmonics + 1), spkr_emb_dim)
        self.head_H = MLPFiLM(cfg.hidden_dim, h, cfg.nbands, spkr_emb_dim)
        self.reverb = mlp(spkr_emb_dim, cfg.reverb_len, cfg.reverb_len, 1, 1)
        self.register_buffer("hann_up", torch.hann_window(2 * frame + 1, periodic=False), persistent=False)
        self.register_buffer("hann_fir", torch.hann_window(2 * cfg.nbands - 1, periodic=False), persistent=False)
        self.register_buffer("k", torch.arange(1, cfg.nharmonics + 1, dtype=torch.float32)[:, None], persistent=False)

    # -- frame-rate controls ----------------------------------------------------------------------
    def _controls(self, h: torch.Tensor, g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = h.transpose(1, 2)                                   # [B, t, C]
        amp = self.head_amp(h, g)
        n = self.cfg.nharmonics + 1
        amps = amp.view(*amp.shape[:2], 2, n)                    # sin, cos banks
        a = scaled_sigmoid(amps[..., 0])                         # [B, t, 2] overall amplitude
        c = amps[..., 1:]                                        # [B, t, 2, nh] harmonic logits
        H = scaled_sigmoid(self.head_H(h, g)) * self.cfg.attenuate  # [B, t, nbands]
        return a, c, H

    def _inputs(self, f0, loudness, periodicity, ema) -> torch.Tensor:
        return torch.cat([f0 / self.cfg.f0_norm, loudness, periodicity, ema], 1)

    # -- audio-rate synthesis (shared by forward and step) ---------------------------------------
    def _upsample(self, s: torch.Tensor, tail: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        """Causal Hann overlap-add: s [B, C, t] -> [B, C, t*F], tail [B, C, F+1]."""
        B, C, t = s.shape
        k = self.hann_up.to(s.dtype).expand(C, 1, -1)
        y = F.conv_transpose1d(s, k, stride=self.F, groups=C)    # [B, C, (t-1)F + 2F + 1]
        if tail is not None:
            y[..., : tail.shape[-1]] += tail
        return y[..., : t * self.F], y[..., t * self.F:]

    def _synth(self, f0, a, c, H, g, st: State, gen: torch.Generator) -> tuple[torch.Tensor, State]:
        B, t = f0.shape[0], f0.shape[-1]
        f0u, st_f0 = self._upsample(f0, st.get("f0"))                          # [B, 1, N]
        au, st_a = self._upsample(a.transpose(1, 2), st.get("a"))              # [B, 2, N]
        cs, st_cs = self._upsample(c[:, :, 0].transpose(1, 2), st.get("cs"))   # [B, nh, N]
        cc, st_cc = self._upsample(c[:, :, 1].transpose(1, 2), st.get("cc"))
        harm = self.k * f0u                                                    # [B, nh, N]
        alias = harm >= self.sr / 2
        cs = torch.softmax(cs.masked_fill(alias, -1e20), 1)
        cc = torch.softmax(cc.masked_fill(alias, -1e20), 1)
        ph0 = st.get("phase", torch.zeros(B, 1, 1, dtype=torch.float64, device=f0.device))
        base = ph0 + 2 * math.pi * torch.cumsum(f0u.double() / self.sr, -1)    # [B, 1, N] float64
        phase = (self.k.double() * base).remainder(2 * math.pi).to(f0.dtype)
        harmonic = (au[:, :1] * cs * torch.sin(phase)).sum(1) + (au[:, 1:] * cc * torch.cos(phase)).sum(1)
        # filtered noise: zero-phase FIR from H, shifted to linear phase, Hann-windowed, per-frame OLA
        nb = self.cfg.nbands
        fir = torch.fft.irfft(H, n=2 * nb - 1, dim=-1).roll(nb - 1, -1) * self.hann_fir  # [B, t, L]
        L = fir.shape[-1]
        noise = (torch.rand(B, t, self.F, generator=gen, dtype=torch.float32) * 2 - 1).to(f0.device, f0.dtype)
        n_fft = 1 << (self.F + L - 2).bit_length()
        filt = torch.fft.irfft(torch.fft.rfft(noise, n_fft) * torch.fft.rfft(fir, n_fft), n_fft)[..., : self.F + L - 1]
        ola = F.fold(filt.transpose(1, 2), (1, (t - 1) * self.F + self.F + L - 1), (1, self.F + L - 1),
                     stride=(1, self.F)).reshape(B, -1)
        if "noise" in st:
            ola[:, : L - 1] += st["noise"]
        noise_out, st_noise = ola[:, : t * self.F], ola[:, t * self.F:]
        y = harmonic + noise_out
        # reverb: causal FIR from the speaker embedding, overlap-added
        kern = self.reverb(g)                                                  # [B, R]
        R = kern.shape[-1]
        n_fft = 1 << (y.shape[-1] + R - 2).bit_length()
        full = torch.fft.irfft(torch.fft.rfft(y, n_fft) * torch.fft.rfft(kern, n_fft), n_fft)[:, : y.shape[-1] + R - 1]
        if "rev" in st:
            full[:, : R - 1] += st["rev"]
        out = full[:, : y.shape[-1]]
        new = {"f0": st_f0, "a": st_a, "cs": st_cs, "cc": st_cc, "phase": base[..., -1:].remainder(2 * math.pi),
               "noise": st_noise, "rev": full[:, y.shape[-1]:]}
        return out, new

    # -- public ----------------------------------------------------------------------------------
    def forward(self, f0, loudness, periodicity, ema, g, seed: int = 0) -> torch.Tensor:
        """Frame-rate inputs [B, *, t] (f0 Hz) + speaker embedding [B, E] -> waveform [B, t*80]."""
        a, c, H = self._controls(self.enc(self._inputs(f0, loudness, periodicity, ema)), g)
        gen = torch.Generator().manual_seed(seed)
        # DSP synthesis in fp32 with autocast off: cuFFT has no bf16 kernels, so bf16 training of the
        # vocoder stage raised in torch.fft; the networks above still run under autocast.
        with torch.autocast(device_type=f0.device.type, enabled=False):
            y, _ = self._synth(f0.float(), a.float(), c.float(), H.float(), g.float(), {}, gen)
        return y

    def step(self, f0, loudness, periodicity, ema, g, state: State | None, seed: int = 0) -> tuple[torch.Tensor, State]:
        state = state or {"enc": None, "synth": {}, "gen": torch.Generator().manual_seed(seed)}
        if f0.shape[-1] == 0:
            return f0.new_zeros(f0.shape[0], 0), state
        h, s_enc = self.enc.step(self._inputs(f0, loudness, periodicity, ema), state["enc"])
        a, c, H = self._controls(h, g)
        y, s_syn = self._synth(f0, a, c, H, g, state["synth"], state["gen"])
        return y, {"enc": s_enc, "synth": s_syn, "gen": state["gen"]}
