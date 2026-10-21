"""RT-VC streaming state: front-end sample buffer (944 samples), conv buffers of the pitch tracker, EMA
inverter and vocoder encoder, Hann-upsampling tails, oscillator phase (float64, wrapped), noise
generator + OLA tail, reverb tail. All bounded; nothing grows with stream length."""

from ...common.streaming_nn import state_nbytes

__all__ = ["state_nbytes"]
