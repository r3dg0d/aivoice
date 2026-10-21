"""Zero-VC streaming state: one nested dict of tensors per stream (content CNN conv buffers, transformer
KV caches + positions, decoder conv / transposed-conv overlap buffers). All sizes are bounded by the
receptive fields and the attention window; nothing grows with stream length."""

from ...common.streaming_nn import state_nbytes

__all__ = ["state_nbytes"]
