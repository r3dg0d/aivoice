"""Voice modulation: change attributes of the speaker's own voice (pitch, formants, resonance, quality)."""

from .dsp import ModulationParams, SpectralModulator, measure_delay
from .presets import all_presets, get_preset

__all__ = ["ModulationParams", "SpectralModulator", "all_presets", "get_preset", "measure_delay"]
