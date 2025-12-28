# voicebiomarkers.config
"""Configuration and constants modules."""

from voicebiomarkers.config.settings import Settings, get_settings
from voicebiomarkers.config.constants import (
    SAMPLE_RATE,
    N_MFCC,
    MIN_VOICE_DURATION,
    NOISE_REDUCTION_AGGRESSION,
)

__all__ = [
    "Settings",
    "get_settings",
    "SAMPLE_RATE",
    "N_MFCC",
    "MIN_VOICE_DURATION",
    "NOISE_REDUCTION_AGGRESSION",
]
