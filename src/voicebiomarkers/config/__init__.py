# voicebiomarkers.config
"""Configuration and constants modules."""

from voicebiomarkers.config.constants import (
    MIN_VOICE_DURATION,
    N_MFCC,
    NOISE_REDUCTION_AGGRESSION,
    SAMPLE_RATE,
)
from voicebiomarkers.config.settings import Settings, get_settings

__all__ = [
    "Settings",
    "get_settings",
    "SAMPLE_RATE",
    "N_MFCC",
    "MIN_VOICE_DURATION",
    "NOISE_REDUCTION_AGGRESSION",
]
