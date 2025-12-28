# voicebiomarkers
"""
Voice Biomarkers Extraction Library

A scientifically rigorous pipeline for extracting acoustic features
from voice recordings for cardiovascular risk assessment.

Modules:
    core: Feature extraction (MFCC, Praat, preprocessing, quality)
    config: Settings and constants
    io: Audio loading, metadata parsing, export
    utils: Logging, reproducibility
"""

__version__ = "1.0.0"

from voicebiomarkers.core.mfcc import extract_mfcc_features
from voicebiomarkers.core.praat import extract_praat_features
from voicebiomarkers.core.preprocessing import preprocess_audio
from voicebiomarkers.core.quality import calculate_quality_metrics

__all__ = [
    "preprocess_audio",
    "extract_mfcc_features",
    "extract_praat_features",
    "calculate_quality_metrics",
]
