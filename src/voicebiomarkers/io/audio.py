# voicebiomarkers/io/audio.py
"""
Audio File I/O

Handles loading and basic audio operations.
"""

from pathlib import Path

import librosa
import numpy as np

from voicebiomarkers.config.constants import SAMPLE_RATE


def load_audio(
    file_path: str | Path,
    target_sr: int = SAMPLE_RATE,
    mono: bool = True,
) -> tuple[np.ndarray, int]:
    """
    Load audio file with optional resampling.

    Args:
        file_path: Path to audio file
        target_sr: Target sample rate (None to keep original)
        mono: Convert to mono

    Returns:
        Tuple of (audio signal, sample rate)
    """
    y, sr = librosa.load(file_path, sr=target_sr, mono=mono)
    return y, sr


def resample_audio(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int,
) -> np.ndarray:
    """
    Resample audio to target sample rate.

    Args:
        audio: Audio signal
        orig_sr: Original sample rate
        target_sr: Target sample rate

    Returns:
        Resampled audio signal
    """
    if orig_sr == target_sr:
        return audio

    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)


def get_audio_duration(audio: np.ndarray, sample_rate: int) -> float:
    """
    Get audio duration in seconds.

    Args:
        audio: Audio signal
        sample_rate: Sample rate in Hz

    Returns:
        Duration in seconds
    """
    return len(audio) / sample_rate
