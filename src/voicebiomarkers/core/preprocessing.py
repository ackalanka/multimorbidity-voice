# voicebiomarkers/core/preprocessing.py
"""
Unified Audio Preprocessing Pipeline

Provides consistent preprocessing for all feature extraction,
ensuring reproducibility across MFCC and Praat pipelines.
"""


import librosa
import noisereduce as nr
import numpy as np

from voicebiomarkers.config.constants import (
    FFT_SIZE,
    HOP_LENGTH,
    NOISE_PROFILE_DURATION,
    NOISE_REDUCTION_AGGRESSION,
    SAMPLE_RATE,
    VAD_FRAME_LENGTH,
    VAD_HOP_LENGTH,
    VAD_TOP_DB,
)


def load_audio(file_path: str, target_sr: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """
    Load audio file and resample to target sample rate.

    Args:
        file_path: Path to audio file (WAV, MP3, etc.)
        target_sr: Target sample rate in Hz

    Returns:
        Tuple of (audio signal, sample rate)
    """
    y, sr = librosa.load(file_path, sr=None, mono=True)

    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return y, sr


def reduce_noise(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    noise_profile_duration: float = NOISE_PROFILE_DURATION,
    aggression: float = NOISE_REDUCTION_AGGRESSION,
) -> np.ndarray:
    """
    Apply spectral noise reduction using noisereduce library.

    Uses the first `noise_profile_duration` seconds as noise profile.

    Args:
        audio: Audio signal
        sample_rate: Sample rate in Hz
        noise_profile_duration: Duration of noise profile in seconds
        aggression: Noise reduction strength (0.0 to 2.0)

    Returns:
        Noise-reduced audio signal

    Reference:
        Sainburg et al. (2020) "A spectral subtraction method for
        reducing noise in speech signals"
    """
    noise_samples = int(sample_rate * noise_profile_duration)

    # Use first part of audio as noise profile
    if len(audio) > noise_samples:
        noise_profile = audio[:noise_samples]
    else:
        noise_profile = audio

    y_clean = nr.reduce_noise(
        y=audio,
        y_noise=noise_profile,
        sr=sample_rate,
        stationary=True,
        prop_decrease=aggression,
        n_fft=FFT_SIZE,
        hop_length=HOP_LENGTH,
    )

    return y_clean


def trim_silence(
    audio: np.ndarray,
    top_db: float = VAD_TOP_DB,
    frame_length: int = VAD_FRAME_LENGTH,
    hop_length: int = VAD_HOP_LENGTH,
) -> tuple[np.ndarray, int, int]:
    """
    Remove leading and trailing silence using Voice Activity Detection.

    Args:
        audio: Audio signal
        top_db: Threshold in dB below peak for silence detection
        frame_length: Frame length for analysis
        hop_length: Hop length for analysis

    Returns:
        Tuple of (trimmed audio, start sample, end sample)
    """
    y_trimmed, (start, end) = librosa.effects.trim(
        audio,
        top_db=top_db,
        frame_length=frame_length,
        hop_length=hop_length,
    )

    return y_trimmed, start, end


def apply_preemphasis(audio: np.ndarray, coef: float = 0.97) -> np.ndarray:
    """
    Apply pre-emphasis filter to boost high frequencies.

    Args:
        audio: Audio signal
        coef: Pre-emphasis coefficient (default 0.97)

    Returns:
        Pre-emphasized audio signal

    Reference:
        Standard speech processing technique to compensate
        for spectral tilt in speech production.
    """
    return librosa.effects.preemphasis(audio, coef=coef)


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """
    Normalize audio to [-1, 1] range.

    Args:
        audio: Audio signal

    Returns:
        Normalized audio signal
    """
    return librosa.util.normalize(audio)


def preprocess_audio(
    file_path: str,
    sample_rate: int = SAMPLE_RATE,
    apply_noise_reduction: bool = True,
    apply_trimming: bool = True,
    apply_pre_emphasis: bool = False,
    apply_normalization: bool = False,
) -> tuple[np.ndarray, dict]:
    """
    Complete audio preprocessing pipeline.

    Args:
        file_path: Path to audio file
        sample_rate: Target sample rate
        apply_noise_reduction: Whether to reduce noise
        apply_trimming: Whether to trim silence
        apply_pre_emphasis: Whether to apply pre-emphasis
        apply_normalization: Whether to normalize

    Returns:
        Tuple of (processed audio, metadata dict)

    Example:
        >>> audio, meta = preprocess_audio("recording.wav")
        >>> print(meta["original_duration"], meta["processed_duration"])
    """
    # Load and resample
    y, sr = load_audio(file_path, sample_rate)
    original_duration = len(y) / sr

    metadata = {
        "original_duration": original_duration,
        "original_samples": len(y),
        "sample_rate": sr,
    }

    # Noise reduction
    if apply_noise_reduction:
        y = reduce_noise(y, sr)

    # Voice activity detection / trimming
    trim_start, trim_end = 0, len(y)
    if apply_trimming:
        y, trim_start, trim_end = trim_silence(y)

    processed_duration = len(y) / sr
    metadata.update(
        {
            "processed_duration": processed_duration,
            "processed_samples": len(y),
            "trim_start": trim_start,
            "trim_end": trim_end,
        }
    )

    # Optional: Pre-emphasis
    if apply_pre_emphasis:
        y = apply_preemphasis(y)

    # Optional: Normalization
    if apply_normalization:
        y = normalize_audio(y)

    return y, metadata
