# voicebiomarkers/core/mfcc.py
"""
MFCC Feature Extraction

Extracts Mel-frequency cepstral coefficients and their derivatives
(delta, delta-delta) with statistical summaries.
"""

from typing import Any

import librosa
import numpy as np
from scipy.stats import kurtosis, skew

from voicebiomarkers.config.constants import (
    FFT_SIZE,
    HOP_LENGTH,
    MFCC_FMAX,
    MFCC_FMIN,
    N_MELS,
    N_MFCC,
    PRE_EMPHASIS_COEF,
    SAMPLE_RATE,
)


def extract_mfcc(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    n_mfcc: int = N_MFCC,
    n_mels: int = N_MELS,
    fmin: float = MFCC_FMIN,
    fmax: float = MFCC_FMAX,
    n_fft: int = FFT_SIZE,
    hop_length: int = HOP_LENGTH,
) -> np.ndarray:
    """
    Extract MFCC coefficients from audio.

    Args:
        audio: Audio signal (should be preprocessed)
        sample_rate: Sample rate in Hz
        n_mfcc: Number of MFCC coefficients
        n_mels: Number of Mel filter banks
        fmin: Minimum frequency
        fmax: Maximum frequency
        n_fft: FFT window size
        hop_length: Hop length

    Returns:
        MFCC matrix of shape (n_mfcc, time_frames)
    """
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=n_mfcc,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    return mfcc


def extract_mfcc_deltas(mfcc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract delta and delta-delta (acceleration) coefficients.

    Args:
        mfcc: MFCC matrix

    Returns:
        Tuple of (delta, delta2) matrices
    """
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    return delta, delta2


def calculate_coefficient_statistics(coeff: np.ndarray, prefix: str) -> dict[str, float]:
    """
    Calculate statistical summaries for a single coefficient.

    Args:
        coeff: 1D coefficient array (time series)
        prefix: Prefix for feature names (e.g., "mfcc_01")

    Returns:
        Dictionary with statistical features
    """
    return {
        f"{prefix}_mean": float(np.mean(coeff)),
        f"{prefix}_std": float(np.std(coeff)),
        f"{prefix}_kurtosis": float(kurtosis(coeff)),
        f"{prefix}_skewness": float(skew(coeff)),
        f"{prefix}_max": float(np.max(coeff)),
        f"{prefix}_min": float(np.min(coeff)),
        f"{prefix}_range": float(np.ptp(coeff)),
    }


def calculate_delta_statistics(coeff: np.ndarray, prefix: str) -> dict[str, float]:
    """
    Calculate statistics for delta coefficients (fewer stats).

    Args:
        coeff: 1D coefficient array
        prefix: Prefix for feature names

    Returns:
        Dictionary with statistical features
    """
    return {
        f"{prefix}_mean": float(np.mean(coeff)),
        f"{prefix}_std": float(np.std(coeff)),
        f"{prefix}_max": float(np.max(coeff)),
        f"{prefix}_min": float(np.min(coeff)),
        f"{prefix}_range": float(np.ptp(coeff)),
    }


def extract_mfcc_features(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    n_mfcc: int = N_MFCC,
    apply_pre_emphasis: bool = True,
    apply_normalization: bool = True,
) -> dict[str, Any]:
    """
    Extract complete MFCC feature set with statistics.

    This extracts:
    - N_MFCC coefficients × 7 statistics = 504 features (for n_mfcc=72)
    - N_MFCC deltas × 5 statistics = 360 features
    - N_MFCC delta2 × 5 statistics = 360 features
    - Total: 1224 features (for n_mfcc=72)

    Args:
        audio: Audio signal (already preprocessed: noise reduced, trimmed)
        sample_rate: Sample rate in Hz
        n_mfcc: Number of MFCC coefficients
        apply_pre_emphasis: Apply pre-emphasis filter
        apply_normalization: Normalize audio before extraction

    Returns:
        Dictionary with all MFCC features

    Example:
        >>> features = extract_mfcc_features(audio_array)
        >>> print(features["mfcc_01_mean"])
    """
    features: dict[str, Any] = {}

    # Apply pre-emphasis if requested
    y = audio
    if apply_pre_emphasis:
        y = librosa.effects.preemphasis(y, coef=PRE_EMPHASIS_COEF)

    # Normalize if requested
    if apply_normalization:
        y = librosa.util.normalize(y)

    # Extract MFCC
    mfcc = extract_mfcc(y, sample_rate, n_mfcc)

    # Extract deltas
    delta, delta2 = extract_mfcc_deltas(mfcc)

    # Calculate statistics for each coefficient
    for i in range(n_mfcc):
        # MFCC statistics (7 features per coefficient)
        features.update(
            calculate_coefficient_statistics(mfcc[i], f"mfcc_{i+1:02}")
        )

        # Delta statistics (5 features per coefficient)
        features.update(
            calculate_delta_statistics(delta[i], f"delta_{i+1:02}")
        )

        # Delta-delta statistics (5 features per coefficient)
        features.update(
            calculate_delta_statistics(delta2[i], f"delta2_{i+1:02}")
        )

    # Add extraction metadata
    features["mfcc_extraction_status"] = "success"
    features["n_mfcc_extracted"] = n_mfcc
    features["mfcc_time_frames"] = mfcc.shape[1]

    return features
