# voicebiomarkers/core/praat.py
"""
Praat/Parselmouth Feature Extraction

Extracts acoustic features using Praat via Parselmouth:
- Fundamental frequency (F0) and derivatives
- Jitter and shimmer (perturbation measures)
- Harmonics-to-noise ratio (HNR)
- Formant frequencies
- Cardiovascular-specific features

Note: This module requires praat-parselmouth to be installed.
"""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Check for Parselmouth availability
try:
    import parselmouth
    from parselmouth.praat import call

    PARSELMOUTH_AVAILABLE = True
except ImportError:
    PARSELMOUTH_AVAILABLE = False
    logger.warning("Parselmouth not available. Praat features will be disabled.")


def extract_praat_features(
    audio: np.ndarray,
    sample_rate: int,
) -> dict[str, Any]:
    """
    Extract Praat-based acoustic features.

    Features extracted:
    - F0 statistics (mean, std, min, max, range, IQR, entropy)
    - Jitter (local, ppq5)
    - Shimmer (local, apq11)
    - HNR (mean, std, min, max)
    - Formants (F1, F2, F3 with means and stds)
    - Cardiovascular features (tremor, stability, breathiness)

    Args:
        audio: Audio signal (preprocessed)
        sample_rate: Sample rate in Hz

    Returns:
        Dictionary with all Praat features
    """
    if not PARSELMOUTH_AVAILABLE:
        return {
            "parselmouth_status": "unavailable",
            "praat_errors": "Parselmouth not installed",
        }

    features: dict[str, Any] = {
        "parselmouth_status": "success",
        "praat_errors": None,
    }

    try:
        # Create Parselmouth Sound object
        sound = parselmouth.Sound(audio, sampling_frequency=sample_rate)
        duration = sound.get_total_duration()

        # Extract pitch
        pitch = call(
            sound,
            "To Pitch",
            0.0,  # time step (0 = auto)
            50.0,  # min pitch (Hz)
            500.0,  # max pitch (Hz)
        )

        # F0 statistics
        f0_features = _extract_f0_features(pitch, duration)
        features.update(f0_features)

        # Create PointProcess for jitter/shimmer
        point_process = call(
            sound,
            "To PointProcess (periodic, cc)",
            50.0,  # min pitch
            500.0,  # max pitch
        )

        # Jitter
        jitter_features = _extract_jitter_features(sound, point_process)
        features.update(jitter_features)

        # Shimmer
        shimmer_features = _extract_shimmer_features(sound, point_process)
        features.update(shimmer_features)

        # HNR
        hnr_features = _extract_hnr_features(sound)
        features.update(hnr_features)

        # Formants
        formant_features = _extract_formant_features(sound)
        features.update(formant_features)

        # Cardiovascular-specific features
        cardio_features = _extract_cardiovascular_features(sound, pitch)
        features.update(cardio_features)

    except Exception as e:
        logger.error(f"Praat feature extraction error: {e}")
        features["parselmouth_status"] = "partial"
        features["praat_errors"] = str(e)[:200]

    return features


def _extract_f0_features(pitch, duration: float) -> dict[str, Any]:
    """Extract fundamental frequency features."""
    features = {}

    try:
        f0_values = pitch.selected_array["frequency"]
        f0_voiced = f0_values[f0_values > 0]

        if len(f0_voiced) > 0:
            features["f0_mean"] = float(np.mean(f0_voiced))
            features["f0_std"] = float(np.std(f0_voiced))
            features["f0_min"] = float(np.min(f0_voiced))
            features["f0_max"] = float(np.max(f0_voiced))
            features["f0_range"] = float(np.ptp(f0_voiced))
            features["f0_iqr"] = float(np.percentile(f0_voiced, 75) - np.percentile(f0_voiced, 25))

            # F0 entropy (normalized)
            hist, _ = np.histogram(f0_voiced, bins=20, density=True)
            hist = hist[hist > 0]
            features["f0_entropy"] = float(-np.sum(hist * np.log2(hist + 1e-10)))
        else:
            features.update(
                {
                    "f0_mean": None,
                    "f0_std": None,
                    "f0_min": None,
                    "f0_max": None,
                    "f0_range": None,
                    "f0_iqr": None,
                    "f0_entropy": None,
                }
            )

    except Exception as e:
        logger.debug(f"F0 extraction error: {e}")
        features.update(
            {
                "f0_mean": None,
                "f0_std": None,
                "f0_min": None,
                "f0_max": None,
                "f0_range": None,
                "f0_iqr": None,
                "f0_entropy": None,
            }
        )

    return features


def _extract_jitter_features(sound, point_process) -> dict[str, Any]:
    """Extract jitter (F0 perturbation) features."""
    features = {}

    try:
        features["jitter_local"] = call(
            point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3
        )
        features["jitter_ppq5"] = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    except Exception as e:
        logger.debug(f"Jitter extraction error: {e}")
        features["jitter_local"] = None
        features["jitter_ppq5"] = None

    return features


def _extract_shimmer_features(sound, point_process) -> dict[str, Any]:
    """Extract shimmer (amplitude perturbation) features."""
    features = {}

    try:
        features["shimmer_local"] = call(
            [sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6
        )
        features["shimmer_apq11"] = call(
            [sound, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6
        )
    except Exception as e:
        logger.debug(f"Shimmer extraction error: {e}")
        features["shimmer_local"] = None
        features["shimmer_apq11"] = None

    return features


def _extract_hnr_features(sound) -> dict[str, Any]:
    """Extract harmonics-to-noise ratio features."""
    features = {}

    try:
        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 50.0, 0.1, 1.0)
        hnr_values = harmonicity.values[0]
        hnr_valid = hnr_values[hnr_values != -200]

        if len(hnr_valid) > 0:
            features["hnr_praat"] = float(np.mean(hnr_valid))
            features["hnr_std"] = float(np.std(hnr_valid))
            features["hnr_min"] = float(np.min(hnr_valid))
            features["hnr_max"] = float(np.max(hnr_valid))
        else:
            features.update({"hnr_praat": None, "hnr_std": None, "hnr_min": None, "hnr_max": None})

    except Exception as e:
        logger.debug(f"HNR extraction error: {e}")
        features.update({"hnr_praat": None, "hnr_std": None, "hnr_min": None, "hnr_max": None})

    return features


def _extract_formant_features(sound) -> dict[str, Any]:
    """Extract formant frequency features."""
    features = {}

    try:
        formant = call(sound, "To Formant (burg)", 0.0, 5, 5500.0, 0.025, 50.0)
        duration = sound.get_total_duration()
        n_frames = int(duration / 0.01)

        f1_values, f2_values, f3_values = [], [], []

        for i in range(n_frames):
            time = i * 0.01
            try:
                f1 = call(formant, "Get value at time", 1, time, "Hertz", "Linear")
                f2 = call(formant, "Get value at time", 2, time, "Hertz", "Linear")
                f3 = call(formant, "Get value at time", 3, time, "Hertz", "Linear")

                if f1 and f1 > 0:
                    f1_values.append(f1)
                if f2 and f2 > 0:
                    f2_values.append(f2)
                if f3 and f3 > 0:
                    f3_values.append(f3)
            except Exception:
                continue

        if f1_values:
            features["f1_mean"] = float(np.mean(f1_values))
            features["f1_std"] = float(np.std(f1_values))
        else:
            features["f1_mean"] = None
            features["f1_std"] = None

        if f2_values:
            features["f2_mean"] = float(np.mean(f2_values))
            features["f2_std"] = float(np.std(f2_values))
        else:
            features["f2_mean"] = None
            features["f2_std"] = None

        if f3_values:
            features["f3_mean"] = float(np.mean(f3_values))
            features["f3_std"] = float(np.std(f3_values))
        else:
            features["f3_mean"] = None
            features["f3_std"] = None

        # Formant dispersion (F2/F1 ratio correlation with vocal tract length)
        if f1_values and f2_values:
            features["formant_dispersion"] = float(np.mean(f2_values) - np.mean(f1_values))
        else:
            features["formant_dispersion"] = None

    except Exception as e:
        logger.debug(f"Formant extraction error: {e}")
        features.update(
            {
                "f1_mean": None,
                "f1_std": None,
                "f2_mean": None,
                "f2_std": None,
                "f3_mean": None,
                "f3_std": None,
                "formant_dispersion": None,
            }
        )

    return features


def _extract_cardiovascular_features(sound, pitch) -> dict[str, Any]:
    """
    Extract cardiovascular-relevant voice features.

    These features are specifically designed for cardiovascular
    risk assessment based on literature linking voice to cardiac health.
    """
    features = {}

    try:
        # Voice tremor (F0 modulation rate ~3-7 Hz)
        f0_values = pitch.selected_array["frequency"]
        f0_voiced = f0_values[f0_values > 0]

        if len(f0_voiced) > 20:
            # Calculate tremor as coefficient of variation of F0 changes
            f0_diff = np.diff(f0_voiced)
            features["voice_tremor"] = float(np.std(f0_diff) / (np.mean(f0_voiced) + 1e-10))

            # Vocal stability index
            features["vocal_stability_index"] = float(
                1 / (1 + np.std(f0_voiced) / np.mean(f0_voiced))
            )
        else:
            features["voice_tremor"] = None
            features["vocal_stability_index"] = None

        # Intensity features
        try:
            intensity = call(sound, "To Intensity", 100, 0.0, "yes")
            intensity_values = intensity.values[0]
            features["intensity_mean"] = float(np.mean(intensity_values))
            features["intensity_std"] = float(np.std(intensity_values))
            features["intensity_range"] = float(np.ptp(intensity_values))
        except Exception:
            features["intensity_mean"] = None
            features["intensity_std"] = None
            features["intensity_range"] = None

        # Spectral tilt (breathiness indicator)
        try:
            spectrum = call(sound, "To Spectrum", "yes")
            features["spectral_tilt"] = call(spectrum, "Get slope", 0, 1000, 1000, 4000, "energy")
        except Exception:
            features["spectral_tilt"] = None

    except Exception as e:
        logger.debug(f"Cardiovascular feature extraction error: {e}")
        features.update(
            {
                "voice_tremor": None,
                "vocal_stability_index": None,
                "intensity_mean": None,
                "intensity_std": None,
                "intensity_range": None,
                "spectral_tilt": None,
            }
        )

    return features
