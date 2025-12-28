# voicebiomarkers/core/quality.py
"""
Quality Control Metrics

Calculates quality indicators for each audio file to support
scientific QC and peer-review requirements.
"""

from typing import Any

import numpy as np

from voicebiomarkers.config.constants import (
    QC_MAX_CLIPPING_PERCENT,
    QC_MAX_SILENCE_PERCENT,
    QC_MIN_F0_DETECTION_RATE,
    QC_MIN_SNR_DB,
    SAMPLE_RATE,
)


def calculate_snr(
    original_audio: np.ndarray,
    clean_audio: np.ndarray,
    reference_samples: int = 1000,
) -> float | None:
    """
    Calculate Signal-to-Noise Ratio in dB.

    Args:
        original_audio: Original audio signal
        clean_audio: Processed/clean audio signal
        reference_samples: Number of samples for noise estimation

    Returns:
        SNR in dB or None if calculation fails
    """
    if len(original_audio) <= reference_samples:
        return None

    noise_var = np.var(original_audio[:reference_samples])
    signal_var = np.var(clean_audio)

    if noise_var <= 0:
        return None

    snr_db = 10 * np.log10(signal_var / noise_var)
    return float(snr_db)


def calculate_clipping_percentage(audio: np.ndarray, threshold: float = 0.99) -> float:
    """
    Calculate percentage of clipped samples.

    Args:
        audio: Audio signal (assumed normalized to [-1, 1])
        threshold: Clipping threshold

    Returns:
        Percentage of clipped samples (0-100)
    """
    clipped = np.sum(np.abs(audio) >= threshold)
    return float(clipped / len(audio) * 100)


def calculate_silence_percentage(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    threshold_db: float = -40,
) -> float:
    """
    Calculate percentage of silent frames.

    Args:
        audio: Audio signal
        sample_rate: Sample rate in Hz
        threshold_db: dB threshold below peak for silence

    Returns:
        Percentage of silent frames (0-100)
    """
    import librosa

    # Calculate RMS energy per frame
    frame_length = int(sample_rate * 0.025)  # 25ms frames
    hop_length = int(sample_rate * 0.010)    # 10ms hop

    rms = librosa.feature.rms(
        y=audio,
        frame_length=frame_length,
        hop_length=hop_length,
    )[0]

    if len(rms) == 0:
        return 100.0

    # Convert to dB
    rms_db = librosa.power_to_db(rms**2, ref=np.max)

    # Count silent frames
    silent_frames = np.sum(rms_db < threshold_db)
    return float(silent_frames / len(rms_db) * 100)


def calculate_f0_detection_rate(
    f0_values: np.ndarray,
) -> float:
    """
    Calculate F0 (pitch) detection rate.

    Args:
        f0_values: Array of F0 values (NaN for unvoiced)

    Returns:
        Detection rate (0-1)
    """
    if len(f0_values) == 0:
        return 0.0

    valid_f0 = np.sum(~np.isnan(f0_values))
    return float(valid_f0 / len(f0_values))


def calculate_quality_metrics(
    original_audio: np.ndarray,
    processed_audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    f0_values: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Calculate all quality control metrics for an audio file.

    Args:
        original_audio: Original audio before preprocessing
        processed_audio: Audio after preprocessing
        sample_rate: Sample rate in Hz
        f0_values: Optional F0 values for detection rate

    Returns:
        Dictionary with QC metrics and flags
    """
    metrics: dict[str, Any] = {}

    # SNR
    snr = calculate_snr(original_audio, processed_audio)
    metrics["snr_db"] = snr
    metrics["qc_low_snr"] = snr is not None and snr < QC_MIN_SNR_DB

    # Clipping
    clipping = calculate_clipping_percentage(original_audio)
    metrics["clipping_percent"] = clipping
    metrics["qc_clipping"] = clipping > QC_MAX_CLIPPING_PERCENT

    # Silence
    silence = calculate_silence_percentage(processed_audio, sample_rate)
    metrics["silence_percent"] = silence
    metrics["qc_mostly_silence"] = silence > QC_MAX_SILENCE_PERCENT

    # F0 detection rate (if available)
    if f0_values is not None:
        f0_rate = calculate_f0_detection_rate(f0_values)
        metrics["f0_detection_rate"] = f0_rate
        metrics["qc_unvoiced"] = f0_rate < QC_MIN_F0_DETECTION_RATE
    else:
        metrics["f0_detection_rate"] = None
        metrics["qc_unvoiced"] = False

    # Overall pass/fail
    metrics["qc_passed"] = not any([
        metrics["qc_low_snr"],
        metrics["qc_clipping"],
        metrics["qc_mostly_silence"],
        metrics["qc_unvoiced"],
    ])

    # List of active flags
    flags = []
    if metrics["qc_low_snr"]:
        flags.append("low_snr")
    if metrics["qc_clipping"]:
        flags.append("clipping")
    if metrics["qc_mostly_silence"]:
        flags.append("mostly_silence")
    if metrics["qc_unvoiced"]:
        flags.append("unvoiced")
    metrics["qc_flags"] = flags

    return metrics


def generate_batch_qc_report(
    all_metrics: list[dict[str, Any]],
    total_files: int,
) -> dict[str, Any]:
    """
    Generate batch-level QC summary report.

    Args:
        all_metrics: List of per-file QC metrics
        total_files: Total number of files processed

    Returns:
        Batch QC report dictionary
    """
    passed = sum(1 for m in all_metrics if m.get("qc_passed", False))
    failed = len(all_metrics) - passed

    # Count specific flags
    flag_counts = {
        "low_snr": sum(1 for m in all_metrics if m.get("qc_low_snr", False)),
        "clipping": sum(1 for m in all_metrics if m.get("qc_clipping", False)),
        "mostly_silence": sum(1 for m in all_metrics if m.get("qc_mostly_silence", False)),
        "unvoiced": sum(1 for m in all_metrics if m.get("qc_unvoiced", False)),
    }

    # Calculate statistics
    snr_values = [m["snr_db"] for m in all_metrics if m.get("snr_db") is not None]

    return {
        "total_files": total_files,
        "processed_files": len(all_metrics),
        "passed_files": passed,
        "failed_files": failed,
        "pass_rate": passed / len(all_metrics) if all_metrics else 0,
        "flag_counts": flag_counts,
        "snr_mean": float(np.mean(snr_values)) if snr_values else None,
        "snr_std": float(np.std(snr_values)) if snr_values else None,
        "snr_min": float(np.min(snr_values)) if snr_values else None,
        "snr_max": float(np.max(snr_values)) if snr_values else None,
    }
