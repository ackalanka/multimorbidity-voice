# tests/test_quality.py
"""Unit tests for quality control module."""

import numpy as np
import pytest


def test_calculate_snr():
    """Test SNR calculation."""
    from voicebiomarkers.core.quality import calculate_snr
    
    # Create signal with known SNR
    original = np.concatenate([
        0.1 * np.random.randn(1000),  # Noise
        np.sin(np.linspace(0, 10*np.pi, 1000))  # Signal
    ])
    clean = np.concatenate([
        np.zeros(1000),
        np.sin(np.linspace(0, 10*np.pi, 1000))
    ])
    
    snr = calculate_snr(original, clean)
    
    assert snr is not None
    assert isinstance(snr, float)


def test_calculate_clipping_percentage():
    """Test clipping detection."""
    from voicebiomarkers.core.quality import calculate_clipping_percentage
    
    # No clipping
    audio = 0.5 * np.ones(1000)
    assert calculate_clipping_percentage(audio) < 1.0
    
    # Full clipping
    audio = np.ones(1000)
    assert calculate_clipping_percentage(audio) == 100.0


def test_calculate_quality_metrics(sample_audio, sample_rate):
    """Test full QC metric calculation."""
    from voicebiomarkers.core.quality import calculate_quality_metrics
    
    metrics = calculate_quality_metrics(sample_audio, sample_audio, sample_rate)
    
    assert "snr_db" in metrics
    assert "clipping_percent" in metrics
    assert "qc_passed" in metrics


def test_batch_qc_report():
    """Test batch QC report generation."""
    from voicebiomarkers.core.quality import generate_batch_qc_report
    
    metrics = [
        {"qc_passed": True, "qc_low_snr": False, "snr_db": 20.0},
        {"qc_passed": False, "qc_low_snr": True, "snr_db": 10.0},
    ]
    
    report = generate_batch_qc_report(metrics, 2)
    
    assert report["total_files"] == 2
    assert report["passed_files"] == 1
    assert report["failed_files"] == 1
