# tests/test_mfcc.py
"""Unit tests for MFCC extraction module."""

import numpy as np
import pytest


def test_extract_mfcc(sample_audio, sample_rate):
    """Test basic MFCC extraction."""
    from voicebiomarkers.core.mfcc import extract_mfcc
    
    mfcc = extract_mfcc(sample_audio, sample_rate, n_mfcc=13)
    
    assert isinstance(mfcc, np.ndarray)
    assert mfcc.shape[0] == 13


def test_extract_mfcc_deltas(sample_audio, sample_rate):
    """Test delta coefficient extraction."""
    from voicebiomarkers.core.mfcc import extract_mfcc, extract_mfcc_deltas
    
    mfcc = extract_mfcc(sample_audio, sample_rate, n_mfcc=13)
    delta, delta2 = extract_mfcc_deltas(mfcc)
    
    assert delta.shape == mfcc.shape
    assert delta2.shape == mfcc.shape


def test_extract_mfcc_features(sample_audio, sample_rate):
    """Test full MFCC feature extraction."""
    from voicebiomarkers.core.mfcc import extract_mfcc_features
    
    features = extract_mfcc_features(sample_audio, sample_rate, n_mfcc=13)
    
    assert isinstance(features, dict)
    assert "mfcc_01_mean" in features
    assert "delta_01_mean" in features
    assert "delta2_01_mean" in features
    assert features["mfcc_extraction_status"] == "success"


def test_mfcc_statistics():
    """Test coefficient statistics calculation."""
    from voicebiomarkers.core.mfcc import calculate_coefficient_statistics
    
    coeff = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    stats = calculate_coefficient_statistics(coeff, "test")
    
    assert stats["test_mean"] == 3.0
    assert stats["test_min"] == 1.0
    assert stats["test_max"] == 5.0
    assert stats["test_range"] == 4.0
