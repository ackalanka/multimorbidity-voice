# tests/test_praat.py
"""Unit tests for Praat feature extraction."""

import numpy as np
import pytest


def test_parselmouth_available():
    """Test that Parselmouth is available."""
    from voicebiomarkers.core.praat import PARSELMOUTH_AVAILABLE
    
    assert PARSELMOUTH_AVAILABLE, "Parselmouth not installed"


def test_extract_praat_features(sample_audio, sample_rate):
    """Test Praat feature extraction."""
    from voicebiomarkers.core.praat import extract_praat_features
    
    features = extract_praat_features(sample_audio, sample_rate)
    
    assert isinstance(features, dict)
    assert "parselmouth_status" in features


def test_f0_extraction(sample_audio, sample_rate):
    """Test F0 is extracted."""
    from voicebiomarkers.core.praat import extract_praat_features
    
    features = extract_praat_features(sample_audio, sample_rate)
    
    # F0 should be present
    assert "f0_mean" in features


def test_jitter_shimmer(sample_audio, sample_rate):
    """Test jitter/shimmer extraction."""
    from voicebiomarkers.core.praat import extract_praat_features
    
    features = extract_praat_features(sample_audio, sample_rate)
    
    assert "jitter_local" in features
    assert "shimmer_local" in features


def test_hnr_extraction(sample_audio, sample_rate):
    """Test HNR extraction."""
    from voicebiomarkers.core.praat import extract_praat_features
    
    features = extract_praat_features(sample_audio, sample_rate)
    
    assert "hnr_praat" in features
