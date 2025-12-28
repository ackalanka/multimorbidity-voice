# tests/test_preprocessing.py
"""Unit tests for preprocessing module."""

import numpy as np
import pytest


def test_load_audio(test_audio_path):
    """Test audio loading."""
    from voicebiomarkers.core.preprocessing import load_audio
    
    audio, sr = load_audio(test_audio_path)
    
    assert isinstance(audio, np.ndarray)
    assert sr == 16000
    assert len(audio) > 0


def test_reduce_noise(sample_audio, sample_rate):
    """Test noise reduction."""
    from voicebiomarkers.core.preprocessing import reduce_noise
    
    clean = reduce_noise(sample_audio, sample_rate)
    
    assert isinstance(clean, np.ndarray)
    assert len(clean) == len(sample_audio)


def test_trim_silence(sample_audio):
    """Test silence trimming."""
    from voicebiomarkers.core.preprocessing import trim_silence
    
    trimmed, start, end = trim_silence(sample_audio)
    
    assert isinstance(trimmed, np.ndarray)
    assert start >= 0
    assert end <= len(sample_audio)


def test_preprocess_audio(test_audio_path):
    """Test full preprocessing pipeline."""
    from voicebiomarkers.core.preprocessing import preprocess_audio
    
    audio, meta = preprocess_audio(test_audio_path)
    
    assert isinstance(audio, np.ndarray)
    assert "original_duration" in meta
    assert "processed_duration" in meta
    assert meta["processed_duration"] > 0
