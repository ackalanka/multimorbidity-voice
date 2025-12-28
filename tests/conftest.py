# tests/conftest.py
"""Pytest fixtures for voice biomarkers tests."""

import sys
from pathlib import Path

import numpy as np
import pytest


# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture
def sample_audio():
    """Generate synthetic audio for testing (2 seconds at 16kHz)."""
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Simulate a vowel sound with F0 ~ 200 Hz
    f0 = 200.0
    audio = 0.5 * np.sin(2 * np.pi * f0 * t)
    audio += 0.25 * np.sin(2 * np.pi * 2 * f0 * t)
    audio += 0.125 * np.sin(2 * np.pi * 3 * f0 * t)
    audio += 0.01 * np.random.randn(len(audio))  # Add some noise
    
    return audio.astype(np.float32)


@pytest.fixture
def sample_rate():
    """Return standard sample rate."""
    return 16000


@pytest.fixture
def test_audio_path(tmp_path):
    """Create a temporary test audio file."""
    import soundfile as sf
    
    # Generate audio
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = (0.5 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    
    # Save to temp file
    audio_path = tmp_path / "test_audio.wav"
    sf.write(audio_path, audio, sample_rate)
    
    return str(audio_path)


@pytest.fixture
def data_dir():
    """Return path to test data directory."""
    return Path(__file__).parent.parent / "data" / "2307"
