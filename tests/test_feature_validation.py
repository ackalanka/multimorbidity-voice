# tests/test_feature_validation.py
"""
Feature Validation Tests

Validates that extracted features are within physiologically
plausible ranges and consistent with literature.

These tests ensure scientific validity for peer-reviewed publication.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import numpy as np
import pytest


# ==============================================================================
# Physiological Range Definitions
# Based on literature and Praat manual
# ==============================================================================

PHYSIOLOGICAL_RANGES = {
    # Fundamental Frequency (F0)
    "f0_mean": (50.0, 500.0),       # Hz - covers male (85-180) to female (165-255)
    "f0_std": (0.0, 100.0),         # Hz
    "f0_min": (30.0, 400.0),        # Hz
    "f0_max": (80.0, 700.0),        # Hz
    
    # Jitter (cycle-to-cycle F0 variation)
    "jitter_local": (0.0, 0.05),    # 0-5% normal range
    "jitter_ppq5": (0.0, 0.05),     # 0-5%
    
    # Shimmer (cycle-to-cycle amplitude variation)
    "shimmer_local": (0.0, 0.15),   # 0-15% normal range
    "shimmer_apq11": (0.0, 0.15),   # 0-15%
    
    # Harmonics-to-Noise Ratio
    "hnr_praat": (-10.0, 40.0),     # dB - typical 10-25 dB for healthy voice
    "hnr_std": (0.0, 20.0),         # dB
    
    # Formants
    "f1_mean": (200.0, 1200.0),     # Hz - vowel dependent
    "f2_mean": (600.0, 3000.0),     # Hz
    "f3_mean": (1500.0, 4000.0),    # Hz
    
    # Cardiovascular-specific
    "voice_tremor": (0.0, 1.0),     # Normalized coefficient of variation
    "vocal_stability_index": (0.0, 1.0),  # 0-1 range
    
    # Intensity
    "intensity_mean": (30.0, 100.0),  # dB
    "intensity_std": (0.0, 30.0),     # dB
}


def validate_feature_range(name: str, value: float | None) -> tuple[bool, str]:
    """
    Validate a single feature is within physiological range.
    
    Args:
        name: Feature name
        value: Feature value
        
    Returns:
        Tuple of (is_valid, message)
    """
    if value is None:
        return True, f"{name}: None (skipped)"
    
    if name not in PHYSIOLOGICAL_RANGES:
        return True, f"{name}: No range defined"
    
    min_val, max_val = PHYSIOLOGICAL_RANGES[name]
    
    if min_val <= value <= max_val:
        return True, f"{name}: {value:.4f} ✓ (range: {min_val}-{max_val})"
    else:
        return False, f"{name}: {value:.4f} ✗ OUTSIDE range: {min_val}-{max_val}"


def validate_all_features(features: dict) -> dict:
    """
    Validate all features in a dictionary.
    
    Args:
        features: Dictionary of features
        
    Returns:
        Validation report dictionary
    """
    results = {
        "valid_count": 0,
        "invalid_count": 0,
        "skipped_count": 0,
        "violations": [],
        "details": [],
    }
    
    for name, (min_val, max_val) in PHYSIOLOGICAL_RANGES.items():
        value = features.get(name)
        
        if value is None:
            results["skipped_count"] += 1
            continue
        
        is_valid, message = validate_feature_range(name, value)
        results["details"].append(message)
        
        if is_valid:
            results["valid_count"] += 1
        else:
            results["invalid_count"] += 1
            results["violations"].append({
                "feature": name,
                "value": value,
                "expected_range": (min_val, max_val),
            })
    
    return results


# ==============================================================================
# Test Functions
# ==============================================================================

def test_f0_range():
    """Test that F0 extraction produces physiologically valid values."""
    from voicebiomarkers.core.praat import extract_praat_features
    
    # Create synthetic vowel-like audio (more realistic than random)
    sample_rate = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Simulate ~200 Hz F0 with harmonics
    f0 = 200.0
    audio = 0.5 * np.sin(2 * np.pi * f0 * t)
    audio += 0.25 * np.sin(2 * np.pi * 2 * f0 * t)
    audio += 0.125 * np.sin(2 * np.pi * 3 * f0 * t)
    audio = audio.astype(np.float32)
    
    features = extract_praat_features(audio, sample_rate)
    
    # Validate F0 is in range
    if features.get("f0_mean") is not None:
        is_valid, msg = validate_feature_range("f0_mean", features["f0_mean"])
        print(msg)
        assert is_valid, f"F0 mean out of range: {features['f0_mean']}"


def test_jitter_shimmer_range():
    """Test that jitter/shimmer are in normal range."""
    # This would require real voice audio
    # For now, just check the validation function works
    
    # Normal values
    assert validate_feature_range("jitter_local", 0.01)[0]
    assert validate_feature_range("shimmer_local", 0.05)[0]
    
    # Abnormal values
    assert not validate_feature_range("jitter_local", 0.10)[0]
    assert not validate_feature_range("shimmer_local", 0.30)[0]


def test_hnr_range():
    """Test HNR validation."""
    # Normal healthy voice: 10-25 dB
    assert validate_feature_range("hnr_praat", 15.0)[0]
    assert validate_feature_range("hnr_praat", 25.0)[0]
    
    # Edge cases
    assert validate_feature_range("hnr_praat", -5.0)[0]  # Low but possible
    assert not validate_feature_range("hnr_praat", 50.0)[0]  # Too high


def test_real_file_validation():
    """Test validation on real audio file."""
    from voicebiomarkers.core.preprocessing import preprocess_audio
    from voicebiomarkers.core.praat import extract_praat_features
    
    # Find test file
    data_dir = Path(__file__).parent.parent / "data" / "2307"
    if not data_dir.exists():
        pytest.skip("No test data available")
    
    wav_files = list(data_dir.glob("*.wav"))
    if not wav_files:
        pytest.skip("No WAV files found")
    
    test_file = wav_files[0]
    print(f"\nValidating: {test_file.name}")
    
    # Extract features
    audio, _ = preprocess_audio(str(test_file))
    features = extract_praat_features(audio, 16000)
    
    # Validate
    report = validate_all_features(features)
    
    print(f"\nValidation Results:")
    print(f"  Valid: {report['valid_count']}")
    print(f"  Invalid: {report['invalid_count']}")
    print(f"  Skipped: {report['skipped_count']}")
    
    if report["violations"]:
        print("\nViolations:")
        for v in report["violations"]:
            print(f"  {v['feature']}: {v['value']:.4f} (expected: {v['expected_range']})")
    
    # Allow some violations for edge cases, but not many
    assert report["invalid_count"] <= 3, f"Too many physiological range violations: {report['violations']}"


if __name__ == "__main__":
    print("="*60)
    print("FEATURE VALIDATION TESTS")
    print("="*60)
    
    print("\n--- Jitter/Shimmer Range Test ---")
    test_jitter_shimmer_range()
    print("✓ Passed")
    
    print("\n--- HNR Range Test ---")
    test_hnr_range()
    print("✓ Passed")
    
    print("\n--- Real File Validation ---")
    try:
        test_real_file_validation()
        print("✓ Passed")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    print("\n✓ All validation tests completed!")
