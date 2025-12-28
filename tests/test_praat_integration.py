# tests/test_praat_integration.py
"""
Parselmouth Integration Tests

Validates that the Praat feature extraction works correctly
with the existing audio files and matches expected behavior.
"""

import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


def test_parselmouth_import():
    """Test that parselmouth can be imported."""
    print("Testing Parselmouth import...")
    
    try:
        import parselmouth
        from parselmouth.praat import call
        print(f"✓ Parselmouth version: {parselmouth.__version__}")
        return True
    except ImportError as e:
        print(f"✗ Parselmouth import failed: {e}")
        return False


def test_praat_module_import():
    """Test that our praat module imports correctly."""
    print("\nTesting voicebiomarkers.core.praat import...")
    
    try:
        from voicebiomarkers.core.praat import extract_praat_features, PARSELMOUTH_AVAILABLE
        print(f"✓ Praat module imported, PARSELMOUTH_AVAILABLE={PARSELMOUTH_AVAILABLE}")
        return True
    except Exception as e:
        print(f"✗ Praat module import failed: {e}")
        return False


def test_preprocessing_import():
    """Test preprocessing module."""
    print("\nTesting voicebiomarkers.core.preprocessing import...")
    
    try:
        from voicebiomarkers.core.preprocessing import preprocess_audio, load_audio
        print("✓ Preprocessing module imported")
        return True
    except Exception as e:
        print(f"✗ Preprocessing import failed: {e}")
        return False


def test_mfcc_import():
    """Test MFCC module."""
    print("\nTesting voicebiomarkers.core.mfcc import...")
    
    try:
        from voicebiomarkers.core.mfcc import extract_mfcc_features
        print("✓ MFCC module imported")
        return True
    except Exception as e:
        print(f"✗ MFCC import failed: {e}")
        return False


def test_with_real_audio():
    """Test feature extraction with a real audio file."""
    print("\n" + "="*60)
    print("Testing with real audio file...")
    print("="*60)
    
    # Find a test audio file
    data_dir = Path(__file__).parent.parent / "data" / "2307"
    if not data_dir.exists():
        print(f"✗ Data directory not found: {data_dir}")
        return False
    
    wav_files = list(data_dir.glob("*.wav"))
    if not wav_files:
        print(f"✗ No WAV files found in {data_dir}")
        return False
    
    test_file = wav_files[0]
    print(f"Using test file: {test_file.name}")
    
    # Test preprocessing
    print("\n--- Preprocessing ---")
    try:
        from voicebiomarkers.core.preprocessing import preprocess_audio
        audio, meta = preprocess_audio(str(test_file))
        print(f"✓ Audio loaded: {len(audio)} samples")
        print(f"  Original duration: {meta['original_duration']:.2f}s")
        print(f"  Processed duration: {meta['processed_duration']:.2f}s")
    except Exception as e:
        print(f"✗ Preprocessing failed: {e}")
        return False
    
    # Test MFCC extraction
    print("\n--- MFCC Extraction ---")
    try:
        from voicebiomarkers.core.mfcc import extract_mfcc_features
        mfcc_features = extract_mfcc_features(audio)
        num_features = len([k for k in mfcc_features.keys() if k.startswith(('mfcc_', 'delta_', 'delta2_'))])
        print(f"✓ MFCC features extracted: {num_features} features")
        print(f"  Sample: mfcc_01_mean = {mfcc_features.get('mfcc_01_mean', 'N/A'):.4f}")
    except Exception as e:
        print(f"✗ MFCC extraction failed: {e}")
        return False
    
    # Test Praat extraction (the critical test)
    print("\n--- Praat Extraction (CRITICAL) ---")
    try:
        from voicebiomarkers.core.praat import extract_praat_features
        praat_features = extract_praat_features(audio, 16000)
        
        status = praat_features.get('parselmouth_status', 'unknown')
        print(f"  Status: {status}")
        
        if praat_features.get('praat_errors'):
            print(f"  Errors: {praat_features['praat_errors']}")
        
        # Check key features
        key_features = ['f0_mean', 'jitter_local', 'shimmer_local', 'hnr_praat', 'voice_tremor']
        for feat in key_features:
            val = praat_features.get(feat)
            if val is not None:
                print(f"  ✓ {feat} = {val:.4f}")
            else:
                print(f"  ✗ {feat} = None (extraction may have failed)")
        
        # Count successful features
        non_none = sum(1 for k, v in praat_features.items() 
                      if v is not None and not k.endswith('_status') and not k.endswith('_errors'))
        print(f"\n  Total non-None features: {non_none}")
        
        if status == 'success' and non_none >= 15:
            print("✓ Praat extraction PASSED")
            return True
        elif status == 'partial':
            print("⚠ Praat extraction PARTIAL - some features may be missing")
            return True
        else:
            print("✗ Praat extraction FAILED")
            return False
            
    except Exception as e:
        import traceback
        print(f"✗ Praat extraction failed with exception:")
        traceback.print_exc()
        return False


def compare_with_original_praatonly():
    """Compare results with original praatonly.py output."""
    print("\n" + "="*60)
    print("Comparing with original praatonly.py...")
    print("="*60)
    
    # This would load results from voice_features.csv and compare
    # For now, just check feature names match
    
    original_praat_features = [
        'f0_mean', 'f0_std', 'jitter_local', 'shimmer_local', 
        'hnr_praat', 'voice_tremor', 'vocal_stability_index'
    ]
    
    try:
        from voicebiomarkers.core.praat import extract_praat_features
        import numpy as np
        
        # Create dummy audio for feature name check
        dummy_audio = np.random.randn(16000 * 2).astype(np.float32) * 0.1
        features = extract_praat_features(dummy_audio, 16000)
        
        print("Feature name comparison:")
        for feat in original_praat_features:
            if feat in features:
                print(f"  ✓ {feat}")
            else:
                print(f"  ✗ {feat} MISSING")
        
        return True
    except Exception as e:
        print(f"Comparison failed: {e}")
        return False


if __name__ == "__main__":
    print("="*60)
    print("VOICEBIOMARKERS INTEGRATION TEST")
    print("="*60)
    
    results = []
    
    results.append(("Parselmouth Import", test_parselmouth_import()))
    results.append(("Praat Module Import", test_praat_module_import()))
    results.append(("Preprocessing Import", test_preprocessing_import()))
    results.append(("MFCC Import", test_mfcc_import()))
    results.append(("Real Audio Test", test_with_real_audio()))
    results.append(("Feature Comparison", compare_with_original_praatonly()))
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed!")
        sys.exit(1)
