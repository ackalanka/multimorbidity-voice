# Methods: Voice Feature Extraction

> **Document Version**: 1.0  
> **Pipeline Version**: voicebiomarkers 1.0.0  
> **Last Updated**: 2025-12-28

This document provides a journal-ready methods section for peer-reviewed publication.

---

## 2.X Voice Feature Extraction

Voice recordings were processed using the voicebiomarkers pipeline (v1.0.0, https://github.com/[repository]). All audio files were resampled to 16 kHz mono and preprocessed using spectral noise reduction (Sainburg et al., 2020) with aggression parameter 1.5. Voice activity detection was performed using energy-based trimming (25 dB threshold below peak amplitude).

### 2.X.1 Audio Preprocessing

Audio preprocessing consisted of the following steps:

1. **Resampling**: All recordings were standardized to 16,000 Hz sample rate
2. **Noise Reduction**: Spectral gating with the first 0.5 seconds as noise profile (noisereduce library v3.0.3)
3. **Voice Activity Detection**: Energy-based trimming using librosa.effects.trim() with 25 dB threshold
4. **Pre-emphasis**: High-frequency boost using coefficient 0.97

Files with processed duration less than 0.5 seconds were excluded from analysis.

### 2.X.2 MFCC Feature Extraction

Mel-frequency cepstral coefficients were extracted using librosa v0.10.2 with the following parameters:

| Parameter | Value |
|-----------|-------|
| Number of MFCCs | 72 |
| Mel bands | 128 |
| FFT window | 2048 samples (128 ms) |
| Hop length | 512 samples (32 ms) |
| Frequency range | 50-8000 Hz |
| Pre-emphasis coefficient | 0.97 |

For each of the 72 MFCC coefficients, we computed 7 statistical measures:
- Mean, standard deviation, kurtosis, skewness
- Maximum, minimum, range

Delta (velocity) and delta-delta (acceleration) coefficients were computed using librosa.feature.delta() with 5 statistical measures each (mean, std, max, min, range).

**Total MFCC features per recording**: 72 × (7 + 5 + 5) = **1,224 features**

### 2.X.3 Acoustic Feature Extraction

Fundamental frequency (F0), perturbation measures, and formant frequencies were extracted using Praat (Boersma & Weenink, 2023) via the Parselmouth Python interface v0.5.0 with the following parameters:

#### Pitch (F0) Detection
| Parameter | Value |
|-----------|-------|
| Minimum pitch | 50 Hz |
| Maximum pitch | 500 Hz |
| Time step | Auto |
| Voicing threshold | 0.45 |
| Octave cost | 0.01 |

#### Jitter and Shimmer
Period floor and ceiling were set to 0.0001 and 0.02 seconds respectively, with period factor 1.3 and amplitude factor 1.6, following Praat defaults.

#### Harmonics-to-Noise Ratio
HNR was computed using cross-correlation with time step 0.01 seconds, minimum pitch 50 Hz, silence threshold 0.1, and periods per window of 1.0.

#### Formant Frequencies
Formants were extracted using Burg's algorithm with 5 formants up to 5500 Hz, window length 0.025 seconds, and pre-emphasis from 50 Hz.

### 2.X.4 Cardiovascular-Specific Features

In addition to standard acoustic features, we extracted the following cardiovascular-relevant voice biomarkers:

| Feature | Description | Calculation Method |
|---------|-------------|-------------------|
| Voice Tremor | F0 modulation coefficient | std(ΔF0) / mean(F0) |
| Vocal Stability Index | Overall voice stability | 1 / (1 + CV of F0) |
| Spectral Tilt | Breathiness indicator | Energy slope 0-1kHz vs 1-4kHz |
| Intensity Variation | Amplitude consistency | std(intensity) |

### 2.X.5 Quality Control

Each recording underwent automated quality control with the following thresholds:

| Metric | Threshold | Action |
|--------|-----------|--------|
| Signal-to-Noise Ratio | ≥ 15 dB | Flag if below |
| Duration | 0.5-180 s | Exclude if outside |
| Clipping | < 1% | Flag if above |
| Silence | < 90% | Flag if above |
| F0 Detection Rate | > 50% | Flag if below |

Files failing multiple QC criteria were manually reviewed before exclusion.

### 2.X.6 Reproducibility

To ensure reproducibility:

1. **Environment**: All analyses were conducted in Docker containers with pinned dependency versions
2. **Random Seeds**: NumPy random seed set to 42; BLAS threading limited to single-thread
3. **Versioning**: All library versions (librosa 0.10.2, parselmouth 0.5.0, numpy 2.1.3) were frozen
4. **Audit Trail**: SHA-256 hashes of input files and output CSVs were recorded

The complete pipeline code and analysis scripts are available at [repository URL].

---

## References

Boersma, P., & Weenink, D. (2023). Praat: doing phonetics by computer [Computer program]. Version 6.3.14. http://www.praat.org/

Sainburg, T., Thielk, M., & Bhargava, R. (2020). noisereduce: A noise reduction Python library. https://github.com/timsainb/noisereduce/

McFee, B., et al. (2015). librosa: Audio and Music Signal Analysis in Python. Proceedings of the 14th Python in Science Conference, 18-24.

Jadoul, Y., Thompson, B., & de Boer, B. (2018). Introducing Parselmouth: A Python interface to Praat. Journal of Phonetics, 71, 1-15.
