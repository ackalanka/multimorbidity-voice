- This project is designed to work with **Python 3.12** only.
- Running it with any other version of Python may cause compatibility issues with dependencies.
# Voice Biomarker Extraction Pipeline

A robust audio processing pipeline for extracting Mel-Frequency Cepstral Coefficients (MFCCs) from voice recordings, optimized for medical machine learning research.

## Key Features

- **Advanced Audio Preprocessing**  
  Noise reduction, spectral normalization, and adaptive voice activity detection
- **Comprehensive MFCC Extraction**  
  13 coefficients with 10 statistical measures per coefficient
- **Temporal Feature Analysis**  
  Delta and delta-delta coefficients for voice dynamics
- **Quality Control Metrics**  
  SNR estimation, duration tracking, and processing validation
- **Scalable Architecture**  
  Parallel processing for large datasets
- **Research-Ready Output**  
  CSV format with embedded metadata for longitudinal studies

## Technical Specifications

| Component              | Details                                                                 |
|------------------------|-------------------------------------------------------------------------|
| MFCC Parameters        | 13 coefficients, 128 mel bands, 50-8000Hz range                        |
| Preprocessing          | Noise reduction (spectral gating), pre-emphasis (0.97), peak normalization |
| Temporal Features      | Δ (delta) and Δ² (delta-delta) coefficients                            |
| Statistical Features   | Mean, variance, median, min/max, range, std, kurtosis, skewness, ZCR  |
| Output Format          | CSV with 143 features per recording + metadata columns                 |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Organize WAV files:
```bash
input_dir/
├── patient_001_2024-01-15.wav
├── patient_002_2024-02-20.wav
└── ...
```

2. Run feature extraction:
```bash
python mfcc_extractor.py
```

3. Output CSV contains:
```csv
filename,patient_id,recording_date,mfcc_01_mean,...,delta2_mean,snr_db,...
```

## Audio Preprocessing Pipeline

1. **Noise Reduction**  
   Spectral gating using first 500ms as noise profile (aggression=1.5)
   
2. **Voice Activity Detection**  
   Conservative trimming (-25dB threshold) with duration validation

3. **Spectral Normalization**  
   Pre-emphasis filter → Peak normalization → Mel spectrogram conversion

4. **MFCC Extraction**  
   13 coefficients from 128 mel bands (50-8000Hz)

5. **Feature Engineering**  
   10 statistical measures + temporal derivatives per coefficient

## Why This Pipeline Excels for ML

1. **Rich Feature Representation**  
   - 143 engineered features capture both spectral and temporal patterns
   - Kurtosis/skewness detect non-Gaussian voice characteristics
   - ZCR (Zero Crossing Rate) enhances time-domain analysis

2. **Noise-Robust Processing**  
   - Aggressive spectral gating reduces environmental noise contamination
   - Adaptive trimming ensures clean voice segments for analysis

3. **Temporal Dynamics**  
   Δ and Δ² coefficients model voice changes over time - critical for detecting subtle pathology progression

4. **Quality Control**  
   Embedded SNR and duration metrics enable intelligent data filtering:
   ```python
   df = df[df.processed_duration > 0.5]  # Filter short recordings
   df = df[df.snr_db > 15]               # Filter low-quality samples
   ```

5. **Longitudinal Readiness**  
   Native support for time-series analysis through:
   - Automatic date parsing from filenames
   - Patient-ID tracking
   - Processing parameter versioning

## Future Directions

1. **Real-Time Processing**  
   Web interface for instant voice analysis

2. **Deep Learning Integration**  
   TensorFlow/Keras ready feature format

3. **Multimodal Fusion**  
   Combine with clinical data via:
   ```python
   merged_data = pd.merge(features_df, clinical_df, on=["patient_id", "date"])
   ```

## License

MIT License - See [LICENSE](LICENSE) for details
