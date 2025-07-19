- This project is designed to work with Python 3.12 only.
- Running it with any other version of Python may cause compatibility issues with dependencies.

# Voice Biomarker Extraction Pipeline

A comprehensive toolkit for extracting acoustic features from voice recordings, optimized for medical machine learning research

![Python Version](https://img.shields.io/badge/python-3.12%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Technical Specifications](#technical-specifications)
- [Data Requirements](#data-requirements)
- [Output Format](#output-format)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

## Features

- **Audio Processing Pipeline**
  - Automatic voice activity detection
  - Noise reduction using spectral gating
  - Pre-emphasis filtering
  - Signal normalization

- **Feature Extraction**
  - **MFCC Features** (13 coefficients with delta and delta-delta)
    - Statistical measures: mean, std, kurtosis, skewness, range
  - **Prosodic Features** (via Praat/parselmouth)
    - Fundamental frequency (F0) statistics
    - Jitter (local, ppq5)
    - Shimmer (local, apq11)
    - Harmonic-to-Noise Ratio (HNR)
    - Formant dispersion
    - Spectral tilt

- **Advanced Functionality**
  - Multiprocessing support
  - Automatic metadata extraction from filenames
  - Comprehensive error logging
  - Quality metrics (SNR, duration checks)
  - CSV output with structured columns

## Installation
### Steps
1. Clone repository:
```bash
git clone https://github.com/ackalanka/multimorbidity-voice
```

2. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/MacOS
   venv\Scripts\activate    # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Prepare Input**
   - Place WAV files in `data/raw` directory
   - Follow naming convention: `[PatientID]-[DDMMYYYY]-[HH-MM].wav`

2. **Run Feature Extraction**:
   ```bash
   python scripts/extractor_v2.py
   ```

3. **Output**:
   - Results saved to `voice_features.csv`
   - Logs stored in `feature_extraction.log`

## Configuration

Modify these parameters in `extractor_v2.py`:

| Parameter                      | Default Value     | Description                          |
|--------------------------------|-------------------|--------------------------------------|
| `INPUT_DIR`                    | `data/raw`        | Input directory for WAV files        |
| `OUTPUT_CSV`                   | `voice_features.csv` | Output CSV file name             |
| `SAMPLE_RATE`                  | 16000             | Target sampling rate                 |
| `N_MFCC`                       | 13                | Number of MFCC coefficients          |
| `NOISE_REDUCTION_AGGRESSION`   | 1.5               | Noise reduction intensity (0-2)      |
| `MIN_VOICE_DURATION`           | 0.5               | Minimum valid voice duration (sec)   |

## Technical Specifications

### Audio Processing
- **Resampling**: Librosa's resampling with anti-aliasing
- **Noise Reduction**: Spectral gating via noisereduce
- **MFCC Parameters**:
  - 40 mel bands (50-8000 Hz)
  - 2048-point FFT with 512 hop length
  - Pre-emphasis coefficient: 0.97

### Praat Features
| Feature Category      | Specific Measures                          |
|-----------------------|--------------------------------------------|
| Fundamental Frequency | Mean, STD, Entropy (quantile-based)        |
| Perturbation          | Jitter (local, ppq5), Shimmer (local, apq11)|
| Spectral              | HNR, Formant Dispersion, Spectral Tilt     |

## Data Requirements

### Input Format
- 16-bit PCM WAV files
- Mono channel
- Recommended duration: 1-5 seconds
- Sample rate: ≥16kHz

### File Naming Convention

The script automatically extracts structured metadata from each .wav filename based on a specific naming convention. This metadata is critical for downstream analysis and supervised learning (e.g., grouping by gender or age).

#### Expected Filename Format: 
```
<ID>-<Gender>-<Age>-<DD.MM.YYYY>-<HH>-<MM>.wav
```

#### Example: 
```
2303-F-30-29.11.2023-20-08.wav
```

#### Parsed Metadata Fields:

| Field            | Description                                       |
| ---------------- | ------------------------------------------------- |
| `patient_id`     | Numeric identifier of the participant             |
| `gender`         | Participant gender (`F` for female, `M` for male) |
| `age`            | Age in years (validated to be within 0–120)       |
| `recording_date` | Date in ISO format (e.g., `2023-11-29`)           |
| `recording_time` | Time in 24-hour format (e.g., `20:08:00`)         |

#### Validation and Error Handling

- **Pattern mismatch**: Filenames that do not conform to the expected structure are flagged.
- **Invalid values**: Incorrect age, date, or time values are logged, and an optional `filename_error` field is added for diagnostics.
- **Logging**: All issues are written to the log file and shown in the console during processing.

## Output Format

CSV file with columns:

- **Metadata**
  - patient_id, recording_date, recording_time
  - filename, original_duration, processed_duration

- **MFCC Features**
  - mfcc_01_mean to mfcc_13_range (65 columns)
  - delta_01_mean to delta2_13_range (130 columns)

- **Prosodic Features**
  - f0_mean, f0_std, f0_entropy
  - jitter_local, jitter_ppq5
  - shimmer_local, shimmer_apq11
  - hnr_praat, formant_dispersion, spectral_tilt

- **Quality Metrics**
  - snr_db, trim_start, trim_end
  - parselmouth_status, praat_errors

## Troubleshooting

**Common Issues**:
1. **No WAV files found**
   - Verify files are in `data/raw`
   - Check file extensions (.wav, lowercase)

2. **Praat Command Errors**
   - Ensure minimum audio duration (0.3s after trimming)
   - Check for silent/quiet recordings

3. **Memory Issues**
   - Reduce number of parallel processes
   - Use `cpu_count()//2` in Pool initialization

**Logging**:
- Detailed logs in `feature_extraction.log`
- Set logging level in code for debugging:
  ```python
  logging.basicConfig(level=logging.DEBUG, ...)
  ```

## Contributing

We welcome contributions! Please follow these guidelines:

1. **Reporting Issues**
   - Use GitHub Issues template
   - Include error logs and sample file if possible

2. **Feature Requests**
   - Describe use case and proposed implementation

3. **Development**:
   ```bash
   git checkout -b feature/your-feature
   # Make changes
   pytest tests/  # Add tests for new features
   git push origin feature/your-feature
   ```

**Coding Standards**:
- PEP8 compliance
- Type hints for public functions
- Docstrings for all modules
- Unit tests for core functionality

## License

MIT License - See [LICENSE](LICENSE) for details.

## Acknowledgments

- **Librosa** for MFCC extraction
- **Parselmouth** for Praat integration
- **noisereduce** for audio cleaning

## Contact

**Akalanka Ranasinghe**  
- GitHub: [@ackalanka](https://github.com/ackalanka)  
- Email: akalankar98@gmail.com  

---

**Note**: Clinical validation required for diagnostic use. This tool is intended for research purposes only.
