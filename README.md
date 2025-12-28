# Voice Biomarkers Extraction Pipeline

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/[repo]/voicebiomarkers/actions/workflows/ci.yml/badge.svg)](https://github.com/[repo]/voicebiomarkers/actions)

A **scientifically rigorous, reproducible pipeline** for extracting acoustic voice biomarkers from audio recordings for **cardiovascular risk assessment research** and **multimorbidity prediction**.

---

## Table of Contents

1. [Project Goal](#project-goal)
2. [Scientific Background](#scientific-background)
3. [Why Reproducibility Matters](#why-reproducibility-matters)
4. [Quality Control Framework](#quality-control-framework)
5. [Features Extracted](#features-extracted)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Project Structure](#project-structure)
9. [For Researchers](#for-researchers)
10. [Citation](#citation)

---

## Project Goal

### Primary Objective

Extract **~1,300 acoustic voice biomarkers** from patient voice recordings to build machine learning models that predict:

- **Cardiovascular disease risk** (hypertension, heart failure, arrhythmia)
- **Multimorbidity patterns** (co-occurring chronic conditions)
- **Disease progression** over time

### Why Voice?

The human voice carries physiological signatures of:

| System | Voice Marker | Clinical Relevance |
|--------|-------------|-------------------|
| **Cardiovascular** | Voice tremor, pitch instability | Vagal tone, autonomic dysfunction |
| **Respiratory** | Spectral tilt, HNR | Pulmonary function |
| **Neurological** | Jitter, shimmer | Motor control, Parkinson's early detection |
| **Psychological** | Pitch variability, speech rate | Depression, anxiety screening |

Voice analysis is:
- ✅ **Non-invasive** - No blood draws, imaging, or procedures
- ✅ **Low-cost** - Only requires a smartphone microphone
- ✅ **Scalable** - Can be collected remotely via telehealth
- ✅ **Longitudinal** - Easy to repeat for monitoring

---

## Scientific Background

### Voice as a Biomarker

Research has demonstrated correlations between acoustic features and cardiovascular health:

1. Voice signal characteristics have been shown to be statistically associated with the presence of coronary artery disease (CAD) when analyzed using specific acoustic features such as Mel-Frequency Cepstral Coefficients. [1]

2. Acoustic voice and speech features change with clinical status in patients with acute decompensated heart failure (ADHF): after treatment, patients’ voices tended to show more stable phonation, a creakier quality, faster speech rates, and longer phrases compared to before treatment. [2]

3. Acoustic voice and speech features change with clinical status in patients with acute decompensated heart failure (ADHF): after treatment, patients’ voices tended to show more stable phonation, a creakier quality, faster speech rates, and longer phrases compared to before treatment. [3]

4. Boersma & Weenink’s work on Praat provides a methodological framework and software tools for analyzing acoustic voice parameters (including fundamental frequency, HNR, and perturbation measures) in research and clinical voice evaluation [4]

### This Pipeline

This pipeline implements a **standardized, validated extraction protocol** to enable:

- Multi-site clinical studies with consistent methodology
- Machine learning model development with reproducible training data
- Longitudinal analysis with comparable metrics across time

---

## Why Reproducibility Matters

### The Reproducibility Crisis

Medical research faces a **reproducibility crisis** - over 70% of studies cannot be replicated. For voice biomarker research to be clinically useful, we must ensure:

| Requirement | Our Solution |
|-------------|--------------|
| **Bit-exact results** | Fixed random seeds, pinned dependencies |
| **Environment consistency** | Docker containerization |
| **Audit trail** | SHA-256 hashing of inputs/outputs |
| **Version tracking** | Git commit in provenance record |
| **Parameter transparency** | All constants documented with literature citations |

### Implementation Details

```python
# Random seed locking (reproducibility.py)
np.random.seed(42)
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # Force deterministic BLAS

# File integrity verification
sha256_hash = hashlib.sha256(audio_bytes).hexdigest()

# Environment capture
provenance = {
    "librosa_version": "0.10.2",
    "parselmouth_version": "0.5.0",
    "numpy_version": "2.1.3",
    "git_commit": "abc123...",
    "extraction_timestamp": "2025-12-28T12:00:00Z"
}
```

### Why This Matters for Publication

Peer-reviewed journals increasingly require:
- **Data availability statements** ✅ (provenance records)
- **Code availability** ✅ (GitHub repository)
- **Reproducible methods** ✅ (Docker + pinned dependencies)
- **Registered parameters** ✅ (constants.py with citations)

---

## Quality Control Framework

### Per-File QC Metrics

Every audio file is evaluated against **5 quality thresholds**:

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| **Signal-to-Noise Ratio** | ≥ 15 dB | Below this, noise contaminates features |
| **Duration** | 0.5 - 180 s | Too short = insufficient data; too long = memory issues |
| **Clipping** | < 1% samples | Clipped audio distorts spectral features |
| **Silence** | < 90% | Mostly silence = no voice to analyze |
| **F0 Detection Rate** | > 50% | Unvoiced = cannot extract pitch features |

### QC Flags

Each file receives boolean flags:
```python
{
    "qc_passed": True,
    "qc_low_snr": False,
    "qc_clipping": False,
    "qc_mostly_silence": False,
    "qc_unvoiced": False,
    "qc_flags": []  # List of active issues
}
```

### Batch-Level QC Report

After processing, a summary report is generated:

```json
{
    "total_files": 727,
    "processed_files": 720,
    "passed_files": 698,
    "failed_files": 22,
    "pass_rate": 0.97,
    "flag_counts": {
        "low_snr": 15,
        "clipping": 3,
        "mostly_silence": 4,
        "unvoiced": 5
    },
    "snr_mean": 28.5,
    "snr_std": 8.2
}
```

---

## Features Extracted

### Overview

| Category | Count | Description |
|----------|-------|-------------|
| **MFCC** | 1,224 | Mel-frequency cepstral coefficients + deltas |
| **Praat/Acoustic** | ~30 | F0, jitter, shimmer, HNR, formants |
| **Cardiovascular-specific** | 5 | Tremor, stability, spectral tilt |
| **QC Metrics** | 10 | SNR, duration, flags |
| **Metadata** | 6 | Patient ID, age, gender, date |
| **Total** | **~1,275** | Per recording |

### MFCC Features (1,224)

- 72 MFCC coefficients × 7 statistics = 504
- 72 delta coefficients × 5 statistics = 360  
- 72 delta-delta coefficients × 5 statistics = 360

Statistics: mean, std, kurtosis, skewness, max, min, range

### Acoustic Features (~30)

| Feature | Unit | Normal Range | Clinical Significance |
|---------|------|--------------|----------------------|
| f0_mean | Hz | 85-255 | Fundamental frequency |
| f0_std | Hz | 10-50 | Pitch variability |
| jitter_local | % | < 1% | Cycle-to-cycle F0 variation |
| shimmer_local | % | < 3% | Cycle-to-cycle amplitude variation |
| hnr_praat | dB | 10-25 | Voice quality |
| voice_tremor | ratio | < 0.3 | Cardiovascular marker |
| vocal_stability_index | 0-1 | > 0.7 | Overall stability |

---

## Installation

### Option 1: Local Installation

```bash
# Clone repository
git clone https://github.com/ackalanka/multimorbidity-voice.git
cd multimorbidity-voice

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# .\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Docker (Recommended for Reproducibility)

```bash
# Build container
docker build -t voicebiomarkers:1.0.0 .

# Run extraction
docker run -v $(pwd)/data:/app/data -v $(pwd)/output:/app/output \
    voicebiomarkers:1.0.0 \
    python scripts/extract_features.py -i data/ -o output/
```

---

## Usage

### Basic Extraction

```bash
# Extract all features from audio directory
python scripts/extract_features.py -i data/recordings/ -o output/

# With verbose logging
python scripts/extract_features.py -i data/recordings/ -o output/ -v
```

### Output Files

| File | Contents |
|------|----------|
| `voice_features.csv` | All extracted features (1 row per file) |
| `qc_report.json` | Batch quality control summary |
| `provenance.json` | Reproducibility metadata |

### Programmatic Usage

```python
from voicebiomarkers.core.preprocessing import preprocess_audio
from voicebiomarkers.core.mfcc import extract_mfcc_features
from voicebiomarkers.core.praat import extract_praat_features

# Preprocess audio
audio, metadata = preprocess_audio("patient_001.wav")

# Extract features
mfcc = extract_mfcc_features(audio, sample_rate=16000)
praat = extract_praat_features(audio, sample_rate=16000)

# Combine
features = {**metadata, **mfcc, **praat}
```

---

## Project Structure

```
voicebiomarkers/
├── src/voicebiomarkers/          # Main package
│   ├── core/                     # Feature extraction
│   │   ├── preprocessing.py      # Audio loading, noise reduction, VAD
│   │   ├── mfcc.py               # MFCC + delta extraction
│   │   ├── praat.py              # Parselmouth/Praat features
│   │   └── quality.py            # QC metrics
│   ├── config/                   # Configuration
│   │   ├── constants.py          # Scientific parameters (documented)
│   │   └── settings.py           # Pydantic settings
│   ├── io/                       # Input/Output
│   │   ├── audio.py              # Audio loading
│   │   ├── metadata.py           # Filename parsing
│   │   └── export.py             # CSV/JSON export
│   └── utils/                    # Utilities
│       ├── logging.py            # Structured logging
│       └── reproducibility.py    # Hashing, provenance
├── scripts/
│   └── extract_features.py       # CLI entry point
├── tests/                        # Test suite (26 tests)
├── docs/
│   ├── methods.md                # Journal-ready methods section
│   └── feature_definitions.md    # Complete feature documentation
├── Dockerfile                    # Reproducible environment
├── pyproject.toml                # Modern Python packaging
└── requirements.txt              # Pinned dependencies
```

---

## For Researchers

### Using in Your Study

1. **Fork this repository** for your study
2. **Document any modifications** in a CHANGELOG
3. **Lock the version** used in your study (git tag)
4. **Include provenance.json** in supplementary materials

### Methods Section Template

See `docs/methods.md` for a complete, journal-ready methods section you can adapt for your manuscript.

### Validation

This pipeline has been validated against:
- ✅ Praat GUI (manual extraction comparison)
- ✅ librosa reference implementations
- ✅ Physiological range checks from literature

---

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{voicebiomarkers2025,
    title = {Voice Biomarkers Extraction Pipeline},
    author = {[Authors]},
    year = {2025},
    url = {https://github.com/ackalanka/multimorbidity-voice},
    version = {1.0.0}
}
```

---

## References

1. Maor E, et al. Voice signal characteristics are independently associated with coronary artery disease. Mayo Clinic Proceedings, 2018.
2. Murton OM, et al. Acoustic speech analysis of patients with heart failure. European Journal of Heart Failure, 2017.
3. Teixeira JP, Oliveira C, Lopes C. Vocal acoustic analysis – jitter, shimmer and HNR parameters. Procedia Technology, 2013.
4. Boersma P, Weenink D. Praat: doing phonetics by computer. 2023.

---

## License

MIT License - see [LICENSE](LICENSE) for details.