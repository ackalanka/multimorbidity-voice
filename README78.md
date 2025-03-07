- This project is designed to work with **Python 3.12** only.
- Running it with any other version of Python may cause compatibility issues with dependencies.

# Voice Biomarker Extraction Pipeline

A Python script for extracting Mel-frequency cepstral coefficients (MFCCs) from audio files, designed for voice analysis research.

## Table of Contents
- [Goals](#goals)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Output Details](#output-details)
- [Technical Specifications](#technical-specifications)
- [Calculated Statistics](#calculated-statistics)
- [Example Output](#example-output)
- [Logging](#logging)
- [License](#license)

## Goals
1. Extract **78 MFCC-based features** from WAV files
2. Preserve participant metadata from filenames
3. Generate self-documented outputs
4. Enable batch processing with parallel execution
5. Maintain reproducibility through hardcoded parameters

## Features
- Automated MFCC extraction using Librosa
- Metadata parsing from standardized filenames
- Multiprocessing for efficient computation
- CSV output with timestamps and participant IDs
- Built-in data dictionary generation
- Error handling with detailed logging

##  Installation
```bash
# Install required packages
pip install librosa pandas numpy
```

## Usage
### 1. Directory Structure
```
your_project/
├── scripts/
│   └── mfcc_extractor.py      # Main script
└── data/
    └── raw/                   # Input WAV files
        ├── EU150-001-14112023-1730.wav
        └── ...other_files.wav
```

### 2. Filename Convention
```
[Cohort]-[ParticipantID]-[DDMMYYYY]-[HHMM].wav
Example: EU150-001-14112023-1730.wav
```

### 3. Execution
```bash
python mfcc_extractor.py
```

## Output Details
| File                        | Description                                  |
|-----------------------------|----------------------------------------------|
| `mfcc_features.csv`         | Main output with features + metadata         |
| `mfcc_data_dictionary.csv`  | Documentation of all 78 features             |
| `mfcc_extraction.log`       | Detailed processing log with timestamps      |

## Technical Specifications
| Parameter           | Value        | Description                          |
|---------------------|--------------|--------------------------------------|
| Sample Rate         | 16000 Hz     | Audio sampling rate                  |
| FFT Window Size     | 2048 samples | Fast Fourier Transform window length |
| Hop Length          | 512 samples  | Frame spacing between FFT windows    |
| Mel Bands           | 40           | Number of frequency bands            |
| MFCC Coefficients   | 13           | Number of cepstral coefficients      |
| Audio Normalization | Enabled      | Automatic gain adjustment            |

## Calculated Statistics
For each MFCC coefficient (1-13), we calculate:
1. **Mean**: Average value
2. **Variance**: Spread of values
3. **Median**: Middle value
4. **Max**: Maximum value
5. **Min**: Minimum value
6. **Range**: Difference between max and min

## Example Output
### `mfcc_features.csv`
| cohort | participant_id | date       | time  | mfcc_01_mean | mfcc_01_var | mfcc_01_max | ... |
|--------|----------------|------------|-------|--------------|-------------|-------------|-----|
| EU150  | EU150-001      | 2023-11-14 | 17:30 | -12.45       | 4.78        | 8.92        | ... |
| EU150  | EU150-002      | 2023-11-14 | 18:15 | -10.91       | 3.45        | 7.81        | ... |

### `mfcc_data_dictionary.csv`
| Feature          | Description                     |
|------------------|---------------------------------|
| mfcc_01_mean     | Mean of MFCC coefficient 1      |
| mfcc_01_var      | Variance of MFCC coefficient 1  |
| mfcc_01_median   | Median of MFCC coefficient 1    |
| ...              | ...                             |

## Logging
The script generates `mfcc_extraction.log` with:
- Start/end timestamps
- Number of processed files
- Success/failure status per file
- Error messages with stack traces
- Performance metrics (files processed per minute)

## License
MIT License - See [LICENSE](LICENSE) file for details