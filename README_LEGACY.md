- This project is designed to work with **Python 3.12** only.
- Running it with any other version of Python may cause compatibility issues with dependencies.

### INSTALLATION

Clone the repository
```bash 
git clone https://github.com/ackalanka/multimorbidity-voice.git
```

Create a venv using python 3.12
```bash
python3.12 -m venv venv
```

Activate venv
```bash
.\venv\Scripts\activate
```
  
Install requirements
```bash
pip install -r requirements.txt
```

Navigate to scripts\ and run mfcc81.py
```bash
cd .\venv\Scripts\activate
python mfcc81.py
```

### Breakdown of the script mfcc.py

The script uses Mel Frequency Cepstral Coefficients (MFCCs) to extract 81 voice features. MFCCs capture the most important features of human speech that are useful for analysis. Instead of analyzing raw audio, MFCCs convert voice signals into numerical values that represent speech patterns. They are widely used in speech recognition (Siri, Google Assistant) and medical voice analysis (Parkinson’s, stress detection).

##### **1. Audio Processing Metrics**
Preprocessing of the raw audio by mfcc81.py
- **Sample Rate**: `16000 Hz` (resamples audio to 16 kHz for consistency).
- **Silence Trimming**: `top_db=20` (removes leading/trailing silence below -20 dB).
- **Noise Reduction**: Uses the first `0.5 seconds` of audio as a noise profile to clean recordings.

##### **2. MFCC Feature Extraction Metrics**
mfcc81.py calculates Mel-Frequency Cepstral Coefficients as follows:

- **Number of Coefficients**: `13` (standard for speech/voice analysis).
- **Frame Size**: `25 ms` (window length for spectral analysis).
- **Hop Length**: `10 ms` (step size between frames).
- **Mel Filters**: `40` (frequency bands spaced on the Mel scale, approximating human hearing).
- **Statistics per Coefficient**: For each of the 13 coefficients, the script calculates 6 statistical measures:
  1. **Mean** (average value)
  2. **Variance** (spread of values)
  3. **Median** (middle value)
  4. **Max** (highest value)
  5. **Min** (lowest value)
  6. **Range** (`Max - Min`)

##### **3. Filename Metadata Parsing**
mfcc81.py extracts structured metadata from filenames like `EU150-001-14112023-17-56.wav` into: 
- **Cohort**: `EU150-001` (first two segments).
- **Participant ID**: `001` (second segment).
- **Date**: `14 November 2023` (parsed from `14112023`).
- **Time**: `17:56` (parsed from `17-56`).

##### **4. Output **
The final CSV (`mfcc_voice_features.csv`) contains:
- **81 Features**:  
  `13 MFCC coefficients × 6 statistics = 78 features` + `3 padded zeros` (to reach 81, as in the original study).
- **Metadata Columns**:  
  `cohort`, `participant_id`, `date`, `time`.

##### **Example Output**
| cohort    | participant_id | date       | time  | feature_001 | ... | feature_081 |
| --------- | -------------- | ---------- | ----- | ----------- | --- | ----------- |
| EU150-001 | 001            | 2023-11-14 | 17:56 | 0.45        | ... | -1.2        |

##### **Why These Metrics?**
- **MFCCs**: Standard for voice/speech analysis (captures timbre/texture).
- **Statistics**: Quantify how coefficients change over time (e.g., high variance = unstable voice).
- **Longitudinal Metadata**: Enables tracking of voice changes before/after diagnosis.

##### **Key Parameters to Adjust (If Necessary)**
| Parameter               | Current Value | What It Affects                       |
| ----------------------- | ------------- | ------------------------------------- |
| `n_mfcc`                | 13            | Number of MFCC coefficients extracted |
| `n_fft` (frame size)    | 25 ms         | Frequency resolution                  |
| `hop_length`            | 10 ms         | Temporal resolution                   |
| `n_mels`                | 40            | Smoothing of frequency bands          |
| `top_db` (silence trim) | 20 dB         | Aggressiveness of silence removal     |
