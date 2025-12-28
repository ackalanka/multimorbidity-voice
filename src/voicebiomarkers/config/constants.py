# voicebiomarkers/config/constants.py
"""
Scientific Constants for Voice Biomarker Extraction

All parameters are documented with their scientific justification
for peer-reviewed publication reproducibility.
"""

# ==============================================================================
# AUDIO PROCESSING CONSTANTS
# ==============================================================================

SAMPLE_RATE: int = 16000
"""Target sample rate in Hz.

16 kHz is standard for speech processing, providing adequate frequency
resolution up to 8 kHz (Nyquist) while being computationally efficient.
Reference: Rabiner & Juang (1993) Fundamentals of Speech Recognition.
"""

MIN_VOICE_DURATION: float = 0.5
"""Minimum voice duration in seconds after VAD trimming.

Files with less than 0.5s of voiced content are excluded as they
provide insufficient data for reliable feature extraction.
"""

MAX_VOICE_DURATION: float = 180.0
"""Maximum voice duration in seconds.

Files longer than 3 minutes are flagged for potential recording errors.
"""

# ==============================================================================
# NOISE REDUCTION PARAMETERS
# ==============================================================================

NOISE_REDUCTION_AGGRESSION: float = 1.5
"""Noise reduction strength (0.0 to 2.0).

Value of 1.5 provides good balance between noise removal and
preserving voice characteristics. Uses noisereduce library with
spectral gating (Sainburg et al., 2020).
"""

NOISE_PROFILE_DURATION: float = 0.5
"""Duration in seconds of initial audio used for noise profiling.

First 0.5 seconds are assumed to contain ambient noise without speech.
"""

# ==============================================================================
# MFCC EXTRACTION PARAMETERS
# ==============================================================================

N_MFCC: int = 72
"""Number of Mel-frequency cepstral coefficients.

72 provides extended frequency resolution for cardiovascular-related
voice changes. Standard speech recognition uses 13-40.
"""

N_MELS: int = 128
"""Number of Mel filter banks.

128 provides fine frequency resolution for medical applications.
"""

MFCC_FMIN: float = 50.0
"""Minimum MFCC frequency in Hz.

50 Hz captures low-frequency voice characteristics relevant to
cardiovascular assessment.
"""

MFCC_FMAX: float = 8000.0
"""Maximum MFCC frequency in Hz.

8 kHz is the Nyquist frequency for 16 kHz sample rate.
"""

FFT_SIZE: int = 2048
"""FFT window size in samples.

2048 samples at 16 kHz = 128ms, providing good frequency resolution
for pitch and formant analysis.
"""

HOP_LENGTH: int = 512
"""Hop length in samples.

512 samples at 16 kHz = 32ms, standard for speech analysis.
Provides 75% overlap with FFT_SIZE of 2048.
"""

PRE_EMPHASIS_COEF: float = 0.97
"""Pre-emphasis filter coefficient.

0.97 is standard for speech processing, boosting high frequencies
to compensate for spectral tilt in speech production.
"""

# ==============================================================================
# PRAAT/PARSELMOUTH PARAMETERS
# ==============================================================================

F0_MIN: float = 50.0
"""Minimum F0 in Hz for pitch detection.

50 Hz covers low male voices. Below this is typically non-voiced.
"""

F0_MAX: float = 500.0
"""Maximum F0 in Hz for pitch detection.

500 Hz covers high female/child voices with margin for accuracy.
"""

VOICE_THRESHOLD: float = 0.45
"""Voicing threshold for Praat pitch detection (0.0 to 1.0).

0.45 is a balanced threshold for mixed voice quality recordings.
"""

OCTAVE_COST: float = 0.01
"""Cost per octave for Praat pitch tracking.

Lower cost allows larger pitch jumps. 0.01 is Praat default.
"""

SILENCE_THRESHOLD: float = 0.03
"""Threshold for detecting silence (relative amplitude).

Used for pause detection and articulation rate calculation.
"""

MINIMUM_PITCH_PERIOD: float = 0.0001
"""Minimum pitch period in seconds for jitter/shimmer calculation.

0.0001s = 10,000 Hz maximum frequency.
"""

# ==============================================================================
# VOICE ACTIVITY DETECTION
# ==============================================================================

VAD_TOP_DB: float = 25.0
"""Top dB threshold for librosa.effects.trim().

Frames with power below this (relative to peak) are considered silence.
"""

VAD_FRAME_LENGTH: int = 2048
"""Frame length for VAD analysis in samples."""

VAD_HOP_LENGTH: int = 512
"""Hop length for VAD analysis in samples."""

# ==============================================================================
# QUALITY CONTROL THRESHOLDS
# ==============================================================================

QC_MIN_SNR_DB: float = 15.0
"""Minimum acceptable SNR in dB.

Files below 15 dB are flagged as low quality.
"""

QC_MAX_CLIPPING_PERCENT: float = 1.0
"""Maximum acceptable clipping percentage.

Files with >1% clipped samples are flagged.
"""

QC_MIN_F0_DETECTION_RATE: float = 0.50
"""Minimum F0 detection rate (0.0 to 1.0).

Files with <50% voiced frames are flagged as potentially non-speech.
"""

QC_MAX_SILENCE_PERCENT: float = 90.0
"""Maximum acceptable silence percentage.

Files with >90% silence are flagged as mostly silent.
"""
