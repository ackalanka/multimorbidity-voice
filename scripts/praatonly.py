from typing import Dict, Optional
import os
import re
import numpy as np
import pandas as pd
import librosa
import logging
import noisereduce as nr
import parselmouth
from datetime import datetime
from multiprocessing import Pool, cpu_count
from scipy.stats import kurtosis, skew
from parselmouth.praat import call
from functools import wraps

# Configuration
INPUT_DIR = r"C:\Users\Akalanka\Desktop\multimorbidity_voice\data\raw"
OUTPUT_CSV = "voice_features.csv"
SAMPLE_RATE = 16000

# Processing Parameters
USE_PARSELMOUTH = True
MIN_VOICE_DURATION = 0.5
MIN_PARSELMOUTH_DURATION = 0.3
NOISE_REDUCTION_AGGRESSION = 1.5

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("feature_extraction.log"),
        logging.StreamHandler()
    ]
)

def calculate_formant_dispersion(formant):
    """Calculate formant dispersion using first three formants"""
    try:
        f1 = call(formant, "Get mean", 1, 0, 0, "Hertz")
        f2 = call(formant, "Get mean", 2, 0, 0, "Hertz")
        f3 = call(formant, "Get mean", 3, 0, 0, "Hertz")
        return ((f2 - f1) + (f3 - f2)) / 2
    except Exception as e:
        logging.debug(f"Formant dispersion error: {str(e)}")
        return None

def calculate_formant_centralization_ratio(formant):
    """Calculate Formant Centralization Ratio (FCR) - vowel space area indicator"""
    try:
        f1 = call(formant, "Get mean", 1, 0, 0, "Hertz")
        f2 = call(formant, "Get mean", 2, 0, 0, "Hertz")
        # FCR = (F2u + F2a + F1i + F1u) / (F2i + F1a) - simplified version using mean values
        fcr = (f2 + f1) / np.sqrt(f2 * f1) if f2 > 0 and f1 > 0 else None
        return fcr
    except Exception as e:
        logging.debug(f"FCR error: {str(e)}")
        return None

def calculate_voice_tremor(pitch):
    """Calculate voice tremor using pitch frequency modulation"""
    try:
        # Get pitch values
        pitch_values = []
        n_frames = call(pitch, "Get number of frames")
        for i in range(1, min(n_frames + 1, 500)):  # Limit frames for performance
            try:
                pitch_val = call(pitch, "Get value at time", i * 0.01, "Hertz", "Linear")
                if pitch_val != 0:  # Exclude unvoiced frames
                    pitch_values.append(pitch_val)
            except:
                continue
        
        if len(pitch_values) < 10:
            return None
            
        pitch_array = np.array(pitch_values)
        # Calculate tremor as coefficient of variation of pitch
        tremor = np.std(pitch_array) / np.mean(pitch_array) if np.mean(pitch_array) > 0 else None
        return tremor
    except Exception as e:
        logging.debug(f"Voice tremor error: {str(e)}")
        return None

def calculate_maximum_phonation_time(sound):
    """Estimate maximum phonation time from continuous voiced segments"""
    try:
        # Get intensity contour
        intensity = call(sound, "To Intensity", 100, 0.0, "yes")
        
        # Get voiced segments using intensity threshold
        duration = call(sound, "Get total duration")
        n_frames = int(duration * 100)  # 100 Hz frame rate
        
        voiced_segments = []
        current_segment = 0
        threshold = 50  # dB threshold for voice activity
        
        for i in range(n_frames):
            time = i * 0.01
            try:
                intensity_val = call(intensity, "Get value at time", time, "Cubic")
                if intensity_val > threshold:
                    current_segment += 0.01
                else:
                    if current_segment > 0:
                        voiced_segments.append(current_segment)
                        current_segment = 0
            except:
                continue
        
        if current_segment > 0:
            voiced_segments.append(current_segment)
            
        return max(voiced_segments) if voiced_segments else None
        
    except Exception as e:
        logging.debug(f"MPT error: {str(e)}")
        return None

def calculate_pause_metrics(sound):
    """Calculate pause-related metrics for cardiovascular assessment"""
    try:
        # Get intensity to detect pauses
        intensity = call(sound, "To Intensity", 100, 0.0, "yes")
        duration = call(sound, "Get total duration")
        
        # Detect pauses (low intensity regions)
        n_frames = int(duration * 100)
        pause_threshold = 45  # dB
        
        pauses = []
        current_pause = 0
        total_speech_time = 0
        
        for i in range(n_frames):
            time = i * 0.01
            try:
                intensity_val = call(intensity, "Get value at time", time, "Cubic")
                if intensity_val < pause_threshold:
                    current_pause += 0.01
                else:
                    total_speech_time += 0.01
                    if current_pause > 0.1:  # Minimum pause duration
                        pauses.append(current_pause)
                        current_pause = 0
            except:
                continue
        
        if current_pause > 0.1:
            pauses.append(current_pause)
        
        # Calculate metrics
        pause_rate = len(pauses) / duration if duration > 0 else 0
        mean_pause_duration = np.mean(pauses) if pauses else 0
        pause_time_ratio = sum(pauses) / duration if duration > 0 else 0
        
        return {
            'pause_rate': pause_rate,
            'mean_pause_duration': mean_pause_duration,
            'pause_time_ratio': pause_time_ratio
        }
        
    except Exception as e:
        logging.debug(f"Pause metrics error: {str(e)}")
        return {'pause_rate': None, 'mean_pause_duration': None, 'pause_time_ratio': None}

def calculate_speech_rate_metrics(sound):
    """Calculate speech rate and rhythm metrics"""
    try:
        # Get intensity for syllable detection
        intensity = call(sound, "To Intensity", 100, 0.0, "yes")
        duration = call(sound, "Get total duration")
        
        # Simple syllable detection using intensity peaks
        n_frames = int(duration * 100)
        intensity_values = []
        
        for i in range(n_frames):
            time = i * 0.01
            try:
                intensity_val = call(intensity, "Get value at time", time, "Cubic")
                intensity_values.append(intensity_val)
            except:
                intensity_values.append(0)
        
        # Find peaks (simplified syllable counting)
        intensity_array = np.array(intensity_values)
        mean_intensity = np.mean(intensity_array)
        
        # Count syllables as intensity peaks above mean
        syllable_count = 0
        in_syllable = False
        
        for val in intensity_array:
            if val > mean_intensity + 5 and not in_syllable:
                syllable_count += 1
                in_syllable = True
            elif val < mean_intensity:
                in_syllable = False
        
        # Calculate rates
        syllable_rate = syllable_count / duration if duration > 0 else 0
        
        # Rhythm variability (coefficient of variation of inter-syllable intervals)
        rhythm_variability = np.std(intensity_array) / np.mean(intensity_array) if np.mean(intensity_array) > 0 else None
        
        return {
            'syllable_rate': syllable_rate,
            'rhythm_variability': rhythm_variability
        }
        
    except Exception as e:
        logging.debug(f"Speech rate metrics error: {str(e)}")
        return {'syllable_rate': None, 'rhythm_variability': None}

def calculate_vocal_stability_index(pitch, sound):
    """Calculate vocal stability index combining multiple stability measures"""
    try:
        # Get pitch stability
        pitch_values = []
        n_frames = call(pitch, "Get number of frames")
        for i in range(1, min(n_frames + 1, 200)):
            try:
                pitch_val = call(pitch, "Get value at time", i * 0.01, "Hertz", "Linear")
                if pitch_val != 0:
                    pitch_values.append(pitch_val)
            except:
                continue
        
        if len(pitch_values) < 10:
            return None
        
        # Calculate stability metrics
        pitch_array = np.array(pitch_values)
        pitch_stability = 1 / (1 + np.std(pitch_array) / np.mean(pitch_array))
        
        # Get amplitude stability
        intensity = call(sound, "To Intensity", 100, 0.0, "yes")
        duration = call(sound, "Get total duration")
        
        intensity_values = []
        for i in range(0, int(duration * 50)):  # 50 Hz sampling
            time = i * 0.02
            try:
                intensity_val = call(intensity, "Get value at time", time, "Cubic")
                intensity_values.append(intensity_val)
            except:
                continue
        
        if len(intensity_values) < 10:
            return pitch_stability
        
        intensity_array = np.array(intensity_values)
        amplitude_stability = 1 / (1 + np.std(intensity_array) / np.mean(intensity_array))
        
        # Combined stability index
        stability_index = (pitch_stability + amplitude_stability) / 2
        
        return stability_index
        
    except Exception as e:
        logging.debug(f"Vocal stability index error: {str(e)}")
        return None

def calculate_breathiness_measures(sound, harmonicity):
    """Calculate breathiness-related measures"""
    try:
        # Harmonics-to-noise ratio already available from harmonicity
        hnr = call(harmonicity, "Get mean", 0, 0)
        
        # Calculate spectral noise measures
        spectrum = call(sound, "To Spectrum", "yes")
        
        # Get spectral moments for breathiness assessment
        spectral_moments = []
        for i in range(1, 50):  # First 50 frequency bins
            try:
                power = call(spectrum, "Get real value in bin", i)
                spectral_moments.append(power)
            except:
                continue
        
        if len(spectral_moments) < 10:
            return {'breathiness_index': None, 'spectral_noise': None}
        
        # Breathiness index (inverse of HNR with spectral considerations)
        breathiness_index = 1 / (1 + hnr) if hnr > 0 else 1
        
        # Spectral noise (high-frequency energy ratio)
        spectral_array = np.array(spectral_moments)
        spectral_noise = np.sum(spectral_array[25:]) / np.sum(spectral_array) if np.sum(spectral_array) > 0 else None
        
        return {
            'breathiness_index': breathiness_index,
            'spectral_noise': spectral_noise
        }
        
    except Exception as e:
        logging.debug(f"Breathiness measures error: {str(e)}")
        return {'breathiness_index': None, 'spectral_noise': None}

import scipy.stats   
def calculate_spectral_tilt(sound):
    """Compute spectral tilt by fitting a regression line to the log power spectrum."""
    try:
        spectrum = call(sound, "To Spectrum", "yes")  # Convert to spectrum
        freqs = np.array([call(spectrum, "Get frequency from bin number", i) for i in range(1, 100)])  # First 100 bins
        powers = np.array([call(spectrum, "Get real value in bin", i) for i in range(1, 100)])

        # Convert power to dB (log scale)
        log_powers = np.log10(np.maximum(powers, 1e-10))  # Prevent log(0) error

        # Fit linear regression (spectral tilt = slope of this line)
        slope, _, _, _, _ = scipy.stats.linregress(freqs, log_powers)
        return slope  # Spectral tilt (dB/Hz)
    
    except Exception as e:
        logging.debug(f"Spectral tilt error: {str(e)}")
        return None   

# Enhanced PARSELMOUTH_FEATURES dictionary with cardiovascular-relevant features
PARSELMOUTH_FEATURES = {
    # Original jitter features
    'jitter_local': lambda pp: call(pp, "Get jitter (local)", 0, 0, 0.001, 0.02, 1.3),
    'jitter_ppq5': lambda pp: call(pp, "Get jitter (ppq5)", 0, 0, 0.001, 0.02, 1.3),

    # Original shimmer features
    'shimmer_local': lambda s, pp: call([s, pp], "Get shimmer (local)", 0, 0, 0.001, 0.02, 1.3, 1.6),
    'shimmer_apq11': lambda s, pp: call([s, pp], "Get shimmer (apq11)", 0, 0, 0.001, 0.02, 1.3, 1.6),

    # Original pitch features
    'f0_mean': lambda p: call(p, "Get mean", 0, 0, "Hertz"),
    'f0_std': lambda p: call(p, "Get standard deviation", 0, 0, "Hertz"),
    'f0_entropy': lambda p: call(p, "Get quantile", 0, 0, 0.5, "Hertz"),

    # Original harmonicity
    'hnr_praat': lambda h: call(h, "Get mean", 0, 0),

    # Original formant features
    'formant_dispersion': lambda f: calculate_formant_dispersion(f),

    # Original spectral features
    'spectral_tilt': lambda s: calculate_spectral_tilt(s),

    # New formant-based features
    'formant_centralization_ratio': lambda f: calculate_formant_centralization_ratio(f),
    'f1_mean': lambda f: call(f, "Get mean", 1, 0, 0, "Hertz"),
    'f2_mean': lambda f: call(f, "Get mean", 2, 0, 0, "Hertz"),
    'f1_std': lambda f: call(f, "Get standard deviation", 1, 0, 0, "Hertz"),
    'f2_std': lambda f: call(f, "Get standard deviation", 2, 0, 0, "Hertz"),

    # Voice stability and tremor
    'voice_tremor': lambda p: calculate_voice_tremor(p),
    'vocal_stability_index': lambda p, s: calculate_vocal_stability_index(p, s),

    # Phonation time and breathing
    'maximum_phonation_time': lambda s: calculate_maximum_phonation_time(s),

    # Pitch dynamics
    'f0_range': lambda p: call(p, "Get maximum", 0, 0, "Hertz", "Parabolic") - call(p, "Get minimum", 0, 0, "Hertz", "Parabolic"),
    'f0_iqr': lambda p: call(p, "Get quantile", 0, 0, 0.75, "Hertz") - call(p, "Get quantile", 0, 0, 0.25, "Hertz"),

    # Intensity dynamics (FIXED)
    'intensity_mean': lambda s: call(call(s, "To Intensity", 100, 0.0, "yes"), "Get mean", 0, 0),
    'intensity_std': lambda s: call(call(s, "To Intensity", 100, 0.0, "yes"), "Get standard deviation", 0, 0),
    'intensity_range': lambda s: call(call(s, "To Intensity", 100, 0.0, "yes"), "Get maximum", 0, 0, "Parabolic") - call(call(s, "To Intensity", 100, 0.0, "yes"), "Get minimum", 0, 0, "Parabolic"),

    # Harmonicity variations (FIXED)
    'hnr_std': lambda h: call(h, "Get standard deviation", 0, 0),
    'hnr_min': lambda h: call(h, "Get minimum", 0, 0, "Parabolic"),
    'hnr_max': lambda h: call(h, "Get maximum", 0, 0, "Parabolic"),
}



# Features that require multiple parameters - handled separately
COMPOSITE_FEATURES = {
    'pause_rate': lambda s: calculate_pause_metrics(s)['pause_rate'],
    'mean_pause_duration': lambda s: calculate_pause_metrics(s)['mean_pause_duration'],
    'pause_time_ratio': lambda s: calculate_pause_metrics(s)['pause_time_ratio'],
    'syllable_rate': lambda s: calculate_speech_rate_metrics(s)['syllable_rate'],
    'rhythm_variability': lambda s: calculate_speech_rate_metrics(s)['rhythm_variability'],
    'breathiness_index': lambda s, h: calculate_breathiness_measures(s, h)['breathiness_index'],
    'spectral_noise': lambda s, h: calculate_breathiness_measures(s, h)['spectral_noise'],
}

def parselmouth_safe(func):
    """Decorator for robust Praat feature extraction"""
    @wraps(func)
    def wrapper(y: np.ndarray, sr: int) -> Dict:
        all_features = {**PARSELMOUTH_FEATURES, **COMPOSITE_FEATURES}
        features = {'parselmouth_status': 'success', 'praat_errors': {}}
        if y.size < sr * MIN_PARSELMOUTH_DURATION:
            return {k: None for k in all_features} | features
            
        try:
            # Create Praat objects
            sound = parselmouth.Sound(y, sampling_frequency=sr)
            pitch = call(sound, "To Pitch", 0.0, 75, 600)
            point_process = call(sound, "To PointProcess (periodic, cc)", 75, 600)
            harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 500, 0.1, 1.0)
            formant = call(sound, "To Formant (burg)", 0.0025, 5, 5000, 0.025, 50)

            # Extract standard features
            for name, extractor in PARSELMOUTH_FEATURES.items():
                try:
                    if name in ['jitter_local', 'jitter_ppq5']:
                        features[name] = extractor(point_process)
                    elif name in ['shimmer_local', 'shimmer_apq11']:
                        features[name] = extractor(sound, point_process)
                    elif name in ['f0_mean', 'f0_std', 'f0_entropy', 'f0_range', 'f0_iqr', 'voice_tremor']:
                        features[name] = extractor(pitch)
                    elif name in ['hnr_praat', 'hnr_std', 'hnr_min', 'hnr_max']:
                        features[name] = extractor(harmonicity)
                    elif name in ['formant_dispersion', 'formant_centralization_ratio', 'f1_mean', 'f2_mean', 'f1_std', 'f2_std']:
                        features[name] = extractor(formant)
                    elif name in ['spectral_tilt', 'maximum_phonation_time', 'intensity_mean', 'intensity_std', 'intensity_range']:
                        features[name] = extractor(sound)
                    elif name == 'vocal_stability_index':
                        features[name] = extractor(pitch, sound)
                    else:
                        features[name] = None
                except Exception as e:
                    features[name] = None
                    error_msg = str(e).split("\n")[0][:100]
                    features['praat_errors'][name] = error_msg.strip()
                    features['parselmouth_status'] = 'partial_success'

            # Extract composite features
            for name, extractor in COMPOSITE_FEATURES.items():
                try:
                    if name in ['pause_rate', 'mean_pause_duration', 'pause_time_ratio', 'syllable_rate', 'rhythm_variability']:
                        features[name] = extractor(sound)
                    elif name in ['breathiness_index', 'spectral_noise']:
                        features[name] = extractor(sound, harmonicity)
                    else:
                        features[name] = None
                except Exception as e:
                    features[name] = None
                    error_msg = str(e).split("\n")[0][:100]
                    features['praat_errors'][name] = error_msg.strip()
                    features['parselmouth_status'] = 'partial_success'

            if all(v is None for v in features.values() if isinstance(v, (int, float))):
                features['parselmouth_status'] = 'failed'
                
        except Exception as e:
            features.update({k: None for k in all_features})
            features['parselmouth_status'] = f'failed: {str(e)[:100]}'
            logging.error(f"Praat processing failed: {str(e)}")
            
        return features
    return wrapper

@parselmouth_safe
def extract_parselmouth_features(y: np.ndarray, sr: int) -> Dict:
    """Praat feature extraction handler"""
    return {}

# Edited the following function to recognize new file name convention
def parse_filename_metadata(filename: str) -> Dict: 
    """Extract metadata from filenames in the format: ID-Gender-Age-Date-Time.wav"""
    metadata = {
        "patient_id": "unknown",
        "gender": None,
        "age": None,
        "recording_date": None,
        "recording_time": None,
        "filename_error": None
    }
    
    try:
        # Expected format: 2303-F-30-29.11.2023-20-08.wav
        pattern = r"^(?P<patient_id>\d+)-(?P<gender>[FM])-(?P<age>\d+)-(?P<date>\d{2}\.\d{2}\.\d{4})-(?P<hour>\d{2})-(?P<minute>\d{2})\.wav$"
        
        match = re.match(pattern, filename)
        
        if match:
            groups = match.groupdict()
            
            # Extract basic info
            metadata["patient_id"] = groups["patient_id"]
            metadata["gender"] = groups["gender"]
            
            # Validate and parse age
            try:
                metadata["age"] = int(groups["age"])
                if not (0 <= metadata["age"] <= 120):
                    raise ValueError("Age out of reasonable range")
            except ValueError as e:
                metadata["filename_error"] = f"Invalid age: {str(e)}"
                logging.warning(f"Age parsing error in {filename}: {str(e)}")
            
            # Parse date (DD.MM.YYYY)
            try:
                metadata["recording_date"] = datetime.strptime(
                    groups["date"], "%d.%m.%Y"
                ).strftime("%Y-%m-%d")
            except ValueError as e:
                metadata["filename_error"] = f"Invalid date: {str(e)}"
                logging.warning(f"Date parsing error in {filename}: {str(e)}")
            
            # Parse time (HH-MM)
            try:
                time_str = f"{groups['hour']}:{groups['minute']}"
                metadata["recording_time"] = datetime.strptime(
                    time_str, "%H:%M"
                ).strftime("%H:%M:%S")
            except ValueError as e:
                metadata["filename_error"] = f"Invalid time: {str(e)}"
                logging.warning(f"Time parsing error in {filename}: {str(e)}")
            
            # Additional validations
            if metadata["gender"] not in ["F", "M"]:
                metadata["filename_error"] = "Invalid gender code"
                logging.warning(f"Invalid gender in {filename}")
                
        else:
            metadata["filename_error"] = "Filename pattern mismatch"
            logging.warning(f"Filename pattern mismatch: {filename}")
            
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)[:100]}"
        metadata["filename_error"] = error_msg
        logging.error(f"Metadata parsing failed for {filename}: {error_msg}", exc_info=True)
    
    return metadata

def process_audio_file(file_path: str) -> Optional[Dict]:
    """Complete audio processing pipeline"""
    try:
        filename = os.path.basename(file_path)
        features = {"filename": filename}
        
        # === Metadata Extraction ===
        features.update(parse_filename_metadata(filename))
        
        # === Audio Preprocessing ===
        y, sr = librosa.load(file_path, sr=None, mono=True)
        features["original_duration"] = len(y)/sr
        
        # Resample if needed
        if sr != SAMPLE_RATE:
            y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
        
        # Noise reduction
        noise_profile = y[:int(SAMPLE_RATE * 0.5)] if len(y) > SAMPLE_RATE//2 else y
        y_clean = nr.reduce_noise(
            y=y, y_noise=noise_profile, sr=SAMPLE_RATE,
            stationary=True, prop_decrease=NOISE_REDUCTION_AGGRESSION,
            n_fft=2048, hop_length=512
        )
        
        # Voice activity detection
        y_trimmed, (start, end) = librosa.effects.trim(
            y_clean, top_db=25, frame_length=2048, hop_length=512
        )
        features["processed_duration"] = len(y_trimmed)/SAMPLE_RATE
        
        if features["processed_duration"] < MIN_VOICE_DURATION:
            logging.error(f"Insufficient voice content: {filename}")
            return None

        # Quality metrics
        features.update({
            "snr_db": 10 * np.log10(np.var(y_trimmed)/np.var(y[:1000])) if len(y) > 1000 else None,
            "trim_start": start,
            "trim_end": end
        })

        # === Parselmouth Features ===
        if USE_PARSELMOUTH:
            praat_features = extract_parselmouth_features(y_trimmed, SAMPLE_RATE)
            features.update(praat_features)

        return features

    except Exception as e:
        logging.error(f"Processing failed {filename}: {str(e)}", exc_info=True)
        return None

def main():
    """Main processing workflow"""
    try:
        files = [os.path.join(INPUT_DIR, f) 
                for f in os.listdir(INPUT_DIR) 
                if f.lower().endswith(".wav")]
        
        if not files:
            logging.error("No WAV files found")
            return

        logging.info(f"Processing {len(files)} files")
        
        with Pool(cpu_count()) as pool:
            results = []
            for i, result in enumerate(pool.imap(process_audio_file, files)):
                if result:
                    results.append(result)
                if (i+1) % 10 == 0:
                    logging.info(f"Processed {i+1}/{len(files)} files")
                    logging.info(f"Success rate: {len(results)/(i+1):.1%}")

        # Save results with column validation
        if results:
            df = pd.DataFrame(results)
            
            # Dynamically get available columns
            available_columns = set(df.columns)
            
            # Define desired column groups with new features
            column_groups = [
                # Metadata columns
                ['patient_id', 'gender', 'age', 'recording_date', 'recording_time', 'filename',
                 'original_duration', 'processed_duration'],
                
                # Original Parselmouth features
                ['jitter_local', 'jitter_ppq5', 'shimmer_local', 'shimmer_apq11', 
                 'f0_mean', 'f0_std', 'f0_entropy', 'hnr_praat', 'formant_dispersion', 'spectral_tilt'],
                
                # New cardiovascular-relevant features
                ['formant_centralization_ratio', 'f1_mean', 'f2_mean', 'f1_std', 'f2_std',
                 'voice_tremor', 'vocal_stability_index', 'maximum_phonation_time',
                 'f0_range', 'f0_iqr', 'intensity_mean', 'intensity_std', 'intensity_range',
                 'hnr_std', 'hnr_min', 'hnr_max', 'pause_rate', 'mean_pause_duration', 
                 'pause_time_ratio', 'syllable_rate', 'rhythm_variability', 
                 'breathiness_index', 'spectral_noise'],
                
                # System columns
                ['snr_db', 'trim_start', 'trim_end', 'parselmouth_status', 'praat_errors']
            ]

            # Build ordered columns list with existing columns only
            ordered_columns = []
            for group in column_groups:
                filtered = [col for col in group if col in available_columns]
                ordered_columns += filtered

            # Create final DataFrame
            final_df = df[ordered_columns]
            final_df.to_csv(OUTPUT_CSV, index=False)
            logging.info(f"Features saved to {OUTPUT_CSV}")

            # Generate quality report
            quality_report = {
                "total_files": len(files),
                "processed_files": len(final_df),
                "success_rate": len(final_df)/len(files),
                "mean_duration": final_df["processed_duration"].mean(),
                "valid_metadata": (~final_df["patient_id"].str.contains("unknown")).mean(),
                "praat_success_rate": (final_df['parselmouth_status'] == 'success').mean()
            }
            logging.info(f"Quality Report:\n{pd.Series(quality_report).to_string()}")
        else:
            logging.warning("No valid features extracted - creating empty output")
            pd.DataFrame().to_csv(OUTPUT_CSV)

    except Exception as e:
        logging.error(f"Fatal error: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
