# everything works!
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
N_MFCC = 13

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

PARSELMOUTH_FEATURES = {
    # Jitter features
    'jitter_local': lambda pp: call(pp, "Get jitter (local)", 0, 0, 0.001, 0.02, 1.3),
    'jitter_ppq5': lambda pp: call(pp, "Get jitter (ppq5)", 0, 0, 0.001, 0.02, 1.3),
    
    # Shimmer features (corrected argument structure)
    'shimmer_local': lambda s, pp: call([s, pp], "Get shimmer (local)", 0, 0, 0.001, 0.02, 1.3, 1.6),
    'shimmer_apq11': lambda s, pp: call([s, pp], "Get shimmer (apq11)", 0, 0, 0.001, 0.02, 1.3, 1.6),
    
    # Pitch features
    'f0_mean': lambda p: call(p, "Get mean", 0, 0, "Hertz"),
    'f0_std': lambda p: call(p, "Get standard deviation", 0, 0, "Hertz"),
    'f0_entropy': lambda p: call(p, "Get quantile", 0, 0, 0.5, "Hertz"),
    
    # Harmonicity
    'hnr_praat': lambda h: call(h, "Get mean", 0, 0),
    
    # Formant features
    'formant_dispersion': lambda f: calculate_formant_dispersion(f),
    
    # Spectral features (corrected spectrum handling)
    'spectral_tilt': lambda s: calculate_spectral_tilt(s)
}

def parselmouth_safe(func):
    """Decorator for robust Praat feature extraction"""
    @wraps(func)
    def wrapper(y: np.ndarray, sr: int) -> Dict:
        features = {'parselmouth_status': 'success', 'praat_errors': {}}
        if y.size < sr * MIN_PARSELMOUTH_DURATION:
            return {k: None for k in PARSELMOUTH_FEATURES} | features
            
        try:
            # Create Praat objects
            sound = parselmouth.Sound(y, sampling_frequency=sr)
            pitch = call(sound, "To Pitch", 0.0, 75, 600)
            point_process = call(sound, "To PointProcess (periodic, cc)", 75, 600)
            harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 500, 0.1, 1.0)
            formant = call(sound, "To Formant (burg)", 0.0025, 5, 5000, 0.025, 50)

            # Extract features with proper object passing
            for name, extractor in PARSELMOUTH_FEATURES.items():
                try:
                    if name in ['jitter_local', 'jitter_ppq5']:
                        features[name] = extractor(point_process)
                    elif name in ['shimmer_local', 'shimmer_apq11']:
                        features[name] = extractor(sound, point_process)
                    elif name in ['f0_mean', 'f0_std', 'f0_entropy']:
                        features[name] = extractor(pitch)
                    elif name == 'hnr_praat':
                        features[name] = extractor(harmonicity)
                    elif name == 'formant_dispersion':
                        features[name] = extractor(formant)
                    elif name == 'spectral_tilt':
                        features[name] = extractor(sound)
                    else:
                        features[name] = None
                except Exception as e:
                    features[name] = None
                    error_msg = str(e).split("\n")[0][:100]
                    features['praat_errors'][name] = error_msg.strip()
                    features['parselmouth_status'] = 'partial_success'

            if all(v is None for v in features.values() if isinstance(v, float)):
                features['parselmouth_status'] = 'failed'
                
        except Exception as e:
            features.update({k: None for k in PARSELMOUTH_FEATURES})
            features['parselmouth_status'] = f'failed: {str(e)[:100]}'
            logging.error(f"Praat processing failed: {str(e)}")
            
        return features
    return wrapper

@parselmouth_safe
def extract_parselmouth_features(y: np.ndarray, sr: int) -> Dict:
    """Praat feature extraction handler"""
    return {}

def parse_filename_metadata(filename: str) -> Dict:
    """Extract metadata from filename"""
    metadata = {
        "patient_id": "unknown",
        "recording_date": None,
        "recording_time": None
    }
    
    try:
        pattern = r"^(.*?)-(\d{8})-(\d{1,2}-\d{2})\.wav$"
        match = re.match(pattern, filename)
        
        if match:
            metadata["patient_id"] = match.group(1)
            date_str = match.group(2)
            time_str = match.group(3).replace("-", ":")
            
            metadata["recording_date"] = datetime.strptime(
                date_str, "%d%m%Y"
            ).strftime("%Y-%m-%d")
            
            metadata["recording_time"] = datetime.strptime(
                time_str, "%H:%M"
            ).strftime("%H:%M:%S")
            
    except Exception as e:
        logging.warning(f"Metadata parsing failed for {filename}: {str(e)}")
    
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

        # === MFCC Feature Extraction ===
        y_preemph = librosa.effects.preemphasis(y_trimmed, coef=0.97)
        y_normalized = librosa.util.normalize(y_preemph)

        mfcc = librosa.feature.mfcc(
            y=y_normalized, sr=SAMPLE_RATE,
            n_mfcc=N_MFCC, n_mels=128,
            fmin=50, fmax=8000,
            n_fft=2048, hop_length=512
        )

        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)

        # MFCC Statistics
        for i in range(N_MFCC):
            coeff = mfcc[i]
            features.update({
                f"mfcc_{i+1:02}_mean": np.mean(coeff),
                f"mfcc_{i+1:02}_std": np.std(coeff),
                f"mfcc_{i+1:02}_kurtosis": kurtosis(coeff),
                f"mfcc_{i+1:02}_skewness": skew(coeff),
                f"mfcc_{i+1:02}_max": np.max(coeff),
                f"mfcc_{i+1:02}_min": np.min(coeff),
                f"mfcc_{i+1:02}_range": np.ptp(coeff)
            })

            d_coeff = delta[i]
            features.update({
                f"delta_{i+1:02}_mean": np.mean(d_coeff),
                f"delta_{i+1:02}_std": np.std(d_coeff),
                f"delta_{i+1:02}_max": np.max(d_coeff),
                f"delta_{i+1:02}_min": np.min(d_coeff),
                f"delta_{i+1:02}_range": np.ptp(d_coeff)
            })

            dd_coeff = delta2[i]
            features.update({
                f"delta2_{i+1:02}_mean": np.mean(dd_coeff),
                f"delta2_{i+1:02}_std": np.std(dd_coeff),
                f"delta2_{i+1:02}_max": np.max(dd_coeff),
                f"delta2_{i+1:02}_min": np.min(dd_coeff),
                f"delta2_{i+1:02}_range": np.ptp(dd_coeff)
            })

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
            
            # Define desired column groups
            column_groups = [
                ['patient_id', 'recording_date', 'recording_time', 'filename',
                 'original_duration', 'processed_duration'],
                [c for c in df.columns if c.startswith('mfcc')],
                [c for c in df.columns if c.startswith('delta')],
                list(PARSELMOUTH_FEATURES.keys()),
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