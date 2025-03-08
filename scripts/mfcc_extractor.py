"""
Voice Feature Extraction Pipeline

What's new?
- Includes comprehensive preprocessing
- Full delta/delta-delta statistics
- Maintains metadata parsing and quality control
"""

from typing import Dict, Optional
import os
import re
import numpy as np
import pandas as pd
import librosa
import logging
import noisereduce as nr
from datetime import datetime
from multiprocessing import Pool, cpu_count
from scipy.stats import kurtosis, skew

# Configuration
INPUT_DIR = r"C:\Users\Akalanka\Desktop\multimorbidity_voice\data\raw"
OUTPUT_CSV = "mfcc_features.csv"
SAMPLE_RATE = 16000
N_MFCC = 13

# Audio processing parameters
NOISE_REDUCTION_AGGRESSION = 1.5
MIN_VOICE_DURATION = 0.5

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("feature_extraction.log"),
        logging.StreamHandler()
    ]
)

def parse_filename_metadata(filename: str) -> Dict:
    """Extract patient ID, date, and time from filename"""
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
    """Complete processing pipeline with enhanced delta features"""
    try:
        filename = os.path.basename(file_path)
        features = {"filename": filename}
        
        # === Metadata Extraction ===
        metadata = parse_filename_metadata(filename)
        features.update(metadata)
        
        # === Audio Preprocessing ===
        y, sr = librosa.load(file_path, sr=None, mono=True)
        features["original_duration"] = len(y)/sr
        
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

        # Spectral normalization
        y_preemph = librosa.effects.preemphasis(y_trimmed, coef=0.97)
        y_normalized = librosa.util.normalize(y_preemph)

        # === Enhanced Feature Extraction ===
        mfcc = librosa.feature.mfcc(
            y=y_normalized, sr=SAMPLE_RATE,
            n_mfcc=N_MFCC, n_mels=128,
            fmin=50, fmax=8000,
            n_fft=2048, hop_length=512
        )

        # Calculate delta coefficients
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

            # Delta Statistics
            d_coeff = delta[i]
            features.update({
                f"delta_{i+1:02}_mean": np.mean(d_coeff),
                f"delta_{i+1:02}_std": np.std(d_coeff),
                f"delta_{i+1:02}_max": np.max(d_coeff),
                f"delta_{i+1:02}_min": np.min(d_coeff),
                f"delta_{i+1:02}_range": np.ptp(d_coeff)
            })

            # Delta-Delta Statistics
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
            "snr_db": 10 * np.log10(np.var(y_trimmed)/np.var(y[:1000])),
            "noise_reduction": NOISE_REDUCTION_AGGRESSION,
            "trim_start": start,
            "trim_end": end
        })

        return features

    except Exception as e:
        logging.error(f"Failed {filename}: {str(e)}", exc_info=True)
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

        logging.info(f"Processing {len(files)} voice recordings")

        with Pool(cpu_count()) as pool:
            results = []
            for i, result in enumerate(pool.imap(process_audio_file, files)):
                if result:
                    results.append(result)
                if (i+1) % 10 == 0:
                    logging.info(f"Processed {i+1}/{len(files)} files")
                    logging.info(f"Success rate: {len(results)/(i+1):.1%}")

        # Save results with column ordering
        df = pd.DataFrame(results)
        
        # Organized column structure
        base_cols = ['patient_id', 'recording_date', 'recording_time', 'filename',
                    'original_duration', 'processed_duration']
        mfcc_cols = [c for c in df.columns if c.startswith('mfcc')]
        delta_cols = [c for c in df.columns if c.startswith('delta')]
        quality_cols = ['snr_db', 'noise_reduction', 'trim_start', 'trim_end']
        
        df = df[base_cols + mfcc_cols + delta_cols + quality_cols]
        df.to_csv(OUTPUT_CSV, index=False)
        logging.info(f"Features saved to {OUTPUT_CSV}")

        # Quality report
        if not df.empty:
            quality_report = {
                "success_rate": len(df)/len(files),
                "mean_duration": df["processed_duration"].mean(),
                "valid_metadata": (~df["patient_id"].str.contains("unknown")).mean(),
                "total_features": len(df.columns) - len(base_cols + quality_cols),
                "snr_stats": df["snr_db"].describe().to_dict()
            }
            logging.info(f"Quality Report: {quality_report}")

    except Exception as e:
        logging.error(f"Fatal error: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()