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
OUTPUT_CSV = "voice_features.csv"
SAMPLE_RATE = 16000
N_MFCC = 72

# Processing Parameters
MIN_VOICE_DURATION = 0.5
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
                ['patient_id', 'gender', 'age', 'recording_date', 'recording_time', 'filename',
                 'original_duration', 'processed_duration'],
                [c for c in df.columns if c.startswith('mfcc')],
                [c for c in df.columns if c.startswith('delta')],
                ['snr_db', 'trim_start', 'trim_end']
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
                "valid_metadata": (~final_df["patient_id"].str.contains("unknown")).mean()
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