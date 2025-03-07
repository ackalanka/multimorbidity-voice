"""
Robust Voice Feature Extraction Pipeline for Thrombosis Prediction
- Handles single recordings per patient
- Version compatible with librosa 0.10.2
- Includes comprehensive preprocessing
"""

import os
import numpy as np
import pandas as pd
import librosa
import logging
import noisereduce as nr
from multiprocessing import Pool, cpu_count
from typing import Dict, Optional
from scipy.stats import kurtosis, skew

# Configuration
INPUT_DIR = r"C:\Users\Akalanka\Desktop\multimorbidity_voice\data\raw"
OUTPUT_CSV = "mfcc_features.csv"
SAMPLE_RATE = 16000
N_MFCC = 13

# Audio processing parameters
NOISE_REDUCTION_AGGRESSION = 1.5
MIN_VOICE_DURATION = 0.5  # Seconds of valid audio required

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("feature_extraction.log"),
        logging.StreamHandler()
    ]
)

def process_audio_file(file_path: str) -> Optional[Dict]:
    """Robust audio processing pipeline with fixed feature extraction"""
    try:
        filename = os.path.basename(file_path)
        features = {"filename": filename}
        
        # 1. Load audio with safety checks
        y, sr = librosa.load(file_path, sr=None, mono=True)
        features["original_duration"] = len(y)/sr
        
        # 2. Resample with anti-aliasing
        if sr != SAMPLE_RATE:
            y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
        
        # 3. Advanced noise reduction
        noise_profile = y[:int(SAMPLE_RATE * 0.5)] if len(y) > SAMPLE_RATE//2 else y
        y_clean = nr.reduce_noise(
            y=y,
            y_noise=noise_profile,
            sr=SAMPLE_RATE,
            stationary=True,
            prop_decrease=NOISE_REDUCTION_AGGRESSION,
            n_fft=2048,
            hop_length=512
        )
        
        # 4. Adaptive voice activity detection
        y_trimmed, (start, end) = librosa.effects.trim(
            y_clean, 
            top_db=25,  # Conservative trimming
            frame_length=2048,
            hop_length=512
        )
        features["processed_duration"] = len(y_trimmed)/SAMPLE_RATE
        
        # 5. Validate sufficient voice content
        if features["processed_duration"] < MIN_VOICE_DURATION:
            logging.error(f"Insufficient voice content: {filename}")
            return None

        # 6. Spectral normalization
        y_preemph = librosa.effects.preemphasis(y_trimmed, coef=0.97)
        y_normalized = librosa.util.normalize(y_preemph)

        # 7. MFCC extraction with fixed parameters
        mfcc = librosa.feature.mfcc(
            y=y_normalized,
            sr=SAMPLE_RATE,
            n_mfcc=N_MFCC,
            n_mels=128,
            fmin=50,
            fmax=8000,
            n_fft=2048,
            hop_length=512
        )

        # 8. Robust feature engineering using SciPy
        for i in range(N_MFCC):
            coeff = mfcc[i]
            features.update({
                f"mfcc_{i+1:02}_mean": np.mean(coeff),
                f"mfcc_{i+1:02}_std": np.std(coeff),
                f"mfcc_{i+1:02}_kurtosis": kurtosis(coeff),
                f"mfcc_{i+1:02}_skewness": skew(coeff)
            })

        # 9. Temporal features
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        features.update({
            "delta_mean": np.mean(delta),
            "delta_std": np.std(delta),
            "delta2_mean": np.mean(delta2)
        })

        # 10. Quality metrics
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
    """Main processing workflow with error handling"""
    try:
        files = [os.path.join(INPUT_DIR, f) 
                for f in os.listdir(INPUT_DIR) 
                if f.lower().endswith(".wav")]
        
        if not files:
            logging.error("No WAV files found")
            return

        logging.info(f"Processing {len(files)} voice recordings")

        # Parallel processing with progress tracking
        with Pool(cpu_count()) as pool:
            results = []
            for i, result in enumerate(pool.imap(process_audio_file, files)):
                if result:
                    results.append(result)
                if (i+1) % 10 == 0:
                    logging.info(f"Processed {i+1}/{len(files)} files")
                    logging.info(f"Success rate: {len(results)/(i+1):.1%}")

        # Save features with metadata
        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_CSV, index=False)
        logging.info(f"Features saved to {OUTPUT_CSV}")

        # Generate quality report
        if not df.empty:
            quality_report = {
                "success_rate": len(df)/len(files),
                "mean_duration": df["processed_duration"].mean(),
                "snr_stats": df["snr_db"].describe().to_dict()
            }
            logging.info(f"Quality Report: {quality_report}")

    except Exception as e:
        logging.error(f"Fatal error: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()