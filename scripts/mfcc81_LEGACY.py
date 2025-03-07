import os
import librosa
import numpy as np
import pandas as pd
from datetime import datetime

# Function to extract MFCC features
def extract_mfcc_features(file_path):
    try:
        # Load audio
        y, sr = librosa.load(file_path, sr=16000)
        
        # Trim silence
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        
        # Extract MFCCs (13 coefficients, 40 Mel filters)
        mfcc = librosa.feature.mfcc(
            y=y_trimmed, sr=sr, n_mfcc=13,
            n_fft=int(0.025 * sr),  # 25ms frame
            hop_length=int(0.01 * sr),  # 10ms overlap
            n_mels=40
        )
        
        # Compute statistics for each coefficient (mean, var, median, max, min, range)
        features = []
        for coeff in mfcc:
            features.extend([
                np.mean(coeff), np.var(coeff), np.median(coeff),
                np.max(coeff), np.min(coeff), np.ptp(coeff)  # ptp = range (max - min)
            ])
        
        # Pad to 81 features if needed (13 MFCCs * 6 stats = 78 → pad with 3 zeros)
        features += [0.0] * (81 - len(features))
        return features
    
    except Exception as e:
        print(f"Error processing {os.path.basename(file_path)}: {str(e)}")
        return None

# Parse filename metadata
def parse_filename(filename):
    parts = filename.split("-")
    try:
        study_group = parts[0]  # "EU150"
        participant_num = parts[1]  # "001"
        date_str = parts[2]  # "14112023"
        time_str = f"{parts[3]}:{parts[4].split('.')[0]}"  # "17:56"
        
        # Format date to YYYY-MM-DD
        date = datetime.strptime(date_str, "%d%m%Y").strftime("%Y-%m-%d")
        
        return {
            "participant_id": f"{study_group}-{participant_num}",
            "date": date,
            "time": time_str,
            "study_group": study_group
        }
    except Exception as e:
        print(f"Failed to parse {filename}: {str(e)}")
        return None

# Main script
if __name__ == "__main__":
    # Configure paths
    input_folder = "C:/Users/Akalanka/Desktop/multimorbidity_voice/datasets/2023.11"
    output_csv = "mfcc_voice_features.csv"
    
    # Process files
    data = []
    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):
            file_path = os.path.join(input_folder, filename)
            
            # Extract metadata
            metadata = parse_filename(filename)
            if not metadata:
                continue
                
            # Extract features
            features = extract_mfcc_features(file_path)
            if features is None:
                continue
            
            # Combine metadata + features
            row = [
                metadata["participant_id"],
                metadata["date"],
                metadata["time"],
                metadata["study_group"]
            ] + features
            
            data.append(row)

    # Create DataFrame
    columns = [
        "participant_id", "date", "time", "study_group"
    ] + [f"feature_{i:03d}" for i in range(1, 82)]
    
    df = pd.DataFrame(data, columns=columns)
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"Success! Saved {len(df)} recordings to {output_csv}")