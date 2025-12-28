#!/usr/bin/env python
# scripts/extract_features.py
"""
Voice Biomarkers Feature Extraction CLI

Command-line interface for extracting MFCC and Praat features
from voice recordings.

Usage:
    python scripts/extract_features.py --input data/2307 --output output/
    python scripts/extract_features.py --help
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from voicebiomarkers.config.settings import Settings
from voicebiomarkers.core.mfcc import extract_mfcc_features
from voicebiomarkers.core.praat import extract_praat_features, PARSELMOUTH_AVAILABLE
from voicebiomarkers.core.preprocessing import preprocess_audio
from voicebiomarkers.core.quality import calculate_quality_metrics, generate_batch_qc_report
from voicebiomarkers.io.export import export_features, export_qc_report
from voicebiomarkers.io.metadata import parse_filename_metadata
from voicebiomarkers.utils.logging import setup_logging, get_logger
from voicebiomarkers.utils.reproducibility import set_random_seed, create_provenance_record


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract voice biomarker features from audio files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Extract all features from directory
    python scripts/extract_features.py -i data/2307 -o output/
    
    # Extract only MFCC features
    python scripts/extract_features.py -i data/2307 -o output/ --mfcc-only
    
    # Extract only Praat features
    python scripts/extract_features.py -i data/2307 -o output/ --praat-only
        """,
    )
    
    parser.add_argument(
        "-i", "--input",
        type=Path,
        required=True,
        help="Input directory containing WAV files",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("output"),
        help="Output directory for CSV files (default: output/)",
    )
    parser.add_argument(
        "--mfcc-only",
        action="store_true",
        help="Extract only MFCC features",
    )
    parser.add_argument(
        "--praat-only",
        action="store_true",
        help="Extract only Praat features",
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=0,
        help="Number of worker processes (0 = auto)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    
    return parser.parse_args()


def process_single_file(
    file_path: Path,
    extract_mfcc: bool = True,
    extract_praat: bool = True,
    logger = None,
) -> Optional[dict]:
    """
    Process a single audio file and extract features.
    
    Args:
        file_path: Path to audio file
        extract_mfcc: Whether to extract MFCC features
        extract_praat: Whether to extract Praat features
        logger: Logger instance
        
    Returns:
        Dictionary of features or None if failed
    """
    if logger is None:
        logger = get_logger()
    
    filename = file_path.name
    features = {"filename": filename}
    
    try:
        # Parse metadata from filename
        metadata = parse_filename_metadata(filename)
        features.update(metadata)
        
        # Preprocess audio
        audio, preprocess_meta = preprocess_audio(str(file_path))
        features.update(preprocess_meta)
        
        # Check minimum duration
        if preprocess_meta["processed_duration"] < 0.5:
            logger.warning(f"Skipping {filename}: insufficient duration")
            return None
        
        # Keep original audio for QC
        from voicebiomarkers.io.audio import load_audio
        original_audio, _ = load_audio(str(file_path))
        
        # Extract MFCC features
        if extract_mfcc:
            mfcc_features = extract_mfcc_features(audio)
            features.update(mfcc_features)
        
        # Extract Praat features
        if extract_praat and PARSELMOUTH_AVAILABLE:
            praat_features = extract_praat_features(audio, 16000)
            features.update(praat_features)
        
        # Calculate quality metrics
        qc_metrics = calculate_quality_metrics(
            original_audio,
            audio,
            16000,
        )
        features.update(qc_metrics)
        
        logger.debug(f"Processed: {filename}")
        return features
        
    except Exception as e:
        logger.error(f"Failed to process {filename}: {e}")
        return None


def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(level=log_level)
    set_random_seed(args.seed)
    
    logger.info("="*60)
    logger.info("Voice Biomarkers Feature Extraction")
    logger.info("="*60)
    
    # Validate input
    if not args.input.exists():
        logger.error(f"Input directory not found: {args.input}")
        sys.exit(1)
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Find WAV files
    wav_files = list(args.input.glob("*.wav"))
    if not wav_files:
        logger.error(f"No WAV files found in {args.input}")
        sys.exit(1)
    
    logger.info(f"Found {len(wav_files)} WAV files")
    
    # Determine what to extract
    extract_mfcc = not args.praat_only
    extract_praat = not args.mfcc_only
    
    if args.praat_only and not PARSELMOUTH_AVAILABLE:
        logger.error("Parselmouth not available but --praat-only requested")
        sys.exit(1)
    
    logger.info(f"Extracting: MFCC={extract_mfcc}, Praat={extract_praat}")
    
    # Process files
    results = []
    qc_metrics_list = []
    
    for i, wav_file in enumerate(wav_files):
        features = process_single_file(
            wav_file,
            extract_mfcc=extract_mfcc,
            extract_praat=extract_praat,
            logger=logger,
        )
        
        if features:
            results.append(features)
            # Collect QC metrics
            qc_metrics_list.append({
                "qc_passed": features.get("qc_passed", True),
                "qc_low_snr": features.get("qc_low_snr", False),
                "qc_clipping": features.get("qc_clipping", False),
                "qc_mostly_silence": features.get("qc_mostly_silence", False),
                "qc_unvoiced": features.get("qc_unvoiced", False),
                "snr_db": features.get("snr_db"),
            })
        
        # Progress
        if (i + 1) % 10 == 0:
            logger.info(f"Progress: {i+1}/{len(wav_files)} ({len(results)} successful)")
    
    logger.info(f"Completed: {len(results)}/{len(wav_files)} files processed successfully")
    
    # Export features
    if results:
        # Determine output filename
        if extract_mfcc and extract_praat:
            output_name = "voice_features.csv"
        elif extract_mfcc:
            output_name = "mfcc_features.csv"
        else:
            output_name = "praat_features.csv"
        
        output_path = args.output / output_name
        export_features(results, output_path)
        logger.info(f"Features exported to: {output_path}")
        
        # Generate QC report
        qc_summary = generate_batch_qc_report(qc_metrics_list, len(wav_files))
        qc_path = args.output / "qc_report.json"
        export_qc_report(qc_summary, qc_path)
        logger.info(f"QC report exported to: {qc_path}")
        
        # Generate provenance
        provenance = create_provenance_record(
            input_files=[str(f) for f in wav_files[:10]],  # Sample
            output_file=output_path,
        )
        provenance_path = args.output / "provenance.json"
        import json
        with open(provenance_path, "w") as f:
            json.dump(provenance, f, indent=2, default=str)
        logger.info(f"Provenance exported to: {provenance_path}")
        
        # Print summary
        logger.info("="*60)
        logger.info("EXTRACTION COMPLETE")
        logger.info("="*60)
        logger.info(f"Total files: {len(wav_files)}")
        logger.info(f"Successful: {len(results)}")
        logger.info(f"Failed: {len(wav_files) - len(results)}")
        logger.info(f"QC passed: {qc_summary['passed_files']}")
        logger.info(f"QC failed: {qc_summary['failed_files']}")
    else:
        logger.error("No features extracted!")
        sys.exit(1)


if __name__ == "__main__":
    main()
