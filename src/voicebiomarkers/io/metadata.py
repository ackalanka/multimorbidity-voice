# voicebiomarkers/io/metadata.py
"""
Filename Metadata Parsing

Extracts patient metadata from structured filenames.
Supports the naming convention: ID-Gender-Age-Date-Time.wav
"""

import logging
import re
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# Filename pattern: ID-Gender-Age-Date-Time.wav
# Example: 2307-F-29-01.01.2024-09-34.wav
FILENAME_PATTERN = re.compile(
    r"^(\d+)-([FM])-(\d+)-(\d{2}\.\d{2}\.\d{4})-(\d{2}-\d{2})\.wav$",
    re.IGNORECASE,
)

# Alternative patterns for flexibility
ALT_PATTERNS = [
    # ID-Gender-Age-Date-Time.wav (with underscores)
    re.compile(r"^(\d+)_([FM])_(\d+)_(\d{2}\.\d{2}\.\d{4})_(\d{2}-\d{2})\.wav$", re.I),
    # ID-Gender-Age.wav (minimal)
    re.compile(r"^(\d+)-([FM])-(\d+)\.wav$", re.I),
]


def parse_filename_metadata(filename: str) -> dict[str, Any]:
    """
    Extract metadata from filename following the naming convention.

    Expected formats:
    - Full: ID-Gender-Age-Date-Time.wav (e.g., 2307-F-29-01.01.2024-09-34.wav)
    - Minimal: ID-Gender-Age.wav (e.g., 2307-F-29.wav)

    Args:
        filename: Audio filename (not full path)

    Returns:
        Dictionary with extracted metadata fields:
        - patient_id: str
        - gender: str ("F" or "M")
        - age: int
        - recording_date: str (optional)
        - recording_time: str (optional)
        - filename_error: str (if parsing failed)
    """
    metadata: dict[str, Any] = {
        "patient_id": "unknown",
        "gender": "unknown",
        "age": None,
        "recording_date": None,
        "recording_time": None,
        "filename_error": None,
    }

    try:
        # Try main pattern
        match = FILENAME_PATTERN.match(filename)

        if match:
            patient_id, gender, age, date, time = match.groups()

            metadata["patient_id"] = patient_id
            metadata["gender"] = gender.upper()
            metadata["age"] = int(age)

            # Parse date
            try:
                parsed_date = datetime.strptime(date, "%d.%m.%Y")
                metadata["recording_date"] = parsed_date.strftime("%Y-%m-%d")
            except ValueError:
                metadata["recording_date"] = date

            # Parse time (convert 09-34 to 09:34)
            metadata["recording_time"] = time.replace("-", ":")

            # Validate age range
            if not (0 <= metadata["age"] <= 120):
                metadata["filename_error"] = f"Age out of range: {metadata['age']}"
                logger.warning(f"Age out of range in {filename}: {metadata['age']}")

            # Validate gender
            if metadata["gender"] not in ("F", "M"):
                metadata["filename_error"] = f"Invalid gender: {metadata['gender']}"
                logger.warning(f"Invalid gender in {filename}")

        else:
            # Try alternative patterns
            for alt_pattern in ALT_PATTERNS:
                alt_match = alt_pattern.match(filename)
                if alt_match:
                    groups = alt_match.groups()
                    metadata["patient_id"] = groups[0]
                    metadata["gender"] = groups[1].upper()
                    metadata["age"] = int(groups[2])
                    if len(groups) > 3:
                        metadata["recording_date"] = groups[3]
                    if len(groups) > 4:
                        metadata["recording_time"] = groups[4].replace("-", ":")
                    break
            else:
                metadata["filename_error"] = "Filename pattern mismatch"
                logger.warning(f"Filename pattern mismatch: {filename}")

    except Exception as e:
        error_msg = f"Parsing error: {str(e)[:100]}"
        metadata["filename_error"] = error_msg
        logger.error(f"Metadata parsing failed for {filename}: {error_msg}")

    return metadata
