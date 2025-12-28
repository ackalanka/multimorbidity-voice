# voicebiomarkers/utils/reproducibility.py
"""
Reproducibility Utilities

Provides tools for ensuring consistent, reproducible results
across different machines and runs.
"""

import hashlib
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def set_random_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.

    Also configures environment variables to ensure deterministic
    behavior in BLAS/LAPACK operations.

    Args:
        seed: Random seed value
    """
    np.random.seed(seed)

    # Force single-threaded BLAS for reproducibility
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"


def get_file_hash(file_path: str | Path, algorithm: str = "sha256") -> str:
    """
    Calculate hash of a file.

    Args:
        file_path: Path to file
        algorithm: Hash algorithm (sha256, md5, etc.)

    Returns:
        Hexadecimal hash string
    """
    hash_func = hashlib.new(algorithm)

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_func.update(chunk)

    return hash_func.hexdigest()


def get_data_hash(data: bytes, algorithm: str = "sha256") -> str:
    """
    Calculate hash of bytes data.

    Args:
        data: Bytes to hash
        algorithm: Hash algorithm

    Returns:
        Hexadecimal hash string
    """
    hash_func = hashlib.new(algorithm)
    hash_func.update(data)
    return hash_func.hexdigest()


def get_environment_info() -> dict[str, Any]:
    """
    Get environment information for reproducibility tracking.

    Returns:
        Dictionary with system and library versions
    """
    import librosa
    import pandas as pd
    import scipy

    try:
        import parselmouth
        parselmouth_version = parselmouth.__version__
    except (ImportError, AttributeError):
        parselmouth_version = "unknown"

    try:
        import noisereduce
        noisereduce_version = noisereduce.__version__
    except (ImportError, AttributeError):
        noisereduce_version = "unknown"

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "librosa_version": librosa.__version__,
        "pandas_version": pd.__version__,
        "parselmouth_version": parselmouth_version,
        "noisereduce_version": noisereduce_version,
    }


def get_git_commit() -> str | None:
    """
    Get current git commit hash if in a git repository.

    Returns:
        Git commit hash or None
    """
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return None


def create_provenance_record(
    input_files: list[str | Path],
    output_file: str | Path,
    pipeline_version: str = "1.0.0",
) -> dict[str, Any]:
    """
    Create a provenance record for audit trail.

    Args:
        input_files: List of input file paths
        output_file: Output file path
        pipeline_version: Version of the extraction pipeline

    Returns:
        Provenance record dictionary
    """
    input_hashes = {
        str(f): get_file_hash(f)
        for f in input_files
        if Path(f).exists()
    }

    output_hash = None
    if Path(output_file).exists():
        output_hash = get_file_hash(output_file)

    return {
        "pipeline_version": pipeline_version,
        "git_commit": get_git_commit(),
        "environment": get_environment_info(),
        "input_files_count": len(input_files),
        "input_hashes_sample": dict(list(input_hashes.items())[:5]),
        "output_hash": output_hash,
        "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
    }
