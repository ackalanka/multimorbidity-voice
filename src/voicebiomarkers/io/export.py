# voicebiomarkers/io/export.py
"""
Feature Export

Exports extracted features to CSV and Parquet formats.
Generates QC reports.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from voicebiomarkers.utils.reproducibility import get_environment_info


def export_features(
    features: list[dict[str, Any]],
    output_path: str | Path,
    format: str = "csv",
    sort_by: str = "filename",
) -> Path:
    """
    Export features to file.

    Args:
        features: List of feature dictionaries
        output_path: Output file path
        format: Output format ("csv" or "parquet")
        sort_by: Column to sort by for reproducibility

    Returns:
        Path to output file
    """
    output_path = Path(output_path)

    df = pd.DataFrame(features)

    # Sort for reproducibility
    if sort_by in df.columns:
        df = df.sort_values(sort_by).reset_index(drop=True)

    # Export
    if format.lower() == "parquet":
        if not output_path.suffix == ".parquet":
            output_path = output_path.with_suffix(".parquet")
        df.to_parquet(output_path, index=False)
    else:
        if not output_path.suffix == ".csv":
            output_path = output_path.with_suffix(".csv")
        df.to_csv(output_path, index=False)

    return output_path


def export_qc_report(
    qc_summary: dict[str, Any],
    output_path: str | Path,
    include_env_info: bool = True,
) -> Path:
    """
    Export QC summary report as JSON.

    Args:
        qc_summary: QC summary dictionary
        output_path: Output file path
        include_env_info: Include environment information

    Returns:
        Path to output file
    """
    output_path = Path(output_path)

    report = {
        "report_timestamp": datetime.now(UTC).isoformat(),
        "qc_summary": qc_summary,
    }

    if include_env_info:
        report["environment"] = get_environment_info()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    return output_path


def generate_column_order(
    df: pd.DataFrame,
    priority_groups: list[list[str]] | None = None,
) -> list[str]:
    """
    Generate ordered column list for output.

    Args:
        df: DataFrame with features
        priority_groups: List of column groups in priority order

    Returns:
        Ordered list of columns
    """
    if priority_groups is None:
        priority_groups = [
            # Metadata first
            ["patient_id", "gender", "age", "recording_date", "recording_time", "filename"],
            # Duration/processing
            ["original_duration", "processed_duration"],
            # QC metrics
            ["qc_passed", "snr_db", "clipping_percent", "silence_percent"],
        ]

    available = set(df.columns)
    ordered = []

    # Add priority columns in order
    for group in priority_groups:
        for col in group:
            if col in available:
                ordered.append(col)
                available.remove(col)

    # Add remaining columns alphabetically
    ordered.extend(sorted(available))

    return ordered
