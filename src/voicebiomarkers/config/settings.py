# voicebiomarkers/config/settings.py
"""
Centralized Configuration Module

Uses Pydantic Settings for validation and type safety.
Supports environment variables and .env files.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with validation."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ===========================================================
    # Paths
    # ===========================================================
    input_dir: Path = Field(
        default=Path("data/2307"),
        description="Directory containing WAV files",
    )
    output_dir: Path = Field(
        default=Path("output"),
        description="Directory for output files",
    )

    # ===========================================================
    # Processing
    # ===========================================================
    sample_rate: int = Field(
        default=16000,
        description="Target sample rate in Hz",
    )
    n_mfcc: int = Field(
        default=72,
        description="Number of MFCC coefficients",
    )
    num_workers: int = Field(
        default=0,
        description="Number of parallel workers (0 = auto)",
    )

    # ===========================================================
    # Reproducibility
    # ===========================================================
    random_seed: int = Field(
        default=42,
        description="Random seed for reproducibility",
    )

    # ===========================================================
    # Output Format
    # ===========================================================
    output_format: Literal["csv", "parquet"] = Field(
        default="csv",
        description="Output file format",
    )

    # ===========================================================
    # Quality Control
    # ===========================================================
    qc_min_snr_db: float = Field(
        default=15.0,
        description="Minimum SNR in dB",
    )
    qc_exclude_low_quality: bool = Field(
        default=False,
        description="Exclude files that fail QC (vs. flag only)",
    )

    # ===========================================================
    # Logging
    # ===========================================================
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging verbosity",
    )
    log_file: Path | None = Field(
        default=None,
        description="Path to log file (None = stdout only)",
    )

    @field_validator("input_dir", "output_dir")
    @classmethod
    def validate_paths(cls, v: Path) -> Path:
        """Convert to absolute path."""
        return Path(v).resolve()

    @property
    def effective_workers(self) -> int:
        """Get effective number of workers."""
        if self.num_workers <= 0:
            return os.cpu_count() or 1
        return self.num_workers


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Convenience alias
settings = get_settings()
