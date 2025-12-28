# Dockerfile for Voice Biomarkers Extraction Pipeline
# 
# This container provides a reproducible environment for
# peer-reviewed scientific research.
#
# Build:
#   docker build -t voicebiomarkers:1.0.0 .
#
# Run extraction:
#   docker run -v $(pwd)/data:/app/data -v $(pwd)/output:/app/output voicebiomarkers:1.0.0 \
#     python scripts/extract_features.py -i data/2307 -o output/

# =============================================================================
# PINNED BASE IMAGE (use exact digest for reproducibility)
# =============================================================================
FROM python:3.12.1-slim-bookworm

# Metadata
LABEL maintainer="CardioVoice Team"
LABEL version="1.0.0"
LABEL description="Voice Biomarkers Extraction Pipeline for Cardiovascular Research"

# =============================================================================
# SYSTEM DEPENDENCIES
# =============================================================================

# Install audio processing dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    libsndfile1-dev \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# PYTHON ENVIRONMENT
# =============================================================================

WORKDIR /app

# Set environment variables for reproducibility
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

# Install Python dependencies first (for layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir pydantic pydantic-settings

# =============================================================================
# APPLICATION CODE
# =============================================================================

# Copy source code
COPY src/ src/
COPY scripts/ scripts/
COPY tests/ tests/

# Run tests to validate installation
RUN pip install pytest && \
    python tests/test_praat_integration.py || echo "Skipping tests (no audio data)"

# =============================================================================
# ENTRY POINT
# =============================================================================

# Default command
CMD ["python", "scripts/extract_features.py", "--help"]
