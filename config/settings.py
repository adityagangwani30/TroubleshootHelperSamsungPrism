"""
Configuration settings for the OCR-based appliance troubleshooting system.
This module contains global configurations including file paths, 
generation parameters, and image settings.
"""

import os
from pathlib import Path

# Base directory setup
BASE_DIR = Path(__file__).resolve().parent.parent

# --- Path Configurations ---
# Directory for real (captured) data
DATA_REAL_DIR = BASE_DIR / "data" / "real"

# Directory for synthetic data (training/testing infrastructure)
DATA_SYNTHETIC_DIR = BASE_DIR / "data" / "synthetic"

# Directory for configuration assets (fonts, templates, etc.)
ASSETS_DIR = BASE_DIR / "config" / "assets"

# Directory where EasyOCR models will be stored
OCR_MODEL_DIR = ASSETS_DIR / "models"

# Path to the washing machine error codes definition JSON
# This file should define the possible error codes and their properties
ERROR_CODES_PATH = BASE_DIR / "data" / "washing_machine" / "error_codes.json"


# --- Synthetic Data Generation Settings ---
# Number of synthetic images to generate per error code
SAMPLES_PER_CLASS = 50

# Resolution for generated images (width, height)
IMAGE_RESOLUTION = (128, 64)

# Output image format (e.g., 'png', 'jpg')
OUTPUT_FORMAT = "png"

# Random seed for reproducibility
RANDOM_SEED = 42


# --- OCR Pipeline Settings ---
# Placeholder: Settings for the OCR engine, preprocessing thresholds, etc.
OCR_CONFIDENCE_THRESHOLD = 0.5
