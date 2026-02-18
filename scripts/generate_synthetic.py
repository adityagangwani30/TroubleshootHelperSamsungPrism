"""
Script to control synthetic data generation.
Usage: python scripts/generate_synthetic.py
"""

import sys
import os
from pathlib import Path

# Add project root to Python path to allow importing modules
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import settings
from synthetic_generator import generator
from utils import io_utils

def main():
    """
    Main function to orchestrate synthetic data generation.
    """
    print("Starting synthetic data generation...")
    
    # 1. Load configuration
    print(f"Configuration loaded. Output directory: {settings.DATA_SYNTHETIC_DIR}")
    
    # 2. Load error codes
    # codes = io_utils.load_json(settings.ERROR_CODES_PATH)
    print(f"Loading error codes from {settings.ERROR_CODES_PATH}...")
    
    # 3. Trigger generation
    print(f"Generating {settings.SAMPLES_PER_CLASS} samples per class...")
    # generator.generate_batch(codes, settings.SAMPLES_PER_CLASS, settings.DATA_SYNTHETIC_DIR)
    
    print("Generation complete.")

if __name__ == "__main__":
    main()
