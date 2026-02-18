"""
OCR Engine module.
Interacts with the underlying OCR library (EasyOCR) to extract raw text
from preprocessed images.
"""

import easyocr
import numpy as np
from typing import List, Tuple, Union
from pathlib import Path

# Add project root to Python path to allow importing config
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import settings
from utils import io_utils

# Global instance for caching the reader
_READER = None

def load_ocr_engine(languages: List[str] = ['en']):
    """
    Initialize and configure the OCR engine (EasyOCR).
    Uses a singleton pattern to avoid reloading model.
    Configures custom model storage directory.
    """
    global _READER
    if _READER is None:
        print(f"Loading EasyOCR model from {settings.OCR_MODEL_DIR}...")
        
        # Ensure model directory exists
        io_utils.ensure_directory(settings.OCR_MODEL_DIR)
        
        _READER = easyocr.Reader(
            languages,
            model_storage_directory=str(settings.OCR_MODEL_DIR),
            download_enabled=True
        )
    return _READER

def extract_text(image: Union[np.ndarray, str], config=None) -> Tuple[str, float]:
    """
    Perform text extraction on the provided image using EasyOCR.
    Uses detail=1 to retrieve confidence scores for each detected text region.

    Args:
        image: The preprocessed image to analyze (numpy array or path).
        config: Optional configuration dictionary. 
                Supported keys: 'allowlist', 'batch_size', 'adjust_contrast'.

    Returns:
        Tuple[str, float]: A tuple of:
            - The extracted raw text string (concatenated).
            - The average confidence score (0.0–1.0) across all detections.
    """
    reader = load_ocr_engine()
    
    # Default configs
    allowlist = config.get('allowlist') if config else None
    
    # detail=1 returns list of (bbox, text, confidence) tuples
    results = reader.readtext(
        image, 
        detail=1,
        allowlist=allowlist,
        batch_size=1
    )
    
    if not results:
        return "", 0.0
    
    # Extract text strings and confidence scores
    texts = [entry[1] for entry in results]
    confidences = [entry[2] for entry in results]
    
    joined_text = " ".join(texts)
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    
    return joined_text, avg_confidence
