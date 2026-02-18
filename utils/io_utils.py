
"""
Input/Output utility functions for file handling, directory management, 
and data loading/saving.
"""

import json
import os
from pathlib import Path

def ensure_directory(path: Path) -> None:
    """
    Ensure that the directory exists. If not, create it.
    
    Args:
        path (Path): The pathlib.Path object for the directory.
    """
    path.mkdir(parents=True, exist_ok=True)

def load_json(file_path: Path) -> dict:
    """
    Load data from a JSON file.

    Args:
        file_path (Path): Path to the JSON file.

    Returns:
        dict: The loaded data.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")
        
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data: dict, file_path: Path) -> None:
    """
    Save data to a JSON file.

    Args:
        data (dict): The dictionary to save.
        file_path (Path): Destination path.
    """
    ensure_directory(file_path.parent)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

def save_image(image_data, file_path: Path) -> None:
    """
    Save image data to disk.
    
    Args:
        image_data: Image object/array to save (PIL Image or numpy array).
        file_path (Path): Destination path.
    """
    ensure_directory(file_path.parent)
    
    # Handle numpy array (OpenCV)
    import numpy as np
    import cv2
    from PIL import Image
    
    if isinstance(image_data, np.ndarray):
        cv2.imwrite(str(file_path), image_data)
    elif isinstance(image_data, Image.Image):
        image_data.save(file_path)
    else:
        raise ValueError("Unsupported image format")
