"""
Script to run the OCR pipeline on an image.
Produces a professional engineering report with performance metrics,
confidence scores, and match type classification.

Usage: python scripts/run_ocr.py --image <path_to_image> [--debug]
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import settings
from ocr_pipeline import pipeline
from ocr_pipeline.postprocess import DIRECT_MATCH, FUZZY_MATCH, NO_MATCH

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_report(result, metrics, image_path: str) -> None:
    """
    Print a professional engineering report with OCR results and metrics.

    Args:
        result: OCRResult Pydantic model.
        metrics: Dict with timing and match metadata.
        image_path: Path to the processed image.
    """
    match_type = metrics.get("match_type", NO_MATCH)

    # Match type display with indicator
    match_icons = {
        DIRECT_MATCH: "✅ DIRECT_MATCH",
        FUZZY_MATCH:  "🔄 FUZZY_MATCH",
        NO_MATCH:     "❌ NO_MATCH",
    }
    match_display = match_icons.get(match_type, f"❓ {match_type}")

    print()
    print("=" * 60)
    print("       OCR PIPELINE — ENGINEERING REPORT")
    print("=" * 60)
    print(f"  Image       : {image_path}")
    print("-" * 60)

    # --- Results Section ---
    print("  📋 RESULTS")
    print(f"     Raw Text     : {result.raw_text or '(empty)'}")
    print(f"     Clean Text   : {result.clean_text or '(empty)'}")
    print(f"     Valid Code   : {'Yes' if result.is_valid else 'No'}")
    print(f"     Match Type   : {match_display}")
    print(f"     Confidence   : {result.confidence:.2%}")
    print("-" * 60)

    # --- Performance Section ---
    print("  ⏱  PERFORMANCE METRICS")
    print(f"     Preprocessing / Cropping : {metrics.get('preprocess_time_ms', 0):>8.2f} ms")
    print(f"     OCR Inference            : {metrics.get('ocr_time_ms', 0):>8.2f} ms")
    print(f"     Total Pipeline           : {metrics.get('total_time_ms', 0):>8.2f} ms")
    print("-" * 60)

    # --- Error Details Section ---
    if result.error_details:
        print("  🔧 ERROR DETAILS")
        details_list = result.error_details if isinstance(result.error_details, list) else [result.error_details]
        for i, detail in enumerate(details_list, 1):
            if len(details_list) > 1:
                print(f"     --- Interpretation {i} ---")
            print(f"     Name            : {detail.name}")
            print(f"     Description     : {detail.description}")
            print(f"     Troubleshooting : {detail.troubleshooting}")
        print("-" * 60)
    elif result.is_valid:
        print("  ℹ  No detailed troubleshooting info found for this code.")
        print("-" * 60)

    print("=" * 60)
    print()


def main():
    """
    Main function to run the OCR pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Run OCR on an appliance display image."
    )
    parser.add_argument(
        "--image",
        help="Path to input image",
        required=False
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with step-by-step OpenCV visualization"
    )
    args = parser.parse_args()

    image_path = args.image
    if not image_path:
        logger.warning("No image provided. Running in demo mode with placeholder.")
        image_path = "demo_image.png"

    logger.info(f"Processing image: {image_path}")
    
    # Run the pipeline
    try:
        result, metrics = pipeline.run_pipeline(
            image_path,
            debug_mode=args.debug
        )
        print_report(result, metrics, image_path)
        
    except Exception as e:
        logger.error(f"An error occurred during pipeline execution: {e}")
        raise

if __name__ == "__main__":
    main()
