"""
Main pipeline orchestration module.
Connects preprocessing, OCR, and post-processing steps into a single workflow.
Supports an optional Debug Mode with step-by-step OpenCV visualization.
"""

import logging
import time
import cv2
import numpy as np
from pathlib import Path

from config import settings
from utils import io_utils
from . import preprocessing
from . import postprocess
from ocr_pipeline import ocr_engine
from ocr_pipeline.schemas import OCRResult, ErrorTroubleshooting
from PIL import Image

logger = logging.getLogger(__name__)


def _debug_show(title: str, image, debug_mode: bool) -> None:
    """
    Display an image in an OpenCV window if debug mode is active.
    Pauses execution until the user presses any key.
    Images exist in memory only — nothing is saved to disk.

    Args:
        title (str): Window title describing the pipeline step.
        image: The image to display (numpy array).
        debug_mode (bool): Whether debug visualization is enabled.
    """
    if not debug_mode:
        return
    try:
        cv2.imshow(title, image)
        cv2.waitKey(0)
    except cv2.error as e:
        # highgui module not available (headless build or no display)
        logger.warning(
            f"Cannot display '{title}' — OpenCV GUI not available. "
            f"Install 'opencv-python' (not headless) and ensure a display is connected. "
            f"Error: {e}"
        )


def run_pipeline(image_path: str, debug_mode: bool = False) -> OCRResult:
    """
    Run the full OCR pipeline on an image file.

    Steps:
    1. Load image.
    2. Auto-crop display region.
    3. Preprocess (channel extraction, threshold, morphology).
    4. Extract text using OCR engine (with confidence).
    5. Post-process, validate, and fuzzy-match result.

    Args:
        image_path (str): Path to the input image.
        debug_mode (bool): If True, opens OpenCV windows at each step.

    Returns:
        OCRResult: A Pydantic model containing the extracted text and confidence/metadata.
    """
    logger.info(f"Starting OCR pipeline for image: {image_path}")
    pipeline_start = time.perf_counter()
    
    # Load error mapping
    mapping_path = settings.ERROR_CODES_PATH.parent / "error_code_mapping.json"
    try:
        error_mapping = io_utils.load_json(mapping_path)
        allowed_codes = list(error_mapping.keys())
    except Exception as e:
        logger.error(f"Could not load error mapping: {e}")
        error_mapping = {}
        allowed_codes = []

    # Build dynamic regex from error codes
    regex_pattern = postprocess.build_dynamic_regex(allowed_codes)
    logger.debug(f"Dynamic regex pattern: {regex_pattern}")

    # ================================================================
    # STAGE 1: Load Image
    # ================================================================
    image = cv2.imread(str(image_path))
    if image is None:
        logger.error(f"Failed to load image at {image_path}")
        return OCRResult(raw_text="", is_valid=False, clean_text="Error: Image Load Failed")
        
    logger.debug("Loading image... Done.")
    _debug_show("Step 1: Original Image", image, debug_mode)

    # ================================================================
    # STAGE 2: Auto-Crop Display Region
    # ================================================================
    preprocess_start = time.perf_counter()

    cropped, debug_annotated = preprocessing.auto_crop_display(image)

    # Debug: show annotated image (red=rejected, green=hero) alongside crop
    if debug_mode:
        crop_h, crop_w = cropped.shape[:2]
        dbg_h, dbg_w = debug_annotated.shape[:2]
        # Scale annotated image to same height as crop for side-by-side
        scale = crop_h / dbg_h if dbg_h > 0 else 1
        dbg_resized = cv2.resize(debug_annotated, (int(dbg_w * scale), crop_h))
        crop_display = cropped if len(cropped.shape) == 3 else cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR)
        side_by_side = np.hstack([dbg_resized, crop_display])
        _debug_show("Step 2: Contour Filter [RED=rejected GREEN=hero] | Crop", side_by_side, debug_mode)

    # ================================================================
    # STAGE 3: Channel Extraction
    # ================================================================
    if len(cropped.shape) == 3:
        b, g, r = cv2.split(cropped)
        max_b, max_g, max_r = b.max(), g.max(), r.max()
        if max_g > max_r and max_g > max_b:
            logger.debug("Detected Green LED. Using Green channel.")
            gray = g
        elif max_r > max_g and max_r > max_b:
            logger.debug("Detected Red LED. Using Red channel.")
            gray = r
        else:
            gray = preprocessing.convert_to_grayscale(cropped)
    else:
        gray = cropped

    _debug_show("Step 3: Channel Extraction", gray, debug_mode)

    # Resize (Upscale) - 300% to make segments thick enough for closing
    upscaled = preprocessing.resize_image(gray, scale_percent=300)
    
    # Denoise
    denoised = preprocessing.denoise_image(upscaled)

    # Boost local contrast so dim segments survive binarization
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    denoised = clahe.apply(denoised)

    # ================================================================
    # STAGE 4: Binarization
    # ================================================================
    binary = preprocessing.apply_thresholding(denoised)
    _debug_show("Step 4: Binarization (Threshold)", binary, debug_mode)
    
    # --- LED/7-Segment Handling ---
    white_pixel_ratio = cv2.countNonZero(binary) / (binary.shape[0] * binary.shape[1])
    
    if white_pixel_ratio > 0.5:
        logger.debug("Detected dark text on light background. Inverting for processing...")
        processing_canvas = preprocessing.invert_image(binary)
    else:
        logger.debug("Detected light text on dark background (Standard LED).")
        processing_canvas = binary

    # ================================================================
    # STAGE 5: Safe-Limit Directional Closing + Bridge Snapper
    # ================================================================
    crop_height = processing_canvas.shape[0]

    # Pass 1: Vertical Snap (12% reach — fixes 'd', 'b', '8')
    v_len = max(3, int(crop_height * 0.18))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
    processing_canvas = cv2.morphologyEx(processing_canvas, cv2.MORPH_CLOSE, v_kernel)
    logger.debug(f"Stage 5a — Vertical snap: 1x{v_len} (crop height: {crop_height}px)")

    # Pass 2: Horizontal Snap (4% reach — reduced from 6% for tighter safety)
    h_len = max(3, int(crop_height * 0.04))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    processing_canvas = cv2.morphologyEx(processing_canvas, cv2.MORPH_CLOSE, h_kernel)
    logger.debug(f"Stage 5b — Horizontal snap: {h_len}x1")

    # Pass 3: Corner Polish (2x2 closing — no bolding)
    polish_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    processing_canvas = cv2.morphologyEx(processing_canvas, cv2.MORPH_CLOSE, polish_kernel)
    logger.debug("Stage 5c — Corner polish: 2x2 closing")

    # Pass 4: Bridge Snapper — horizontal erosion to break hairline inter-digit bridges
    # Solid 'U' bottom survives; weak 1px bridges between 'U' and 'd' get snapped
    # Only apply if there's enough white content to survive the erosion
    white_ratio = cv2.countNonZero(processing_canvas) / (processing_canvas.shape[0] * processing_canvas.shape[1])
    if white_ratio > 0.15:
        snapper_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
        processing_canvas = cv2.erode(processing_canvas, snapper_kernel, iterations=1)
        logger.debug(f"Stage 5d — Bridge snapper: 2x1 horizontal erosion (white_ratio={white_ratio:.2f})")
    else:
        logger.debug(f"Stage 5d — Bridge snapper SKIPPED (white_ratio={white_ratio:.2f} <= 0.15)")

    _debug_show("Step 5: Directional Closing + Bridge Snapper", processing_canvas, debug_mode)

    # --- Final Prep for OCR ---
    final_processed = preprocessing.invert_image(processing_canvas)
    
    # Add borders to ensure text isn't touching edges
    final_processed = cv2.copyMakeBorder(
        final_processed, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[255, 255, 255]
    )
    
    processed_img = final_processed
    preprocess_end = time.perf_counter()
    preprocess_time_ms = (preprocess_end - preprocess_start) * 1000

    # ================================================================
    # STAGE 6: OCR Inference
    # ================================================================
    logger.debug("Running OCR engine...")
    ocr_start = time.perf_counter()
    
    ocr_config = {
        'allowlist': 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
    }
    raw_text, ocr_confidence = ocr_engine.extract_text(processed_img, config=ocr_config)
    
    # ── Lightweight Fallback: if primary pipeline produced nothing,
    #    retry on minimally-processed crop (skip binarization + morphology) ──
    if not raw_text or not raw_text.strip():
        logger.debug("Primary OCR returned empty — trying lightweight fallback on raw crop...")
        
        # Use the cropped image from Stage 2 (before heavy preprocessing)
        if len(cropped.shape) == 3:
            fallback_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        else:
            fallback_gray = cropped
        
        # Minimal processing: upscale → CLAHE → border
        fallback_up = preprocessing.resize_image(fallback_gray, scale_percent=300)
        fallback_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        fallback_enhanced = fallback_clahe.apply(fallback_up)
        fallback_final = cv2.copyMakeBorder(
            fallback_enhanced, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[255, 255, 255]
        )
        
        fb_text, fb_conf = ocr_engine.extract_text(fallback_final, config=ocr_config)
        if fb_text and fb_text.strip():
            raw_text = fb_text
            ocr_confidence = fb_conf
            logger.debug(f"Fallback OCR produced: '{raw_text}' (conf={ocr_confidence:.2%})")

    ocr_end = time.perf_counter()
    ocr_time_ms = (ocr_end - ocr_start) * 1000
    logger.debug(f"Raw OCR output: {raw_text} (confidence: {ocr_confidence:.2%})")

    # ================================================================
    # STAGE 7: Post-processing & Validation
    # ================================================================
    logger.debug("Post-processing text...")
    clean_text = postprocess.clean_text(raw_text, regex_pattern=regex_pattern)
    
    # Apply specific 7-segment corrections
    clean_text = postprocess.correct_common_misreads(clean_text)
    
    # Validate with fuzzy matching
    is_valid, match_type, matched_code = postprocess.validate_error_code(
        clean_text, allowed_codes
    )
    
    error_details = None
    if is_valid and matched_code:
        clean_text = matched_code  # Normalize to official code
        
        details = error_mapping.get(matched_code)
        if details:
            if isinstance(details, list):
                error_details = [ErrorTroubleshooting(**d) for d in details]
            else:
                error_details = ErrorTroubleshooting(**details)

    # Close all debug windows
    if debug_mode:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass  # headless build — no windows to destroy

    pipeline_end = time.perf_counter()
    total_time_ms = (pipeline_end - pipeline_start) * 1000

    # Log performance metrics
    logger.info(
        f"Pipeline completed — "
        f"Preprocess: {preprocess_time_ms:.1f}ms | "
        f"OCR: {ocr_time_ms:.1f}ms | "
        f"Total: {total_time_ms:.1f}ms | "
        f"Match: {match_type}"
    )

    # Build metrics dict (kept separate to not modify OCRResult schema)
    metrics = {
        "preprocess_time_ms": round(preprocess_time_ms, 2),
        "ocr_time_ms": round(ocr_time_ms, 2),
        "total_time_ms": round(total_time_ms, 2),
        "match_type": match_type,
    }

    # Return structured Pydantic model + separate metrics
    result = OCRResult(
        raw_text=raw_text,
        clean_text=clean_text,
        confidence=ocr_confidence,
        is_valid=is_valid,
        error_details=error_details,
    )

    return result, metrics

