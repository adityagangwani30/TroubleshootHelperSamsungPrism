"""
Preprocessing module for OCR images.
Handles tasks such as grayscale conversion, noise reduction, and thresholding
to improve OCR accuracy.
"""

import cv2
import numpy as np

def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image to grayscale using OpenCV.

    Args:
        image (np.ndarray): Input image array.

    Returns:
        np.ndarray: Grayscale image.
    """
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def apply_thresholding(image: np.ndarray) -> np.ndarray:
    """
    Apply thresholding to binarize the image using OpenCV.
    Uses Otsu's binarization.
    
    Args:
        image (np.ndarray): Grayscale input image.

    Returns:
        np.ndarray: Binary image.
    """
    # Apply Otsu's thresholding
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_image

def apply_adaptive_thresholding(image: np.ndarray) -> np.ndarray:
    """
    Apply adaptive thresholding (Gaussian C).
    Good for variable lighting conditions.
    
    Args:
        image (np.ndarray): Grayscale input image.
        
    Returns:
        np.ndarray: Binary image.
    """
    return cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

def denoise_image(image: np.ndarray) -> np.ndarray:
    """
    Remove noise from the image to enhance character clarity.
    Uses Gaussian Blur.

    Args:
        image (np.ndarray): Input binary/grayscale image.
    
    Returns:
        np.ndarray: Denoised image.
    """
    # Apply Gaussian Blur to remove high freq noise
    return cv2.GaussianBlur(image, (5, 5), 0)

def invert_image(image: np.ndarray) -> np.ndarray:
    """
    Invert the image colors (creates negative).
    Useful for converting black-on-white to white-on-black or vice versa.
    
    Args:
        image (np.ndarray): Input image.
        
    Returns:
        np.ndarray: Inverted image.
    """
    return cv2.bitwise_not(image)

def dilate_image(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Dilate the image. 
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(image, kernel, iterations=1)

def close_gaps(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Apply morphological closing to connect segments.
    Closing is Dilation followed by Erosion. It fills small holes and gaps between segments
    while maintaining the overall shape size better than pure dilation.
    
    Args:
        image (np.ndarray): Input binary image.
        kernel_size (int): Size of the kernel.
    
    Returns:
        np.ndarray: Processed image.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

def resize_image(image: np.ndarray, scale_percent: int = 150) -> np.ndarray:
    """
    Resize image to improve OCR readability on small text.
    
    Args:
        image (np.ndarray): Input image.
        scale_percent (int): Percent of original size.
        
    Returns:
        np.ndarray: Resized image.
    """
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_CUBIC)


def auto_crop_display(image: np.ndarray, padding_ratio: float = 0.2):
    """
    Locate and crop the 7-segment LED display using the Dominant Character
    Cluster algorithm — a generalized approach that works for any error code
    (e.g., "5E", "Ub", "End", "dC") at any position in the image.

    Instead of assuming a fixed location, this finds the "Most Significant
    Text-Like Object" by:
      1. HSV + Brightness filtering to isolate glowing LED pixels.
      2. Finding all candidate contours (digits + dots + noise).
      3. Relative Height filtering — anything < 50% of the tallest is noise.
      4. Horizontal Grouping — nearby tall contours form one "word" cluster.
      5. Dynamic padding (percentage-based, not pixel-based) for any resolution.

    Args:
        image (np.ndarray): Input BGR or grayscale image.
        padding_ratio (float): Padding as fraction of crop height (default 20%).

    Returns:
        tuple: (cropped_image, debug_annotated)
            - cropped_image: Cropped from the original unmodified image.
              Returns original image if no valid region is found.
            - debug_annotated: BGR image with blue boxes (tall digits) and
              red X's (rejected dots) drawn for debug visualization.
    """
    h, w = image.shape[:2]
    total_area = h * w
    import logging
    logger = logging.getLogger(__name__)

    # Ensure color image for HSV
    if len(image.shape) == 3:
        color_img = image
    else:
        color_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # ════════════════════════════════════════════════════════════════
    # PHASE A: Isolate glowing LED pixels (HSV + brightness fallback)
    # ════════════════════════════════════════════════════════════════
    hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([0, 30, 150])
    upper_bound = np.array([180, 255, 255])
    hsv_mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Fallback: if HSV finds almost nothing, use raw brightness
    if cv2.countNonZero(hsv_mask) < total_area * 0.001:
        gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
        _, hsv_mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)

    # ════════════════════════════════════════════════════════════════
    # STEP 1: Extract All Candidates
    # ════════════════════════════════════════════════════════════════
    # Light morphology to connect broken segments within a single digit
    # but NOT across separate characters
    connect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    connected = cv2.dilate(hsv_mask, connect_kernel, iterations=1)
    connected = cv2.morphologyEx(connected, cv2.MORPH_CLOSE, connect_kernel)

    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Prepare debug annotation image
    debug_annotated = color_img.copy()

    if not contours:
        return image, debug_annotated

    # Collect all bounding boxes
    all_boxes = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        box_area = cw * ch
        # Skip giant panels (> 50% of image)
        if box_area > total_area * 0.50:
            continue
        # Skip near-zero-area specks
        if box_area < total_area * 0.0001:
            continue
        all_boxes.append((x, y, cw, ch))

    if not all_boxes:
        return image, debug_annotated

    # ════════════════════════════════════════════════════════════════
    # STEP 2: Filter by Relative Height — "Big vs Small" Rule
    # ════════════════════════════════════════════════════════════════
    max_height = max(b[3] for b in all_boxes)
    height_threshold = max_height * 0.50  # 50% of tallest = minimum for digits

    tall_boxes = []   # Digit candidates
    small_boxes = []  # Rejected dots/noise

    for box in all_boxes:
        x, y, cw, ch = box
        if ch >= height_threshold:
            tall_boxes.append(box)
        else:
            small_boxes.append(box)

    # ── Debug: draw RED X's on small rejected dots ──
    for (sx, sy, sw, sh) in small_boxes:
        cx, cy = sx + sw // 2, sy + sh // 2
        r = max(sw, sh) // 2 + 2
        cv2.line(debug_annotated, (cx - r, cy - r), (cx + r, cy + r), (0, 0, 255), 2)
        cv2.line(debug_annotated, (cx + r, cy - r), (cx - r, cy + r), (0, 0, 255), 2)

    # ── Debug: draw BLUE boxes on tall digit candidates ──
    for (tx, ty, tw, th) in tall_boxes:
        cv2.rectangle(debug_annotated, (tx, ty), (tx + tw, ty + th), (255, 180, 0), 2)

    if not tall_boxes:
        logger.warning("Auto-crop: no tall contours survived height filter.")
        return image, debug_annotated

    # ════════════════════════════════════════════════════════════════
    # STEP 3: Horizontal Grouping — "Sentence" Rule
    # ════════════════════════════════════════════════════════════════
    # Sort tall contours left-to-right
    tall_boxes.sort(key=lambda b: b[0])

    # Group contours that are horizontally close
    groups = []
    current_group = [tall_boxes[0]]

    for i in range(1, len(tall_boxes)):
        prev = current_group[-1]
        curr = tall_boxes[i]

        prev_right = prev[0] + prev[2]  # x + w
        gap = curr[0] - prev_right       # horizontal gap

        # "Close" = gap is less than the average width of the two contours
        avg_width = (prev[2] + curr[2]) / 2.0
        if gap < avg_width * 1.5:
            current_group.append(curr)
        else:
            groups.append(current_group)
            current_group = [curr]
    groups.append(current_group)

    # Pick the group with the MOST members (the error code has 2-4 digits)
    # In case of tie, pick the one closest to the vertical center
    center_y = h / 2.0
    best_group = max(groups, key=lambda g: (
        len(g),  # Primary: most members
        -abs(sum(b[1] + b[3] / 2.0 for b in g) / len(g) - center_y)  # Secondary: central
    ))

    # ════════════════════════════════════════════════════════════════
    # STEP 4: The Final Crop
    # ════════════════════════════════════════════════════════════════
    x_min = min(b[0] for b in best_group)
    y_min = min(b[1] for b in best_group)
    x_max = max(b[0] + b[2] for b in best_group)
    y_max = max(b[1] + b[3] for b in best_group)

    crop_h = y_max - y_min
    crop_w = x_max - x_min

    # ── Validation Check ──
    aspect = crop_w / crop_h if crop_h > 0 else 0
    if aspect < 0.2 or aspect > 5.0:
        logger.warning(
            f"Auto-crop: suspicious aspect ratio {aspect:.2f} "
            f"(expected 0.2–5.0). Proceeding anyway."
        )

    # Draw GREEN rectangle for the final accepted cluster
    cv2.rectangle(debug_annotated, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)

    # Dynamic padding: percentage of crop height (resolution-independent)
    pad = int(crop_h * padding_ratio)

    crop_x1 = max(0, x_min - pad)
    crop_y1 = max(0, y_min - pad)
    crop_x2 = min(w, x_max + pad)
    crop_y2 = min(h, y_max + pad)

    cropped = image[crop_y1:crop_y2, crop_x1:crop_x2]

    return cropped, debug_annotated
