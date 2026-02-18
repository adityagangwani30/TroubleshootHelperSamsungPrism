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


def auto_crop_display(image: np.ndarray, padding: int = 20):
    """
    Locate and crop the 7-segment LED display using Color-Intensity Isolation
    with strict Size & Geometry filtering.

    Pipeline:
      1. HSV Color Filtering — keeps Sat > 30 AND Val > 150 (removes printed text).
      2. Brightest Cluster Fallback — threshold at 220+ for white LEDs.
      3. Horizontal Joining — dilates to merge digit segments.
      4. Size Gating — rejects contours shorter than 5% of image height (indicator dots)
         and thin vertical lines (width < 2% of image width).
      5. Hero Cluster — merges all surviving large contours into one bounding box.
         If multiple clusters exist, prefers the one highest in the image (top-half bias).

    Args:
        image (np.ndarray): Input BGR or grayscale image.
        padding (int): Pixels of generous padding around the detected ROI.

    Returns:
        tuple: (cropped_image, debug_annotated)
            - cropped_image: Cropped from the original unmodified image.
              Returns original image if no valid region is found.
            - debug_annotated: BGR image with red rects (rejected) and green
              rect (accepted hero cluster) drawn for debug visualization.
    """
    h, w = image.shape[:2]
    total_area = h * w

    # Ensure color image for HSV
    if len(image.shape) == 3:
        color_img = image
    else:
        color_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # ── Step 1: HSV Color Filtering ────────────────────────────────
    hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([0, 30, 150])
    upper_bound = np.array([180, 255, 255])
    hsv_mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # ── Step 2: Brightest Cluster Fallback ─────────────────────────
    hsv_pixels = cv2.countNonZero(hsv_mask)
    if hsv_pixels < total_area * 0.001:
        gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
        _, hsv_mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)

    # ── Step 3: Horizontal Joining ─────────────────────────────────
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    joined = cv2.dilate(hsv_mask, h_kernel, iterations=2)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    joined = cv2.morphologyEx(joined, cv2.MORPH_CLOSE, close_kernel)

    # ── Find all contours ──────────────────────────────────────────
    contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Prepare debug annotation image (copy of original with rectangles)
    debug_annotated = color_img.copy()

    # ── Step 4: Size Gating — "Big Lights Only" ────────────────────
    min_height = h * 0.05      # Contour must be ≥ 5% of image height
    min_width = w * 0.02       # Contour must be ≥ 2% of image width
    max_area = total_area * 0.50  # Reject panels > 50%

    hero_boxes = []       # Accepted large contours
    rejected_boxes = []   # Rejected small contours (for debug)

    for cnt in contours:
        x, y_pos, cw, ch_val = cv2.boundingRect(cnt)
        box_area = cw * ch_val

        # Reject panels covering > 50% of the image
        if box_area > max_area:
            rejected_boxes.append((x, y_pos, cw, ch_val))
            continue

        # SIZE GATE: reject if too short (indicator dots)
        if ch_val < min_height:
            rejected_boxes.append((x, y_pos, cw, ch_val))
            continue

        # SIZE GATE: reject thin vertical lines (artifacts between buttons)
        if cw < min_width:
            rejected_boxes.append((x, y_pos, cw, ch_val))
            continue

        # This contour is big enough — it's a candidate
        hero_boxes.append((x, y_pos, cw, ch_val))

    # Draw debug annotations: RED for rejected, will draw GREEN for hero later
    for (rx, ry, rw, rh) in rejected_boxes:
        cv2.rectangle(debug_annotated, (rx, ry), (rx + rw, ry + rh), (0, 0, 255), 2)

    if not hero_boxes:
        # No large contours survived — return original
        return image, debug_annotated

    # ── Step 5: Hero Cluster — union bounding box ──────────────────
    # If we have multiple large contours (e.g., "d" and "C" separately),
    # check if they're in the same horizontal band (likely same display).
    # Group by vertical proximity, then pick the topmost group (top-half bias).

    # Sort hero_boxes by y position (top to bottom)
    hero_boxes.sort(key=lambda b: b[1])

    # Group contours that are vertically close (within 2x the tallest height)
    groups = []
    current_group = [hero_boxes[0]]

    for i in range(1, len(hero_boxes)):
        prev = current_group[-1]
        curr = hero_boxes[i]
        prev_bottom = prev[1] + prev[3]
        gap = curr[1] - prev_bottom

        # If the gap between contours is small, they belong to the same display
        max_gap = max(prev[3], curr[3]) * 2
        if gap < max_gap:
            current_group.append(curr)
        else:
            groups.append(current_group)
            current_group = [curr]
    groups.append(current_group)

    # TOP-HALF BIAS: pick the group with the smallest average y (highest in image)
    best_group = min(groups, key=lambda g: sum(b[1] for b in g) / len(g))

    # Compute union bounding box of the best group
    x_min = min(b[0] for b in best_group)
    y_min = min(b[1] for b in best_group)
    x_max = max(b[0] + b[2] for b in best_group)
    y_max = max(b[1] + b[3] for b in best_group)

    # Draw GREEN rectangle for the hero cluster on debug image
    cv2.rectangle(debug_annotated, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)

    # ── Crop from original (preserving quality for OCR) ────────────
    crop_x1 = max(0, x_min - padding)
    crop_y1 = max(0, y_min - padding)
    crop_x2 = min(w, x_max + padding)
    crop_y2 = min(h, y_max + padding)

    cropped = image[crop_y1:crop_y2, crop_x1:crop_x2]

    return cropped, debug_annotated
