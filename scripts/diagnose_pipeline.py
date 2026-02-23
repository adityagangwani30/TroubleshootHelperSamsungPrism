"""Diagnostic: trace where text disappears at each pipeline stage."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import cv2
import numpy as np
from pathlib import Path

def diagnose_one(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        return {"file": image_path.name, "error": "load_failed"}

    h, w = img.shape[:2]
    total = h * w
    
    # HSV mask
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 10, 80]), np.array([180, 255, 255]))
    hsv_pct = cv2.countNonZero(mask) / total * 100
    
    fallback = False
    if cv2.countNonZero(mask) < total * 0.001:
        gray_fb = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_fb, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        hsv_pct = cv2.countNonZero(mask) / total * 100
        fallback = True
    
    # Contours
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    conn = cv2.dilate(mask, k, iterations=1)
    conn = cv2.morphologyEx(conn, cv2.MORPH_CLOSE, k)
    contours, _ = cv2.findContours(conn, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Box filtering
    boxes = []
    big_rej = 0
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        a = cw * ch
        if a > total * 0.5:
            big_rej += 1
            continue
        if a < total * 0.0001:
            continue
        boxes.append((x, y, cw, ch))
    
    # Height filter
    if boxes:
        max_h = max(b[3] for b in boxes)
        tall = [b for b in boxes if b[3] >= max_h * 0.5]
    else:
        max_h = 0
        tall = []
    
    return {
        "file": image_path.name,
        "size": f"{w}x{h}",
        "hsv_pct": f"{hsv_pct:.1f}",
        "fallback": str(fallback),
        "n_contours": str(len(contours)),
        "big_rejected": str(big_rej),
        "n_boxes": str(len(boxes)),
        "n_tall": str(len(tall)),
        "max_h": str(max_h),
    }

def main():
    data_dir = Path(__file__).parent.parent / "data" / "washing_machine" / "Washing Machines"
    out_path = Path(__file__).parent / "diag_results.txt"
    
    results = []
    for img_file in sorted(data_dir.glob("*")):
        if img_file.suffix.lower() in (".jpg", ".jpeg", ".png"):
            results.append(diagnose_one(img_file))
    
    with open(out_path, "w", encoding="utf-8") as f:
        # Header
        cols = ["file", "size", "hsv_pct", "fallback", "n_contours", "big_rejected", "n_boxes", "n_tall", "max_h"]
        f.write(" | ".join(cols) + "\n")
        f.write("-" * 120 + "\n")
        for r in results:
            row = [r.get(c, "?") for c in cols]
            f.write(" | ".join(row) + "\n")
    
    print(f"Wrote {len(results)} results to {out_path}")

if __name__ == "__main__":
    main()
