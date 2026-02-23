"""
Batch Evaluator for the OCR Pipeline.
Benchmarks accuracy across the full dataset of washing machine display images,
using error_code_mapping.json as the absolute source of truth.

Usage:
    python eval/batch_evaluate.py --input "data/washing_machine/Washing Machines"
"""

import sys
import os
import re
import argparse
import logging
import difflib
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ── Project path setup ──────────────────────────────────────────────────────
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import settings
from src import pipeline
from utils import io_utils

# ── Logging (suppress noisy pipeline logs during batch run) ─────────────────
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# ═══════════════════════════════════════════════════════════════════════════
#  1.  GROUND TRUTH EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

def extract_ground_truth(filename: str) -> Optional[str]:
    """
    Extract the expected error code from an image filename.

    Convention:  <ErrorCode> [<variant_number>].<ext>
    Examples:
        dC 1.jpg   →  dC
        E2 3.jpg   →  E2
        nF.jpg     →  nF
        phto1.png  →  None  (skipped – not a labelled image)

    Returns:
        The ground-truth error code string, or None if unparseable.
    """
    stem = Path(filename).stem  # e.g. "dC 1" or "phto1"

    # Skip files whose stem starts with non-error-code-like patterns
    # (e.g. 'phto1', 'photo_2', 'test_image', etc.)
    if re.match(r"^(phto|photo|test|sample|demo)", stem, re.IGNORECASE):
        return None

    # Strip parenthesized variants: "SUD (2)" → "SUD"
    label = re.sub(r"\s*\(\d+\)$", "", stem)
    # Strip trailing variant number: "dC 1" → "dC", "E2 3" → "E2"
    label = re.sub(r"\s+\d+$", "", label).strip()
    if not label:
        return None

    return label


# ═══════════════════════════════════════════════════════════════════════════
#  2.  THREE-STEP JSON VALIDATION POST-PROCESSOR
# ═══════════════════════════════════════════════════════════════════════════

# 7-segment character confusion table (OCR-read → likely intended)
SEVEN_SEG_SUBSTITUTIONS: Dict[str, str] = {
    "S": "5",
    "5": "S",
    "O": "0",
    "0": "O",
    "Z": "2",
    "2": "Z",
    "B": "8",
    "8": "B",
    "G": "6",
    "6": "G",
    "I": "1",
    "1": "I",
    "l": "1",
    "D": "0",
    "q": "9",
    "g": "9",
    "o": "d",   # 7-seg 'd' misread as 'o'
    "a": "C",   # bottom-open 'a' ↔ 'C'
    "j": "3",   # tail of 'j' ↔ '3'
    "r": "2",   # '2' misread as 'r'
    "i": "E",   # 'E' misread as 'i'
}


def _build_case_map(valid_codes: Set[str]) -> Dict[str, str]:
    """Build a lowercase → original-case lookup for the valid codes."""
    return {code.lower(): code for code in valid_codes}


def _try_substitutions(
    text: str, valid_codes_lower: Dict[str, str]
) -> Optional[str]:
    """
    Step B: Try every single-character substitution from the 7-segment
    confusion table.  Return the first valid code found, or None.
    """
    chars = list(text)
    for i, ch in enumerate(chars):
        if ch in SEVEN_SEG_SUBSTITUTIONS:
            swapped = chars.copy()
            swapped[i] = SEVEN_SEG_SUBSTITUTIONS[ch]
            candidate = "".join(swapped)
            if candidate.lower() in valid_codes_lower:
                return valid_codes_lower[candidate.lower()]
    # Also try multiple simultaneous substitutions (full sweep)
    full_swapped = "".join(SEVEN_SEG_SUBSTITUTIONS.get(c, c) for c in text)
    if full_swapped.lower() in valid_codes_lower:
        return valid_codes_lower[full_swapped.lower()]
    return None


def _fuzzy_match_with_length_penalty(
    text: str, valid_codes: List[str], threshold: float = 0.5
) -> Optional[str]:
    """
    Step C: Fuzzy match using difflib, with a penalty for length differences.
    Returns the best matching code, or None if below threshold.
    """
    if not text:
        return None

    best_match = None
    best_score = 0.0
    text_lower = text.lower()

    for code in valid_codes:
        code_lower = code.lower()
        # Base similarity
        base_score = difflib.SequenceMatcher(None, text_lower, code_lower).ratio()

        # Length penalty: penalise heavily when lengths differ
        len_diff = abs(len(text) - len(code))
        length_penalty = 1.0 / (1.0 + len_diff * 0.5)  # each char diff → −33%

        score = base_score * length_penalty
        if score > best_score:
            best_score = score
            best_match = code

    if best_score >= threshold and best_match is not None:
        return best_match
    return None


def validate_against_json(
    raw_text: str, valid_codes: Set[str], valid_codes_list: List[str]
) -> Tuple[str, str]:
    """
    Force the raw OCR text to conform to the JSON source of truth.

    Returns:
        (final_prediction, step_used)
        step_used is one of: "DIRECT", "SUBSTITUTION", "FUZZY", "NONE"
    """
    if not raw_text or not raw_text.strip():
        return "", "NONE"

    # Clean: strip whitespace / special chars (keep alnum only)
    cleaned = re.sub(r"[^a-zA-Z0-9]", "", raw_text).strip()
    if not cleaned:
        return "", "NONE"

    case_map = _build_case_map(valid_codes)

    # ── Step A: Direct Match (case-insensitive) ──────────────────────────
    if cleaned.lower() in case_map:
        return case_map[cleaned.lower()], "DIRECT"

    # ── Step B: 7-segment Substitution ───────────────────────────────────
    sub_result = _try_substitutions(cleaned, case_map)
    if sub_result is not None:
        return sub_result, "SUBSTITUTION"

    # ── Step C: Fuzzy Match with length penalty ──────────────────────────
    fuzzy_result = _fuzzy_match_with_length_penalty(
        cleaned, valid_codes_list, threshold=0.5
    )
    if fuzzy_result is not None:
        return fuzzy_result, "FUZZY"

    return cleaned, "NONE"


# ═══════════════════════════════════════════════════════════════════════════
#  3.  EVALUATION LOOP
# ═══════════════════════════════════════════════════════════════════════════

def run_evaluation(input_dir: str) -> None:
    """Run the full batch evaluation and print the report."""
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"❌  Input directory not found: {input_path}")
        return

    # ── Load source of truth ─────────────────────────────────────────────
    mapping_path = settings.ERROR_CODES_PATH.parent / "error_code_mapping.json"
    error_mapping: Dict = io_utils.load_json(mapping_path)
    valid_codes: Set[str] = set(error_mapping.keys())
    valid_codes_list: List[str] = list(error_mapping.keys())
    print(f"✅  Loaded {len(valid_codes)} valid error codes from JSON.\n")

    # ── Gather images ────────────────────────────────────────────────────
    image_files = sorted(
        f
        for f in input_path.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )

    # ── Counters & failure log ───────────────────────────────────────────
    total = 0
    correct = 0
    skipped = 0
    failures: List[Dict[str, str]] = []

    batch_start = time.perf_counter()

    for img_file in image_files:
        ground_truth = extract_ground_truth(img_file.name)
        if ground_truth is None:
            skipped += 1
            print(f"  ⏭  Skipped (no label): {img_file.name}")
            continue

        total += 1

        # Run pipeline
        try:
            result, metrics = pipeline.run_pipeline(str(img_file))
            raw_ocr = result.raw_text or ""
            pipeline_prediction = result.clean_text or ""
        except Exception as e:
            raw_ocr = f"ERROR: {e}"
            pipeline_prediction = ""

        # Run our independent 3-step validator
        final_prediction, step_used = validate_against_json(
            raw_ocr, valid_codes, valid_codes_list
        )

        # ── Compare (case-insensitive) ───────────────────────────────────
        is_correct = final_prediction.lower() == ground_truth.lower()
        if is_correct:
            correct += 1
            status = "✅"
        else:
            status = "❌"
            failures.append(
                {
                    "filename": img_file.name,
                    "ground_truth": ground_truth,
                    "raw_ocr": raw_ocr,
                    "pipeline_pred": pipeline_prediction,
                    "final_pred": final_prediction,
                    "step": step_used,
                }
            )

        print(
            f"  {status}  {img_file.name:<18s}  "
            f"GT={ground_truth:<6s}  Raw={raw_ocr:<8s}  "
            f"Final={final_prediction:<6s}  ({step_used})"
        )

    batch_time = time.perf_counter() - batch_start
    incorrect = total - correct

    # ═════════════════════════════════════════════════════════════════════
    #  4.  CONSOLE REPORT
    # ═════════════════════════════════════════════════════════════════════
    accuracy = (correct / total * 100) if total > 0 else 0.0

    print()
    print("═" * 72)
    print("            BATCH  EVALUATION  REPORT")
    print("═" * 72)
    print(f"  Overall Accuracy   : {accuracy:.2f}%")
    print(f"  Correct / Total    : {correct} / {total}")
    print(f"  Incorrect          : {incorrect}")
    print(f"  Skipped (no label) : {skipped}")
    print(f"  Total time         : {batch_time:.2f}s")
    print("─" * 72)

    if failures:
        print(f"  FAILURE LOG  ({len(failures)} errors)")
        print("─" * 72)
        header = (
            f"  {'#':>3s} │ {'Filename':<18s} │ {'Ground Truth':<12s} │ "
            f"{'Raw OCR':<12s} │ {'Pipeline':<12s} │ {'Final Pred':<12s} │ Step"
        )
        print(header)
        print(f"  {'─'*3:s}─┼─{'─'*18:s}─┼─{'─'*12:s}─┼─{'─'*12:s}─┼─{'─'*12:s}─┼─{'─'*12:s}─┼─{'─'*12:s}")
        for i, f in enumerate(failures, 1):
            print(
                f"  {i:3d} │ {f['filename']:<18s} │ {f['ground_truth']:<12s} │ "
                f"{f['raw_ocr']:<12s} │ {f['pipeline_pred']:<12s} │ {f['final_pred']:<12s} │ {f['step']}"
            )
        print("─" * 72)
    else:
        print("  🎉  PERFECT SCORE — No failures!")
        print("─" * 72)

    print("═" * 72)
    print()


# ═══════════════════════════════════════════════════════════════════════════
#  CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Batch-evaluate the OCR pipeline against labelled images."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(
            settings.BASE_DIR / "data" / "washing_machine" / "Washing Machines"
        ),
        help="Path to folder containing labelled test images.",
    )
    args = parser.parse_args()
    run_evaluation(args.input)


if __name__ == "__main__":
    main()
