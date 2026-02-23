"""
Post-processing module for OCR results.
Handles cleaning, formatting, and validation of the raw extracted text.
Uses data-driven regex generation and fuzzy matching for robust error code recognition.
"""

import re
import difflib
from typing import List, Tuple, Optional


# --- Match Type Constants ---
DIRECT_MATCH = "DIRECT_MATCH"
FUZZY_MATCH = "FUZZY_MATCH"
NO_MATCH = "NO_MATCH"


def build_dynamic_regex(allowed_codes: List[str]) -> str:
    """
    Analyze the keys in the error code mapping and dynamically build
    a regex pattern that matches their structure.

    Examines the character classes and lengths present in the allowed codes
    to construct the most permissive-yet-accurate pattern.

    Args:
        allowed_codes (List[str]): List of valid error code strings.

    Returns:
        str: A compiled-ready regex pattern string (e.g., "^[A-Za-z0-9]{1,4}$").
    """
    if not allowed_codes:
        return r"^[A-Za-z0-9]+$"

    min_len = min(len(code) for code in allowed_codes)
    max_len = max(len(code) for code in allowed_codes)

    # Determine which character classes are present
    has_upper = any(c.isupper() for code in allowed_codes for c in code)
    has_lower = any(c.islower() for code in allowed_codes for c in code)
    has_digit = any(c.isdigit() for code in allowed_codes for c in code)

    # Build character class
    char_class = ""
    if has_upper:
        char_class += "A-Z"
    if has_lower:
        char_class += "a-z"
    if has_digit:
        char_class += "0-9"

    if not char_class:
        char_class = "A-Za-z0-9"

    # Build quantifier
    if min_len == max_len:
        quantifier = f"{{{min_len}}}"
    else:
        quantifier = f"{{{min_len},{max_len}}}"

    return f"^[{char_class}]{quantifier}$"


def clean_text(raw_text: str, regex_pattern: Optional[str] = None) -> str:
    """
    Clean the raw text extracted by the OCR engine.
    Removes whitespace and special characters, then optionally validates
    against a dynamic regex pattern.

    Args:
        raw_text (str): The raw output from the OCR engine.
        regex_pattern (Optional[str]): Dynamic regex to filter noise.
            If provided, only text matching this pattern is kept.

    Returns:
        str: Cleaned text.
    """
    if not raw_text:
        return ""

    # Remove all whitespace (spaces, tabs, newlines)
    text = re.sub(r'\s+', '', raw_text)

    # Remove special characters but keep alphanumeric
    text = re.sub(r'[^a-zA-Z0-9]', '', text)

    # If a dynamic regex is provided, validate the cleaned text against it
    if regex_pattern and text:
        if not re.match(regex_pattern, text):
            # Text doesn't match expected pattern — could be noise
            # Try to extract a substring that matches
            for length in range(len(text), 0, -1):
                for start in range(len(text) - length + 1):
                    substring = text[start:start + length]
                    if re.match(regex_pattern, substring):
                        return substring
            # No valid substring found, return as-is for downstream handling
            return text

    return text


def correct_common_misreads(text: str) -> str:
    """
    Correct common OCR misinterpretations for 7-segment displays.
    """
    # 7-segment specific confusions
    corrections = {
        '2L': 'LE',  # Specific fix for observed issue
        '2l': 'LE',
        'Z': '2',
        'S': '5',
        'G': '6',
        'B': '8',
        'D': '0',
        'O': '0',
        'I': '1',
        'l': '1',
        'o': 'd',   # 7-seg 'd' misread as 'o'
        'a': 'C',   # bottom-open 'a' ↔ 'C'
        'j': '3',   # tail of 'j' ↔ '3'
        'r': '2',   # '2' misread as 'r'
        'i': 'E',   # 'E' misread as 'i'
    }

    # Check for direct mapping
    if text in corrections:
        return corrections[text]

    return text


def find_best_match(
    text: str,
    allowed_codes: List[str],
    threshold: float = 0.8
) -> Tuple[Optional[str], float, str]:
    """
    Find the best matching error code using fuzzy string matching.

    Uses difflib.SequenceMatcher to compute similarity ratios and returns
    the closest match above the given threshold.

    Args:
        text (str): The cleaned OCR text to match.
        allowed_codes (List[str]): List of valid error codes.
        threshold (float): Minimum similarity ratio (0.0–1.0) to accept
            a fuzzy match. Default is 0.8 (80%).

    Returns:
        Tuple[Optional[str], float, str]: A tuple of:
            - matched_code: The best matching code, or None if no match.
            - score: Similarity score (0.0–1.0).
            - match_type: One of DIRECT_MATCH, FUZZY_MATCH, or NO_MATCH.
    """
    if not text or not allowed_codes:
        return None, 0.0, NO_MATCH

    # --- Check 1: Direct exact match ---
    if text in allowed_codes:
        return text, 1.0, DIRECT_MATCH

    # --- Check 2: Case-insensitive direct match ---
    text_lower = text.lower()
    for code in allowed_codes:
        if code.lower() == text_lower:
            return code, 1.0, DIRECT_MATCH

    # --- Check 3: Fuzzy matching ---
    best_match = None
    best_score = 0.0

    for code in allowed_codes:
        # Compare case-insensitively for better fuzzy matching
        score = difflib.SequenceMatcher(
            None, text_lower, code.lower()
        ).ratio()

        if score > best_score:
            best_score = score
            best_match = code

    if best_score >= threshold and best_match is not None:
        return best_match, best_score, FUZZY_MATCH

    return None, best_score, NO_MATCH


def validate_error_code(
    text: str,
    allowed_codes: List[str],
    threshold: float = 0.8
) -> Tuple[bool, str, Optional[str]]:
    """
    Validate the extracted text against known error codes.
    Supports direct matching and fuzzy matching with configurable threshold.

    Args:
        text (str): The cleaned text.
        allowed_codes (List[str]): List of valid error codes.
        threshold (float): Fuzzy matching threshold (default 0.8).

    Returns:
        Tuple[bool, str, Optional[str]]: A tuple of:
            - is_valid: True if a match was found (direct or fuzzy).
            - match_type: DIRECT_MATCH, FUZZY_MATCH, or NO_MATCH.
            - matched_code: The matched code string, or None.
    """
    if not text:
        return False, NO_MATCH, None

    matched_code, score, match_type = find_best_match(text, allowed_codes, threshold)

    is_valid = match_type in (DIRECT_MATCH, FUZZY_MATCH)
    return is_valid, match_type, matched_code
