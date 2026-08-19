"""
ocr_fallback.py
----------------
Last-resort fallback for when a barcode's bars can't be decoded at all
(poor print quality, low-DPI source, damaged barcode, etc.) — but the
barcode's value is also printed as human-readable digits underneath it,
which is standard practice for most institutional/government barcodes.

This OCRs that printed number instead. It's a fallback specifically
because OCR on a single edge digit can occasionally misread a character
(a clipped '1' read as '4', etc.) — good enough to suggest a candidate,
not reliable enough to silently trust as ground truth. Always show the
result to the user for confirmation rather than auto-accepting it.
"""

import re
import cv2
import pytesseract


def ocr_extract_digit_candidates(image_path: str, top_fraction: float = 0.20) -> list[str]:
    """
    Looks for long digit sequences printed near the top of the page
    (where barcodes are conventionally placed) and returns them as
    candidate barcode values, longest/most-likely first.

    Returns an empty list (rather than raising) if Tesseract isn't
    installed/on PATH, or if OCR fails for any other reason — this is a
    best-effort fallback, not something that should break the whole
    upload pipeline when it's unavailable.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return []

        h, w = img.shape[:2]
        top = img[: int(h * top_fraction), :]
        padded = cv2.copyMakeBorder(top, 20, 20, 40, 40, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        gray = cv2.cvtColor(padded, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        config = "--psm 11 -c tessedit_char_whitelist=0123456789"
        text = pytesseract.image_to_string(binary, config=config)

        candidates = re.findall(r"\d{8,25}", text.replace(" ", ""))
        candidates.sort(key=len, reverse=True)
        return candidates
    except Exception as e:
        print(f"[ocr_fallback] OCR unavailable or failed, skipping: {e}")
        return []


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ocr_fallback.py <image_path>")
    else:
        print(ocr_extract_digit_candidates(sys.argv[1]))
