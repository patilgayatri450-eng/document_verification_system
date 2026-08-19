"""
debug_barcode.py
-----------------
Standalone diagnostic for when a document's barcode isn't being
detected. Run this directly on the image that's failing to see
exactly what's happening, without going through the web app.

Usage:
    python debug_barcode.py path/to/your/document.jpg
"""

import sys
import cv2
from utils.barcode_extractor import extract_codes, PYZBAR_AVAILABLE
from utils.dataset_lookup import lookup_barcode


def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_barcode.py <image_path>")
        sys.exit(1)

    path = sys.argv[1]
    print(f"pyzbar available: {PYZBAR_AVAILABLE}")
    if not PYZBAR_AVAILABLE:
        print("!! pyzbar is not installed/working — only QR codes will be "
              "detected via the OpenCV fallback. Run: pip install pyzbar")

    img = cv2.imread(path)
    if img is None:
        print(f"Could not read image at '{path}' — check the path is correct.")
        sys.exit(1)

    h, w = img.shape[:2]
    print(f"Image loaded: {w}x{h} pixels")
    if max(h, w) < 400:
        print("!! Image resolution is quite low — barcodes may be too small to read.")

    codes = extract_codes(path)
    if not codes:
        print("\nRESULT: No barcode/QR code detected, even after trying multiple "
              "preprocessing variants and rotations.")
        print("Things to check:")
        print("  - Is the barcode actually in frame and not cropped/cut off?")
        print("  - Is it in focus? Try a sharper / closer photo.")
        print("  - Is there glare or a shadow across the barcode?")
        print("  - If it's a 1D barcode (lines, not a QR square), make sure it's "
              "roughly horizontal or vertical, not at a steep diagonal angle.")
        return

    print(f"\nRESULT: Detected {len(codes)} code(s):")
    for c in codes:
        print(f"  type={c['type']}  data='{c['data']}'")
        lookup = lookup_barcode(c["data"])
        print(f"  dataset lookup: {lookup}")


if __name__ == "__main__":
    main()
