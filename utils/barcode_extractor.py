"""
barcode_extractor.py
---------------------
Detects and decodes barcodes / QR codes from a document image.

Uses OpenCV for preprocessing and pyzbar for decoding (supports
CODE128, EAN, QR, PDF417, etc.). If pyzbar is unavailable on the
system (missing zbar shared library), falls back to OpenCV's
built-in QRCodeDetector (QR only).

Real-world photos (phone camera shots, scanned documents) often fail
to decode on the raw image alone — blur, uneven lighting, low
resolution, or slight rotation are enough to break a scanner. This
module tries a series of preprocessing variants and rotations before
giving up, rather than a single attempt.
"""

import cv2
import numpy as np

try:
    from pyzbar.pyzbar import decode as zbar_decode
    PYZBAR_AVAILABLE = True
except Exception:
    PYZBAR_AVAILABLE = False


def _variants(image: np.ndarray):
    """Yields a series of preprocessed versions of the image to try decoding."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    yield image  # 1. raw, as-is
    yield gray  # 2. plain grayscale

    # 3. adaptive threshold (helps with uneven lighting / shadows)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    yield cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)

    # 4. Otsu threshold (different lighting profile than adaptive)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    yield otsu

    # 5. contrast-enhanced (CLAHE) — helps low-contrast phone photos
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    yield clahe.apply(gray)

    # 6. sharpened (helps slightly out-of-focus shots)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    yield cv2.filter2D(gray, -1, kernel)

    # 7. upscaled 2x (helps when the barcode occupies a small area of a large photo)
    h, w = gray.shape[:2]
    if max(h, w) < 2000:  # don't blow up already-large images
        yield cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)


def _rotations(img: np.ndarray, fine: bool = False):
    """
    Yields the image at 0/90/180/270 degrees — barcodes are often sideways
    in photos. If fine=True, also sweeps small tilt angles at each of
    those orientations, since 1D barcodes are very rotation-sensitive:
    even a 5-10 degree handheld-camera tilt is often enough to break
    decoding, well short of a full 90-degree turn.
    """
    coarse = [
        img,
        cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(img, cv2.ROTATE_180),
        cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE),
    ]
    for base in coarse:
        yield base

    if fine:
        h, w = img.shape[:2]
        diag = int(np.sqrt(h ** 2 + w ** 2))
        for base in coarse:
            bh, bw = base.shape[:2]
            for angle in (-15, -10, -5, 5, 10, 15, 20, -20):
                # pad to a square canvas first so rotation doesn't clip corners
                pad_h, pad_w = (diag - bh) // 2 + 1, (diag - bw) // 2 + 1
                padded = cv2.copyMakeBorder(base, pad_h, pad_h, pad_w, pad_w,
                                             cv2.BORDER_CONSTANT, value=255)
                ph, pw = padded.shape[:2]
                matrix = cv2.getRotationMatrix2D((pw / 2, ph / 2), angle, 1.0)
                rotated = cv2.warpAffine(padded, matrix, (pw, ph),
                                          borderValue=255)
                yield rotated


def extract_codes(image_path: str) -> list[dict]:
    """
    Returns a list of dicts: [{"type": "QRCODE", "data": "COL2023ID00124"}, ...]
    Two passes: a fast pass (coarse 90-degree rotations across preprocessing
    variants), then — only if that finds nothing — a slower fine-angle tilt
    sweep, since 1D barcodes from handheld phone photos are often just a
    few degrees off-axis rather than a clean 90-degree turn.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")

    if PYZBAR_AVAILABLE:
        # Pass 1: fast — coarse rotations only
        for variant in _variants(image):
            for rotated in _rotations(variant, fine=False):
                decoded = zbar_decode(rotated)
                if decoded:
                    return [
                        {"type": d.type, "data": d.data.decode("utf-8", errors="ignore")}
                        for d in decoded
                    ]

        # Pass 2: slower — fine tilt-angle sweep on the most useful variants
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        for variant in (gray, otsu):
            for rotated in _rotations(variant, fine=True):
                decoded = zbar_decode(rotated)
                if decoded:
                    return [
                        {"type": d.type, "data": d.data.decode("utf-8", errors="ignore")}
                        for d in decoded
                    ]
        return []
    else:
        # Fallback: QR-only via OpenCV, still try rotations
        detector = cv2.QRCodeDetector()
        for rotated in _rotations(image, fine=False):
            data, points, _ = detector.detectAndDecode(rotated)
            if data:
                return [{"type": "QRCODE", "data": data}]
        return []


def get_primary_code(image_path: str) -> str | None:
    """Convenience wrapper: returns the first decoded value, or None."""
    codes = extract_codes(image_path)
    return codes[0]["data"] if codes else None


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python barcode_extractor.py <image_path>")
    else:
        print(extract_codes(sys.argv[1]))
