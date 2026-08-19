"""
doc_type_classifier.py
------------------------
Recognizes what KIND of document was submitted (marksheet, certificate,
ID card, income certificate, domicile certificate, etc.) by OCRing the
page text and matching against keyword rules — independent of whether
the barcode matched anything in the institutional dataset.

This matters because doc_type used to only get set when the barcode
matched a record in data/valid_barcodes.csv — meaning any real-world
document (whose barcode isn't in that sample dataset) always showed
"—" for type, even though the document clearly IS a marksheet, a
certificate, etc. Classification here is independent of validity.
"""

import re
import cv2
import pytesseract

# Ordered rules: (doc_type_key, display_name, [keyword patterns]).
# Checked in order — first match wins, so put more specific types
# (e.g. income certificate) before generic ones.
RULES = [
    ("marksheet", "Marksheet / Statement of Marks",
        [r"statement of marks", r"marksheet", r"mark\s*sheet", r"grade\s*point", r"sgpa", r"cgpa"]),
    ("hsc_ssc_certificate", "HSC/SSC Certificate",
        [r"higher secondary", r"secondary education", r"h\.?s\.?c\.?\s*board", r"board of secondary"]),
    ("income_certificate", "Income Certificate",
        [r"income certificate", r"utpanna", r"annual income", r"\u0909\u0924\u094d\u092a\u0928\u094d\u0928"]),
    ("nationality_domicile", "Nationality / Domicile Certificate",
        [r"domicile", r"nationality", r"citizen of india", r"tehsildar"]),
    ("caste_certificate", "Caste Certificate",
        [r"caste certificate", r"scheduled tribe", r"scheduled caste", r"other backward class", r"\bobc\b"]),
    ("student_id", "Student ID Card",
        [r"student\s*id", r"identity\s*card", r"college\s*id"]),
    ("govt_id", "Government ID",
        [r"aadhaar", r"election commission", r"permanent account number", r"\bpan\b", r"passport"]),
    ("degree_certificate", "Degree Certificate",
        [r"bachelor of", r"master of", r"degree certificate", r"convocation"]),
]


def _ocr_full_text(image_path: str) -> str:
    try:
        img = cv2.imread(image_path)
        if img is None:
            return ""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray)
        return text.lower()
    except Exception as e:
        print(f"[doc_type_classifier] OCR unavailable or failed, skipping: {e}")
        return ""


def classify_document_type(image_path: str, original_filename: str | None = None) -> tuple[str, str] | tuple[None, None]:
    """
    Returns (doc_type_key, display_name) for the best-matching rule, or
    (None, None) if nothing matched.

    Checks the original filename first (cheap, and often very reliable —
    e.g. "Income_Certificate.pdf") before falling back to OCR'd page
    text. This matters especially for non-English documents: Tesseract's
    default English-only OCR garbles Marathi/Hindi text into meaningless
    output, so filename keywords can succeed where OCR text matching
    can't (short of installing additional language packs).
    """
    if original_filename:
        normalized = re.sub(r"[_\-\.]", " ", original_filename).lower()
        for doc_type_key, display_name, patterns in RULES:
            for pattern in patterns:
                if re.search(pattern, normalized, re.IGNORECASE):
                    return doc_type_key, display_name

    text = _ocr_full_text(image_path)
    if not text:
        return None, None

    for doc_type_key, display_name, patterns in RULES:
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return doc_type_key, display_name

    return None, None


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python doc_type_classifier.py <image_path> [original_filename]")
    else:
        fname = sys.argv[2] if len(sys.argv) > 2 else None
        print(classify_document_type(sys.argv[1], fname))
