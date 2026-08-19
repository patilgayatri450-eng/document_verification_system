"""
pdf_utils.py
------------
Converts an uploaded PDF's first page into a PNG image so it can be
run through the same barcode-extraction + vision-model pipeline used
for JPG/PNG uploads. Uses PyMuPDF (no external Poppler install needed,
unlike pdf2image — important for Windows users).
"""

import pymupdf  # PyMuPDF
from pathlib import Path


def pdf_to_image(pdf_path: str, out_path: str, dpi: int = 200) -> str:
    """
    Renders the first page of the PDF at `pdf_path` to a PNG at `out_path`.
    Returns out_path. Raises ValueError if the PDF has no pages.
    """
    doc = pymupdf.open(pdf_path)
    if doc.page_count == 0:
        doc.close()
        raise ValueError("PDF has no pages.")

    page = doc.load_page(0)
    zoom = dpi / 72  # PDF default is 72 DPI
    matrix = pymupdf.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix)
    pix.save(out_path)
    doc.close()
    return out_path


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python pdf_utils.py <input.pdf> <output.png>")
    else:
        pdf_to_image(sys.argv[1], sys.argv[2])
        print(f"Saved {sys.argv[2]}")
