"""
OCR Pipeline Module
-------------------
Extracts text from images and PDFs using Tesseract OCR with preprocessing.
Includes graceful fallback if Tesseract or Poppler are not installed.
"""

import cv2
import numpy as np
import re
import os
import logging

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────
# Dependency checks with graceful fallback
# ────────────────────────────────────────────────────────────────

TESSERACT_AVAILABLE = False
POPPLER_AVAILABLE = False

try:
    import pytesseract
    # Try common Windows paths
    tesseract_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        r"C:\Users\{}\AppData\Local\Tesseract-OCR\tesseract.exe".format(os.getenv("USERNAME", "")),
    ]
    for path in tesseract_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            break

    # Verify Tesseract is accessible
    pytesseract.get_tesseract_version()
    TESSERACT_AVAILABLE = True
    logger.info("✅ Tesseract OCR initialized successfully.")
except Exception as e:
    logger.warning(f"⚠️ Tesseract OCR not available: {e}")

try:
    from pdf2image import convert_from_path

    # Try common Poppler paths
    POPPLER_PATH = None
    poppler_candidates = [
        r"D:\poppler-25.12.0\Library\bin",
        r"C:\poppler\Library\bin",
        r"C:\Program Files\poppler\Library\bin",
    ]
    for path in poppler_candidates:
        if os.path.isdir(path):
            POPPLER_PATH = path
            break

    POPPLER_AVAILABLE = True
    logger.info("✅ pdf2image (Poppler) initialized successfully.")
except ImportError:
    logger.warning("⚠️ pdf2image not available. PDF OCR disabled.")


# ────────────────────────────────────────────────────────────────
# Image preprocessing for OCR
# ────────────────────────────────────────────────────────────────

def preprocess_image(img):
    """Apply preprocessing pipeline to improve OCR accuracy."""
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Increase contrast using CLAHE (better than simple scaling)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Noise removal with bilateral filter (preserves edges)
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)

    # Adaptive threshold for binarization
    thresh = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    # Deskew if needed
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) > 100:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if abs(angle) > 0.5:
            (h, w) = thresh.shape
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            thresh = cv2.warpAffine(thresh, M, (w, h),
                                     flags=cv2.INTER_CUBIC,
                                     borderMode=cv2.BORDER_REPLICATE)

    return thresh


# ────────────────────────────────────────────────────────────────
# Text cleaning
# ────────────────────────────────────────────────────────────────

def clean_text(text):
    """Clean and normalize OCR-extracted text for medical use."""
    if not text:
        return ""

    lines = text.split("\n")
    cleaned = []

    # Noise patterns to filter out
    noise_patterns = [
        r"^[\s\W]+$",           # Lines with only whitespace/symbols
        r"www\.",               # URLs
        r"fax",                 # Fax numbers
        r"tel:",                # Phone numbers
        r"page \d+",           # Page numbers
        r"^\d{1,2}/\d{1,2}/\d{2,4}$",  # Standalone dates
    ]

    for line in lines:
        line = line.strip()

        # Remove very short lines (noise)
        if len(line) < 3:
            continue

        # Check against noise patterns
        is_noise = False
        for pattern in noise_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                is_noise = True
                break

        if is_noise:
            continue

        # Fix common OCR mistakes
        line = line.replace("|", "l")
        line = line.replace("0rgan", "Organ")
        line = line.replace("1ung", "lung")
        line = line.replace("0pacity", "Opacity")
        line = line.replace("p1eural", "pleural")
        line = re.sub(r"\s{2,}", " ", line)  # Multiple spaces → single

        cleaned.append(line)

    return " ".join(cleaned)


# ────────────────────────────────────────────────────────────────
# Image OCR
# ────────────────────────────────────────────────────────────────

def extract_text_from_image(path):
    """
    Extract text from an image file using Tesseract OCR.

    Returns:
        str: Cleaned text, or error message if OCR unavailable.
    """
    if not TESSERACT_AVAILABLE:
        return "[OCR Error] Tesseract OCR is not installed or not found. Please install Tesseract OCR to use this feature."

    try:
        img = cv2.imread(path)
        if img is None:
            return "[OCR Error] Could not read the image file. Please upload a valid image."

        processed = preprocess_image(img)

        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(processed, config=custom_config)

        cleaned = clean_text(text)

        if not cleaned.strip():
            return "[OCR Warning] No readable text could be extracted from this image. The image may not contain text or may be too low quality."

        return cleaned

    except Exception as e:
        logger.error(f"OCR extraction failed: {e}")
        return f"[OCR Error] Text extraction failed: {str(e)}"


# ────────────────────────────────────────────────────────────────
# PDF OCR
# ────────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file by converting pages to images and running OCR.

    Returns:
        str: Cleaned concatenated text from all pages, or error message.
    """
    if not TESSERACT_AVAILABLE:
        return "[OCR Error] Tesseract OCR is not installed. PDF text extraction requires Tesseract."

    if not POPPLER_AVAILABLE:
        return "[OCR Error] Poppler is not installed. PDF processing requires poppler-utils."

    try:
        kwargs = {}
        if POPPLER_PATH:
            kwargs["poppler_path"] = POPPLER_PATH

        pages = convert_from_path(pdf_path, **kwargs)
        full_text = ""

        for i, page in enumerate(pages):
            # Convert PIL → OpenCV
            img = np.array(page)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            processed = preprocess_image(img)

            custom_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(processed, config=custom_config)

            full_text += clean_text(text) + "\n"

        result = full_text.strip()

        if not result:
            return "[OCR Warning] No readable text could be extracted from the PDF. It may be a scanned image with poor quality."

        return result

    except Exception as e:
        logger.error(f"PDF OCR extraction failed: {e}")
        return f"[OCR Error] PDF text extraction failed: {str(e)}"