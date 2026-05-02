"""
Medical Image Processor Module
------------------------------
Mode 1 (Prototype): OpenCV-based anomaly detection using contour analysis.
Uses grayscale conversion, CLAHE enhancement, adaptive thresholding,
morphological operations, and contour detection to highlight suspicious regions.
"""

import cv2
import numpy as np


def highlight_region(path, target_size=(512, 512)):
    """
    Detect and highlight suspicious regions in X-ray/MRI images.

    Uses OpenCV pipeline:
        1. Resize → Grayscale → CLAHE contrast enhancement
        2. Gaussian blur → Adaptive threshold
        3. Morphological close/open to remove noise
        4. Contour detection with area filtering
        5. Region annotation with bounding boxes and labels

    Args:
        path: Path to the medical image file.
        target_size: Tuple (width, height) to resize image to.

    Returns:
        tuple: (annotated_image, region_count, region_details_list)
            - annotated_image: BGR numpy array with annotations
            - region_count: Number of detected suspicious regions
            - region_details_list: List of dicts with region metadata
    """
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Could not load image. Please check the file path and format.")

    img = cv2.resize(img, target_size)
    output = img.copy()  # Keep original for overlay blending
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blur = cv2.GaussianBlur(enhanced, (7, 7), 0)

    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 3
    )

    # Morphological operations to clean noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter by area
    min_area = 300
    max_area = (target_size[0] * target_size[1]) * 0.4
    suspicious_regions = []
    region_count = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            region_count += 1
            x, y, w, h = cv2.boundingRect(contour)

            # Semi-transparent overlay for detected region
            overlay = output.copy()
            cv2.drawContours(overlay, [contour], -1, (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.15, output, 0.85, 0, output)

            # Contour outline and bounding box
            cv2.drawContours(output, [contour], -1, (0, 100, 255), 2)
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(output, f"R{region_count}", (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            suspicious_regions.append({
                "id": region_count, "area": int(area),
                "location": f"({x}, {y})", "size": f"{w}x{h}px"
            })

    # Summary text on image
    cv2.putText(output, f"Detected: {region_count} region(s)", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return output, region_count, suspicious_regions