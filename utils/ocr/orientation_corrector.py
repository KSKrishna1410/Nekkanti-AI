"""
Orientation detection and correction utilities for invoice documents
"""

import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
import re
import os
import logging

logger = logging.getLogger(__name__)


def detect_orientation_tesseract(img):
    """
    Detect page orientation using Tesseract OSD (Orientation & Script Detection).
    Returns the rotation angle (0, 90, 180, 270).
    """
    try:
        osd = pytesseract.image_to_osd(img)
        angle = int(re.search('Rotate: (\d+)', osd).group(1))
        return angle
    except Exception as e:
        logger.warning(f"Tesseract OSD failed: {e}")
        return 0


def correct_invoice_orientation_image(img, page_num=1):
    """
    Detect and correct orientation of a single invoice image
    using projection profiles + Tesseract OSD.
    """
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = 255 - binary  # text = white

        # Projection profiles
        horizontal_sum = np.sum(binary, axis=1)
        vertical_sum = np.sum(binary, axis=0)

        rotated = img
        orientation = "Upright"

        # Step 1: Projection check (sideways vs upright)
        if np.max(vertical_sum) > np.max(horizontal_sum):
            rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            orientation = "Rotated 90°"

        # Step 2: Use Tesseract OSD for final correction
        angle = detect_orientation_tesseract(rotated)
        if angle == 180:
            rotated = cv2.rotate(rotated, cv2.ROTATE_180)
            orientation = "Rotated 180°"
        elif angle == 90:
            rotated = cv2.rotate(rotated, cv2.ROTATE_90_CLOCKWISE)
            orientation = "Rotated 90°"
        elif angle == 270:
            rotated = cv2.rotate(rotated, cv2.ROTATE_90_COUNTERCLOCKWISE)
            orientation = "Rotated 270°"
        else:
            orientation = "Upright"

        logger.info(f"📄 Page {page_num}: {orientation}")
        return rotated
    
    except Exception as e:
        logger.warning(f"Orientation correction failed for page {page_num}: {e}")
        return img


def correct_pdf_orientation(pdf_path, output_path="corrected_invoice.pdf"):
    """
    Corrects orientation of each page in a PDF and saves a new PDF
    """
    try:
        # Convert PDF to list of PIL images
        pages = convert_from_path(pdf_path, dpi=300)

        corrected_images = []
        for i, page in enumerate(pages, start=1):
            try:
                # Convert PIL image -> OpenCV
                img_cv = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)

                # Correct orientation
                corrected = correct_invoice_orientation_image(img_cv, page_num=i)

                # Back to PIL for saving
                corrected_pil = Image.fromarray(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))
                corrected_images.append(corrected_pil)
            except Exception as e:
                logger.warning(f"Failed to correct page {i}: {e}, using original")
                corrected_images.append(page)

        # Save all corrected pages as single PDF
        if corrected_images:
            corrected_images[0].save(output_path, save_all=True, append_images=corrected_images[1:])
            logger.info(f"✅ Saved corrected PDF at: {output_path}")
            return output_path
        else:
            logger.error("No corrected images to save")
            return None
            
    except Exception as e:
        logger.error(f"PDF orientation correction failed: {e}")
        return None


def correct_image_orientation(image_path, output_path=None):
    """
    Correct orientation for a single image file
    """
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Could not read image: {image_path}")
            return None
        
        # Correct orientation
        corrected = correct_invoice_orientation_image(img, page_num=1)
        
        # Save corrected image
        if output_path is None:
            base, ext = os.path.splitext(image_path)
            output_path = f"{base}_corrected{ext}"
        
        success = cv2.imwrite(output_path, corrected)
        if success:
            logger.info(f"✅ Saved corrected image at: {output_path}")
            return output_path
        else:
            logger.error(f"Failed to save corrected image: {output_path}")
            return None
            
    except Exception as e:
        logger.error(f"Image orientation correction failed: {e}")
        return None