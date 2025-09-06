#!/usr/bin/env python3
"""
Common document processing utilities for OCR and text extraction
"""

import os
import io
import logging
import tempfile
from typing import Tuple, List

import PyPDF2
from nekkanti_ocr_font import NekkantiOCR

logger = logging.getLogger(__name__)


class DocumentProcessor:
    def __init__(self):
        # Initialize OCR processor for scanned documents
        self.ocr_processor = None
        try:
            self.ocr_processor = NekkantiOCR()
            logger.info("✅ OCR processor initialized successfully")
        except Exception as e:
            logger.warning(f"⚠️ OCR processor initialization failed: {e}")
            logger.warning("Will only support readable PDFs")
    
    def extract_text_from_pdf(self, pdf_content: bytes, filename: str = "document.pdf") -> str:
        """Extract text content from PDF using PyPDF2 or OCR for scanned documents"""
        try:
            # First try extracting text directly from PDF
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            # Check if extracted text is meaningful (not a scanned PDF)
            if len(text.strip()) > 100:  # Readable PDF
                logger.info("✅ Readable PDF detected, using direct text extraction")
                return text.strip()
            else:
                # Scanned PDF - use OCR
                logger.info("📸 Scanned PDF detected, using OCR extraction")
                return self._extract_text_with_ocr(pdf_content, filename)
                
        except Exception as e:
            # Fallback to OCR if direct extraction fails
            logger.warning(f"Direct text extraction failed: {e}")
            logger.info("🔄 Falling back to OCR extraction")
            return self._extract_text_with_ocr(pdf_content, filename)
    
    def extract_text_from_image(self, image_content: bytes, filename: str) -> str:
        """Extract text from image using OCR"""
        return self._extract_text_with_ocr(image_content, filename)
    
    def _extract_text_with_ocr(self, content: bytes, filename: str) -> str:
        """Extract text using OCR for scanned documents and images"""
        if not self.ocr_processor:
            raise ValueError("OCR processor not available. Cannot process scanned documents.")
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
            temp_file.write(content)
            temp_file.flush()
            temp_path = temp_file.name
        
        try:
            # Use OCR to extract text
            ocr_results, reconstructed_pdf_path = self.ocr_processor.ocr_and_reconstruct(temp_path)
            
            # Extract text with page information for better LLM context
            extracted_text = self.ocr_processor.extract_text_with_page_info(ocr_results)
            
            if not extracted_text.strip():
                raise ValueError("No text could be extracted from the document")
            
            logger.info(f"✅ OCR extracted {len(text_parts)} text segments")
            return extracted_text
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"Could not clean up temp file: {e}")
    
    def is_supported_file_type(self, filename: str) -> Tuple[bool, str]:
        """Check if file type is supported and return file type"""
        allowed_extensions = ['.pdf', '.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
        file_ext = os.path.splitext(filename.lower())[1]
        
        if file_ext not in allowed_extensions:
            return False, ""
        
        file_type = "Image" if file_ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'] else "PDF"
        return True, file_type