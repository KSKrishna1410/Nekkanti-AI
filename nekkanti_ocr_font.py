import os
import json
import numpy as np
from PIL import Image
import cv2
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from paddleocr import PaddleOCR
import fitz  # PyMuPDF
import pandas as pd  
import warnings

warnings.filterwarnings("ignore")


class NekkantiOCR:
    def __init__(self, output_dir="ocr_outputs_reconstructed"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.ocr = PaddleOCR(
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="PP-OCRv5_mobile_rec",
            use_doc_orientation_classify=True,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device="cpu"
        )

    def detect_lines(self, image_path):
        """Detect lines in the image using OpenCV."""
        # Read the image
        img = cv2.imread(image_path)
        if img is None:
            return []
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding to handle varying lighting conditions
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,  # Block size
            2    # Constant subtracted from mean
        )
        
        # Apply morphological operations to enhance lines
        kernel = np.ones((3,3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Detect lines using HoughLinesP with more sensitive parameters
        lines = cv2.HoughLinesP(
            binary,
            rho=1,
            theta=np.pi/180,
            threshold=50,  # Lower threshold to detect more lines
            minLineLength=50,  # Shorter minimum line length
            maxLineGap=20  # Larger gap allowed
        )
        
        if lines is None:
            return []
            
        return lines

    def draw_lines_in_pdf(self, c, lines, img_height):
        """Draw detected lines in the PDF."""
        # Calculate scaling factors
        scale_x = c._pagesize[0] / self.original_width
        scale_y = c._pagesize[1] / self.original_height
        
        # Vertical offset to move lines up (adjust this value as needed)
        vertical_offset = 20  # pixels
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Scale coordinates to match PDF dimensions
            x1_scaled = x1 * scale_x
            x2_scaled = x2 * scale_x
            y1_scaled = (y1 - vertical_offset) * scale_y  # Subtract offset to move up
            y2_scaled = (y2 - vertical_offset) * scale_y  # Subtract offset to move up
            
            # Invert y coordinates to match PDF coordinate system
            y1_pdf = img_height - y1_scaled
            y2_pdf = img_height - y2_scaled
            
            # Draw the line
            c.setStrokeColorRGB(0.7, 0.7, 0.7)  # Lighter gray color for lines
            c.setLineWidth(0.3)  # Thinner line
            c.line(x1_scaled, y1_pdf, x2_scaled, y2_pdf)

    def convert_ndarray(self, obj):
        """Recursively convert NumPy arrays to lists for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, list):
            return [self.convert_ndarray(i) for i in obj]
        if isinstance(obj, dict):
            return {k: self.convert_ndarray(v) for k, v in obj.items()}
        return obj

    def _is_pdf(self, file_path):
        """Check if the input file is a PDF."""
        return file_path.lower().endswith('.pdf')
    
    def _is_scanned_pdf(self, pdf_path):
        """Check if PDF is scanned (no text content)."""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            # If very little text, likely scanned
            return len(text.strip()) < 100
        except:
            return True
    
    def _pdf_to_images(self, pdf_path):
        """Convert PDF pages to temporary image files."""
        doc = fitz.open(pdf_path)
        temp_image_paths = []
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            # Convert page to image (with high DPI for better OCR)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x scaling for better quality
            
            # Save as temporary image
            temp_image_path = os.path.join(self.output_dir, f"{base_name}_temp_page{page_num + 1}.png")
            pix.save(temp_image_path)
            temp_image_paths.append(temp_image_path)
            
        doc.close()
        return temp_image_paths
    
    def _cleanup_temp_files(self, temp_files):
        """Remove temporary files."""
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                print(f"   Warning: Could not remove temp file {temp_file}: {e}")

    def ocr_and_reconstruct(self, input_path):
        temp_image_paths = []
        
        try:
            # Check if input is PDF or image
            if self._is_pdf(input_path):
                print(f"📄 Converting PDF to images for OCR processing...")
                temp_image_paths = self._pdf_to_images(input_path)
                
                if not temp_image_paths:
                    raise ValueError(f"Could not convert PDF pages to images: {input_path}")
                
                # Use first page for getting dimensions (assuming all pages have similar dimensions)
                primary_image_path = temp_image_paths[0]
            else:
                # It's an image file
                primary_image_path = input_path
            
            # Get original image dimensions
            img = cv2.imread(primary_image_path)
            if img is None:
                raise ValueError(f"Could not read image at {primary_image_path}")
            self.original_height = img.shape[0]
            self.original_width = img.shape[1]

            # Run OCR on the input (PaddleOCR can handle both images and PDFs)
            result = self.ocr.ocr(input_path)
            ocr_data = [self.convert_ndarray(dict(res)) for res in result]

            base_name = os.path.splitext(os.path.basename(input_path))[0]
            pdf_path = os.path.join(self.output_dir, f"{base_name}_reconstructed.pdf")

            c = None  # Canvas will be created on the first page

            for i, page_result in enumerate(ocr_data):
                rec_texts = page_result["rec_texts"]
                rec_polys = page_result["rec_polys"]

                if not rec_polys:
                    continue

                # Use original image dimensions for PDF size
                img_width, img_height = self.original_width, self.original_height

                if c is None:
                    c = canvas.Canvas(pdf_path, pagesize=(img_width, img_height))
                else:
                    c.showPage()
                    c.setPageSize((img_width, img_height))

                # Detect and draw lines (use appropriate image for line detection)
                if self._is_pdf(input_path) and i < len(temp_image_paths):
                    lines = self.detect_lines(temp_image_paths[i])
                else:
                    lines = self.detect_lines(input_path)
                    
                self.draw_lines_in_pdf(c, lines, img_height)

                def invert_y(y):
                    return img_height - y

                for text, poly in zip(rec_texts, rec_polys):
                    x = min(p[0] for p in poly)
                    y = min(p[1] for p in poly)
                    y_pdf = invert_y(y)
                    
                    # Add horizontal offset (adjust this value as needed)
                    horizontal_offset = 10  # pixels
                    x = x + horizontal_offset
                    
                    # Calculate the bounding box dimensions
                    height = max(abs(p[1] - poly[0][1]) for p in poly)
                    width = max(abs(p[0] - poly[0][0]) for p in poly)
                    
                    # Calculate average character width (assuming average character is about 60% of height)
                    avg_char_width = height * 0.6
                    
                    # Estimate number of characters that should fit in the width
                    num_chars = max(1, width / avg_char_width)
                    
                    # Calculate font size based on height with a small margin
                    height_based_size = int(height * 0.85)  # 85% of height to leave some margin
                    
                    # Calculate font size based on width and number of characters
                    width_based_size = int(width / num_chars * 0.85)  # 85% of width per character
                    
                    # Use the minimum of both sizes to ensure text fits
                    font_size = max(min(height_based_size, width_based_size), 6)
                    
                    # Set font and draw text
                    c.setFont("Helvetica", font_size)
                    c.drawString(x, y_pdf, text)

            if c:
                c.save()

            return ocr_data, pdf_path
            
        finally:
            # Clean up temporary files
            if temp_image_paths:
                self._cleanup_temp_files(temp_image_paths)

    def extract_tables_from_pdf(self, pdf_path=None, save_to_excel=True):
        """
        Extract tables from a PDF using PyMuPDF.
        
        Args:
            pdf_path (str, optional): Path to the PDF file. If None, will try to use the last reconstructed PDF.
            save_to_excel (bool): Whether to save tables to Excel files.
            
        Returns:
            list: List of dictionaries containing table data and metadata.
        """
        if pdf_path is None:
            raise ValueError("PDF path must be provided")
            
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        # Open the PDF document
        doc = fitz.open(pdf_path)
        
        all_tables = []
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        
        print(f"📄 Processing PDF: {pdf_path}")
        print(f"📖 Total pages: {len(doc)}")
        
        # Process each page
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Find tables on this page
            tabs = page.find_tables()
            
            if tabs.tables:
                print(f"📊 Found {len(tabs.tables)} table(s) on page {page_num + 1}")
                
                for table_idx, tab in enumerate(tabs.tables):
                    try:
                        # Convert table to pandas DataFrame
                        df = tab.to_pandas()
                        
                        if df.empty:
                            print(f"   Table {table_idx + 1} on page {page_num + 1} is empty")
                            continue
                            
                        # Create table metadata
                        table_data = {
                            'page_number': page_num + 1,
                            'table_index': table_idx + 1,
                            'dataframe': df,
                            'shape': df.shape,
                            'columns': df.columns.tolist(),
                        }
                        
                        all_tables.append(table_data)
                        
                        # Save to Excel if requested
                        if save_to_excel:
                            excel_filename = f"{base_name}_page{page_num + 1}_table{table_idx + 1}.xlsx"
                            excel_path = os.path.join(self.output_dir, excel_filename)
                            df.to_excel(excel_path, index=False)
                            table_data['excel_path'] = excel_path
                            print(f"💾 Table saved to: {excel_path}")
                            
                        print(f"✅ Table {table_idx + 1} extracted: {df.shape[0]} rows × {df.shape[1]} columns")
                        
                    except Exception as e:
                        print(f"❌ Error processing table {table_idx + 1} on page {page_num + 1}: {str(e)}")
                        continue
            else:
                print(f"❌ No tables found on page {page_num + 1}")
        
        # Close the document
        doc.close()
        
        if all_tables:
            print(f"🎉 Successfully extracted {len(all_tables)} table(s) from PDF")
        else:
            print("❌ No tables were extracted from the PDF")
            
        return all_tables
    
    def ocr_reconstruct_and_extract_tables(self, input_path, save_to_excel=True):
        """
        Complete pipeline: OCR the input, reconstruct PDF with lines, and extract tables.

        Args:
            input_path (str): Path to input image or PDF
            save_to_excel (bool): Whether to save extracted tables to Excel files

        Returns:
            tuple: (ocr_data, pdf_path, extracted_tables)
        """
        print("🔄 Starting OCR and reconstruction...")

        # Step 1: OCR and reconstruct PDF
        ocr_data, pdf_path = self.ocr_and_reconstruct(input_path)

        print("🔍 Extracting tables from reconstructed PDF...")

        # Step 2: Extract tables from the reconstructed PDF
        extracted_tables = self.extract_tables_from_pdf(pdf_path, save_to_excel)

        return ocr_data, pdf_path, extracted_tables

    def extract_text_with_page_info(self, ocr_results):
        """
        Extract text with page information from OCR results for better LLM context.

        Args:
            ocr_results (list): OCR results from ocr_and_reconstruct method

        Returns:
            str: Extracted text with page separators
        """
        if not ocr_results:
            return ""

        extracted_text_parts = []

        for page_idx, page_result in enumerate(ocr_results):
            if not page_result or 'rec_texts' not in page_result:
                continue

            rec_texts = page_result['rec_texts']

            if not rec_texts:
                continue

            # Add page separator for multi-page documents
            if page_idx > 0:
                extracted_text_parts.append(f"\n=== PAGE {page_idx + 1} ===\n")

            # Join all text from this page
            page_text = " ".join(rec_texts)
            extracted_text_parts.append(page_text)

        return "".join(extracted_text_parts)

    def extract_text_per_page(self, ocr_results):
        """
        Extract text per page from OCR results.

        Args:
            ocr_results (list): OCR results from ocr_and_reconstruct method

        Returns:
            list: A list of strings, where each string is the text of a page.
        """
        if not ocr_results:
            return []

        page_texts = []
        for page_result in ocr_results:
            if not page_result or 'rec_texts' not in page_result:
                page_texts.append("")
                continue

            rec_texts = page_result['rec_texts']
            page_text = " ".join(rec_texts)
            page_texts.append(page_text)

        return page_texts


def is_readable_pdf(file_path):
    """Check if PDF contains readable text (not scanned)."""
    try:
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return len(text.strip()) > 100  # If more than 100 characters, likely readable
    except:
        return False


def convert_to_readable_pdf(input_path, output_dir):
    """Convert any input to a readable PDF for better table extraction."""
    processor = NekkantiOCR(output_dir)
    ocr_data, pdf_path = processor.ocr_and_reconstruct(input_path)
    return pdf_path


if __name__ == "__main__":
    # Example usage
    input_path = "test_document.png"  # or .pdf
    processor = NekkantiOCR()
    
    # Use the complete pipeline: OCR, reconstruct PDF, and extract tables
    ocr_results, pdf_path, extracted_tables = processor.ocr_reconstruct_and_extract_tables(input_path)

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    json_path = os.path.join(processor.output_dir, f"{base_name}_res.json")
    with open(json_path, "w") as f:
        json.dump(ocr_results, f, indent=2)

    print(f"✅ JSON saved to: {json_path}")
    print(f"✅ Reconstructed PDF saved to: {pdf_path}")
    
    # Display information about extracted tables
    if extracted_tables:
        print(f"\n📊 Table Extraction Summary:")
        print(f"   Total tables extracted: {len(extracted_tables)}")
        for i, table_info in enumerate(extracted_tables):
            print(f"   Table {i+1}:")
            print(f"     - Page: {table_info['page_number']}")
            print(f"     - Size: {table_info['shape'][0]} rows × {table_info['shape'][1]} columns")
            print(f"     - Columns: {table_info['columns']}")
            if 'excel_path' in table_info:
                print(f"     - Excel file: {table_info['excel_path']}")
    else:
        print("\n❌ No tables were extracted from the document")
