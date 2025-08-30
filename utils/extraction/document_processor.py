"""
Document processing utilities for OCR and table extraction
"""

import os
import sys
import shutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pathlib import Path
from utils.ocr.document_ocr import DocumentOCR
from utils.extraction.invoice_table_extractor import InvoiceTableExtractor

class DocumentProcessor:
    """
    Complete document processing pipeline that:
    1. Checks if document is readable
    2. Converts to readable PDF if needed using OCR
    3. Extracts tables from the readable PDF
    """
    
    def __init__(self, output_dir="outputs"):
        """
        Initialize the document processor.
        
        Args:
            output_dir (str): Base directory for all outputs
        """
        self.output_dir = output_dir
        
        # Create output subdirectories
        self.readable_pdfs_dir = os.path.join(output_dir, "readable_pdfs")
        self.tables_dir = os.path.join(output_dir, "tables")
        self.temp_dir = os.path.join(output_dir, "temp")
        
        for dir_path in [self.readable_pdfs_dir, self.tables_dir, self.temp_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Initialize processors
        self.ocr_processor = DocumentOCR(output_dir=self.temp_dir)
        self.table_extractor = InvoiceTableExtractor(
            output_dir=self.tables_dir,
            keywords_csv="data/master_csv/Invoice_allkeys.csv"
        )

    def process_document(self, input_path):
        """
        Process a document through the complete pipeline:
        1. Convert to readable PDF if needed
        2. Extract tables
        3. Save results
        
        Args:
            input_path (str): Path to input document (PDF or image)
            
        Returns:
            dict: Processing results including:
                - input_path: Original input path
                - readable_pdf_path: Path to readable version
                - tables: List of extracted tables
                - excel_path: Path to Excel file with tables
        """
        try:
            print(f"\n🔄 Processing document: {input_path}")
            
            # Step 1: Convert to readable PDF if needed
            conversion_result = self._convert_to_readable(input_path)
            readable_pdf_path = conversion_result['reconstructed_path']
            
            if not readable_pdf_path or not os.path.exists(readable_pdf_path):
                raise ValueError("Failed to get readable PDF")
            
            # Step 2: Extract tables
            print(f"\n📊 Extracting tables from readable PDF...")
            tables = self.table_extractor.extract_tables(
                readable_pdf_path,
                save_to_excel=True
            )
            
            # Get path to saved Excel file
            base_name = Path(input_path).stem
            excel_path = os.path.join(self.tables_dir, f"{base_name}_tables.xlsx")
            
            result = {
                'input_path': input_path,
                'readable_pdf_path': readable_pdf_path,
                'tables': tables,
                'excel_path': excel_path if os.path.exists(excel_path) else None
            }
            
            # Print summary
            print(f"\n✨ Processing Summary:")
            print(f"Input file: {result['input_path']}")
            print(f"Readable PDF: {result['readable_pdf_path']}")
            print(f"Tables extracted: {len(result['tables'])}")
            if result['excel_path']:
                print(f"Tables saved to: {result['excel_path']}")
            
            return result
            
        except Exception as e:
            print(f"❌ Error processing document: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            # Cleanup only temporary files
            self._cleanup_temp_files()

    def _convert_to_readable(self, input_path):
        """Convert document to readable PDF if needed."""
        # Check if input is PDF
        is_pdf = input_path.lower().endswith('.pdf')
        is_scanned = False
        
        if is_pdf:
            # Check if PDF is already readable
            import fitz
            doc = fitz.open(input_path)
            
            # Check first few pages for text content
            total_text_length = 0
            meaningful_words = 0
            pages_to_check = min(3, len(doc))
            
            for page_num in range(pages_to_check):
                text = doc[page_num].get_text().strip()
                total_text_length += len(text)
                
                # Count meaningful words (more than 2 characters)
                words = text.split()
                for word in words:
                    if len(word) > 2 and any(c.isalpha() for c in word):
                        meaningful_words += 1
            
            doc.close()
            
            # Determine if PDF needs OCR
            is_scanned = (total_text_length < 300 or meaningful_words < 30)
        else:
            # If not PDF, assume it's an image that needs OCR
            is_scanned = True
        
        if is_scanned:
            print(f"📄 Document needs OCR processing...")
            # Perform OCR and reconstruction
            ocr_data, temp_reconstructed_path = self.ocr_processor.ocr_and_reconstruct(input_path)
            
            # Copy the reconstructed PDF to final location
            if temp_reconstructed_path and os.path.exists(temp_reconstructed_path):
                base_name = Path(input_path).stem
                final_path = os.path.join(self.readable_pdfs_dir, f"{base_name}_readable.pdf")
                shutil.copy2(temp_reconstructed_path, final_path)
                reconstructed_path = final_path
                print(f"💾 Saved readable PDF to: {final_path}")
            else:
                reconstructed_path = None
        else:
            print(f"📄 Document is already readable - no OCR needed")
            # Copy the original PDF to output directory
            base_name = Path(input_path).stem
            final_path = os.path.join(self.readable_pdfs_dir, f"{base_name}_readable.pdf")
            shutil.copy2(input_path, final_path)
            ocr_data = None
            reconstructed_path = final_path
            print(f"💾 Copied readable PDF to: {final_path}")
            
        return {
            'input_path': input_path,
            'ocr_data': ocr_data,
            'reconstructed_path': reconstructed_path,
            'is_scanned': is_scanned
        }

    def _cleanup_temp_files(self):
        """Clean up temporary processing files but preserve output files."""
        try:
            # Only clean up the temp directory
            if os.path.exists(self.temp_dir):
                # Clean up files in temp directory
                for root, dirs, files in os.walk(self.temp_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        try:
                            os.unlink(file_path)
                        except Exception as e:
                            print(f"⚠️ Warning: Could not remove temp file {file_path}: {e}")
                
                # Recreate temp subdirectories
                os.makedirs(os.path.join(self.temp_dir, "ocr"), exist_ok=True)
                os.makedirs(os.path.join(self.temp_dir, "images"), exist_ok=True)
                os.makedirs(os.path.join(self.temp_dir, "pdfs"), exist_ok=True)
                
                print(f"🧹 Cleaned up temporary files")
        except Exception as e:
            print(f"⚠️ Warning during cleanup: {e}")

def main():
    """Example usage of DocumentProcessor"""
    # Initialize processor
    processor = DocumentProcessor(output_dir="document_outputs")
    
    # Example document
    input_file = "/Users/saikrishnakompelly/Desktop/glbyte_bs/data/ocr_inputs/35 Invoices/6.1_13.05.2024 BS Enterprises-1.png"
    
    # Process the document
    result = processor.process_document(input_file)
    
    if result:
        # Print table preview
        print("\nTable Preview:")
        for idx, df in enumerate(result['tables']):
            print(f"\nTable {idx + 1}:")
            print(df.head())
            print("\nColumns:", list(df.columns))

if __name__ == "__main__":
    main() 