#!/usr/bin/env python3
"""
Bank statement processing module using non-AI approach for extracting structured data
"""

import os
import sys
import tempfile
import logging
from typing import Dict, Any, List, Tuple

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from bank_statements.extractors.header_extractor import BankStatementHeaderExtractor
from bank_statements.extractors.bank_statement_extractor import BankStatementExtractor
from utils.extraction.table_extractor import DocumentTableExtractor
from ..utils.ai_provider import TokenUsage

logger = logging.getLogger(__name__)


class BankStatementProcessor:
    def __init__(self):
        # Initialize the non-AI extractors
        self.output_dir = tempfile.mkdtemp(prefix="bankstmt_")
        self._temp_files_to_cleanup = []  # Track temp files for cleanup
        
        # Initialize extractors
        self.table_extractor = DocumentTableExtractor(
            output_dir=os.path.join(self.output_dir, "tables"),
            save_reconstructed_pdfs=True
        )
        
        self.header_extractor = BankStatementHeaderExtractor(
            output_dir=os.path.join(self.output_dir, "headers"),
            keywords_csv="data/master_csv/bankstmt_allkeys.csv",
            ifsc_master_csv="data/master_csv/IFSC_master.csv"
        )
        
        self.bank_statement_extractor = BankStatementExtractor(
            output_dir=self.output_dir
        )
        
        logger.info("✅ Bank Statement Processor initialized with non-AI approach")
    
    def cleanup(self):
        """Clean up temporary files and directories"""
        try:
            # Clean up tracked temporary files
            for temp_file in self._temp_files_to_cleanup:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                    logger.debug(f"🧹 Cleaned up temp file: {temp_file}")
            
            # Clean up output directory
            if os.path.exists(self.output_dir):
                import shutil
                shutil.rmtree(self.output_dir)
                logger.debug(f"🧹 Cleaned up temp directory: {self.output_dir}")
                
            self._temp_files_to_cleanup.clear()
        except Exception as e:
            logger.warning(f"⚠️ Cleanup failed: {e}")
    
    def process_bankstatement(self, file_content: bytes, filename: str) -> Tuple[Dict[str, Any], TokenUsage, List[str]]:
        """Process bank statement document using non-AI approach and extract structured data"""
        errors = []
        token_usage = TokenUsage()
        token_usage.provider = "non-ai"
        token_usage.model = "traditional-extraction"
        
        # Create temporary file
        temp_file = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
                temp_file.write(file_content)
                temp_file.flush()
                temp_file_path = temp_file.name
            
            # Check if it's a PDF or image
            is_pdf = filename.lower().endswith('.pdf')
            
            # Process the file using the non-AI approach from main_api.py logic
            result = self._process_file_non_ai(temp_file_path, is_pdf)
            
            if 'error' in result:
                errors.append(result['error'])
                return {}, token_usage, errors
            
            # Convert to the expected format for the modular API
            converted_data = self._convert_from_non_ai_result(result)
            
            return converted_data, token_usage, errors
                
        except Exception as e:
            errors.append(f"Bank statement processing error: {str(e)}")
            logger.error(f"Bank statement processing error: {e}")
            return {}, token_usage, errors
        
        finally:
            # Clean up temporary file
            if temp_file and os.path.exists(temp_file.name):
                try:
                    os.unlink(temp_file.name)
                except:
                    pass
            
            # Clean up all OCR reconstruction temporary files
            try:
                self.cleanup()
            except Exception as e:
                logger.warning(f"⚠️ Failed to cleanup temporary files: {e}")
    
    def _process_file_non_ai(self, file_path: str, is_pdf: bool) -> Dict[str, Any]:
        """Process file using non-AI approach - based on main_api.py logic"""
        try:
            if is_pdf:
                # Analyze the PDF
                analysis = self._analyze_pdf(file_path)
                raw_text = analysis.get('raw_text', '')
                
                # If PDF is not readable, convert it
                if not analysis['is_readable']:
                    ocr_data, readable_pdf = self.table_extractor.ocr_processor.ocr_and_reconstruct(file_path)
                    if readable_pdf and os.path.exists(readable_pdf):
                        file_path = readable_pdf
                        # Track this temporary file for cleanup
                        self._temp_files_to_cleanup.append(readable_pdf)
                        # Extract text from OCR data
                        raw_text = ""
                        if ocr_data:
                            for page_result in ocr_data:
                                if page_result and 'rec_texts' in page_result:
                                    raw_text += " ".join(page_result['rec_texts']) + "\n"
                    else:
                        return {'error': 'Failed to convert scanned PDF'}
            else:
                # For images, directly use OCR
                ocr_data, readable_pdf = self.table_extractor.ocr_processor.ocr_and_reconstruct(file_path)
                if readable_pdf and os.path.exists(readable_pdf):
                    file_path = readable_pdf
                    # Track this temporary file for cleanup
                    self._temp_files_to_cleanup.append(readable_pdf)
                    # Extract text from OCR data
                    raw_text = ""
                    if ocr_data:
                        for page_result in ocr_data:
                            if page_result and 'rec_texts' in page_result:
                                raw_text += " ".join(page_result['rec_texts']) + "\n"
                else:
                    return {'error': 'Failed to process image'}
                
                analysis = {
                    'is_readable': True,
                    'page_count': 1,
                    'raw_text': raw_text
                }
            
            # Extract headers and tables using non-AI approach
            headers = self.header_extractor.extract_headers(file_path)
            table_df = self.bank_statement_extractor.extract_bank_statement_table(file_path)
            
            # Convert DataFrame to list of lists if not empty
            tables = []
            if table_df is not None and not table_df.empty:
                tables = [table_df.columns.tolist()]
                tables.extend(table_df.values.tolist())
                # Convert to string and handle NaN values
                import pandas as pd
                tables = [['' if pd.isna(cell) else str(cell) for cell in row] for row in tables]
                # Fix merged transaction issues
                tables = self._fix_merged_transactions(tables)
            
            # Convert headers to list format
            header_list = []
            for header_item in headers:
                if isinstance(header_item, dict):
                    header_list.append(header_item)
            
            # Create table info structure
            table_info = []
            if tables and len(tables) > 0:
                columns = tables[0]
                for idx, col in enumerate(columns):
                    table_info.append({
                        "key": col,
                        "position": idx + 1,
                        "coordinates": [idx * 100, (idx + 1) * 100]
                    })
            
            # Return in the format expected by the main API
            return {
                'raw_text': raw_text,
                'headers': header_list,
                'tables': tables,
                'table_info': table_info,
                'page_count': analysis.get('page_count', 1)
            }
            
        except Exception as e:
            logger.error(f"Error in non-AI processing: {e}")
            return {'error': str(e)}
    
    def _analyze_pdf(self, file_path: str) -> Dict[str, Any]:
        """Analyze PDF to check if it's readable - copied from main_api.py"""
        try:
            import fitz
            doc = fitz.open(file_path)
            page_count = len(doc)
            total_chars = 0
            total_words = 0
            pages_with_text = 0
            raw_text = ""
            
            for page in doc:
                text = page.get_text()
                raw_text += text + "\n"
                if text.strip():
                    pages_with_text += 1
                    words = text.split()
                    total_words += len(words)
                    total_chars += len(text)
            
            doc.close()
            
            chars_per_page = total_chars / page_count if page_count > 0 else 0
            avg_word_length = total_chars / total_words if total_words > 0 else 0
            text_page_ratio = pages_with_text / page_count if page_count > 0 else 0
            
            is_readable = (chars_per_page > 100 and
                         text_page_ratio > 0.5 and
                         avg_word_length > 3)
            
            return {
                'is_readable': is_readable,
                'page_count': page_count,
                'total_chars': total_chars,
                'total_words': total_words,
                'chars_per_page': chars_per_page,
                'avg_word_length': avg_word_length,
                'text_page_ratio': text_page_ratio,
                'raw_text': raw_text
            }
            
        except Exception as e:
            logger.error(f"Error analyzing PDF: {str(e)}")
            return {
                'is_readable': False,
                'error': str(e),
                'raw_text': ""
            }
    
    def _fix_merged_transactions(self, tables: List[List[str]]) -> List[List[str]]:
        """Fix merged transactions in table data - copied from main_api.py"""
        if not tables or len(tables) < 2:
            return tables
        
        import re
        fixed_tables = []
        header_row = tables[0]
        fixed_tables.append(header_row)
        
        for row in tables[1:]:  # Skip header row
            if len(row) < 2:  # Skip malformed rows
                continue
                
            # Check if the description field contains multiple transactions
            description = row[1] if len(row) > 1 else ""
            
            # Look for UPI transaction patterns that might be merged
            upi_pattern = r'(UPI/P2[AM]/\d+/[^/]+/[^/]+/[^/]+(?:/[^/]+)*)'
            matches = re.findall(upi_pattern, description)
            
            if len(matches) > 1:
                # Multiple UPI transactions found - split them
                logger.info(f"🔧 Fixing merged transactions: found {len(matches)} transactions")
                
                # For now, just take the first transaction to avoid data corruption
                first_transaction = matches[0]
                row[1] = first_transaction
                logger.info(f"   ✅ Kept first transaction: {first_transaction}")
            
            fixed_tables.append(row)
        
        return fixed_tables
    
    def _convert_from_non_ai_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Convert non-AI result to the format expected by the modular API"""
        # For bank statements, we return simple structure since convert_to_target_format handles the complex formatting
        return {
            'raw_text': result.get('raw_text', ''),
            'headers': result.get('headers', []),
            'tables': result.get('tables', []),
            'table_info': result.get('table_info', []),
            'page_count': result.get('page_count', 1)
        }
    
    def convert_to_target_format(self, extracted_data: Dict[str, Any], process_id: str, filename: str, file_path: str) -> Dict[str, Any]:
        """Convert extracted data to the exact target JSON format matching response_1756545663632.json"""
        headers = extracted_data.get('headers', [])
        tables = extracted_data.get('tables', [])
        raw_text = extracted_data.get('raw_text', '')
        
        # Create the response matching the exact structure from response_1756545663632.json
        return {
            "page": 1,
            "identified_doc_type": "BANKSTMT",
            "rawtext": raw_text,
            "headerInfo": headers,  # These already come in the correct format from the extractor
            "paymentSts": None,
            "incl_Tax": None,
            "lineInfo": {
                "lineData": tables,  # Tables already in correct format
                "tableInfo": extracted_data.get('table_info', []),
                "excludeLine": [],
                "tablePosition": [[0, 529], [0, None]]
            },
            "pageWiseFilePath": f"/files/inHouseOCR/{process_id}/page1",
            "pageWisedocPath": f"/files/inHouseOCR/{process_id}/page1/{filename}"
        }