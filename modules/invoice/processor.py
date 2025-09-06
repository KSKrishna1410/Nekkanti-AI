#!/usr/bin/env python3
"""
Invoice processing module for extracting structured data from invoice documents
"""

import json
import logging
from typing import Dict, Any, List, Tuple

from ..utils.ai_provider import AIProviderClient, TokenUsage
from ..utils.document_processor import DocumentProcessor
from ...utils.prompt_manager import get_prompt_manager

logger = logging.getLogger(__name__)


class InvoiceProcessor:
    def __init__(self):
        self.ai_client = AIProviderClient()
        self.document_processor = DocumentProcessor()
        self.prompt_manager = get_prompt_manager()
    
    def process_invoice(self, file_content: bytes, filename: str) -> Tuple[Dict[str, Any], TokenUsage, List[str]]:
        """Process invoice document and extract structured data"""
        errors = []
        token_usage = TokenUsage()
        
        try:
            # Extract text from document
            if filename.lower().endswith('.pdf'):
                text = self.document_processor.extract_text_from_pdf(file_content, filename)
            else:
                text = self.document_processor.extract_text_from_image(file_content, filename)
            
            if not text:
                errors.append("No text could be extracted from document")
                return {}, token_usage, errors
            
            # Create extraction prompt using template
            prompt = self._create_invoice_extraction_prompt(text)
            
            # Generate AI response
            response_text, token_usage = self.ai_client.generate_response(prompt)
            
            if not response_text:
                errors.append("AI response was empty")
                return {}, token_usage, errors
            
            # Parse JSON response
            try:
                extracted_data = json.loads(response_text)
            except json.JSONDecodeError as e:
                errors.append(f"Failed to parse AI response as JSON: {e}")
                logger.error(f"Invalid JSON response: {response_text[:500]}...")
                return {}, token_usage, errors
            
            # Post-process to ensure serial numbers are present
            extracted_data = self._ensure_serial_numbers(extracted_data)
            
            # Post-process to validate tax data (no corrections in strict mode)
            extracted_data = self._post_process_tax_data(extracted_data)
            
            # Check extraction completeness and add warnings to errors
            completeness_warnings = self._check_extraction_completeness(extracted_data, text)
            if completeness_warnings:
                logger.warning(f"Extraction completeness warnings: {completeness_warnings}")
                errors.extend([f"Completeness warning: {warning}" for warning in completeness_warnings])
            
            return extracted_data, token_usage, errors
                
        except Exception as e:
            errors.append(f"Invoice processing error: {str(e)}")
            logger.error(f"Invoice processing error: {e}")
            return {}, token_usage, errors
    
    def _create_invoice_extraction_prompt(self, text: str) -> str:
        """Create a detailed prompt for AI to extract structured invoice data using template"""
        try:
            return self.prompt_manager.render_template('invoice_extraction.jinja', text=text)
        except Exception as e:
            logger.error(f"CRITICAL: Failed to load invoice extraction template: {e}")
            # Strict fallback - no creativity allowed
            return f"""
FINANCIAL DOCUMENT EXTRACTION - STRICT MODE

RULES: 
1. Extract ONLY what is explicitly written
2. Do NOT calculate or infer anything
3. Use null for missing fields
4. Copy text exactly as written

DOCUMENT TEXT:
{text}

Return JSON with exact structure: {{"header_fields": {{}}, "invoice_table": []}}
"""
    

    def _ensure_serial_numbers(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check for serial numbers but do NOT generate them - strict mode"""
        if not extracted_data or 'invoice_table' not in extracted_data:
            return extracted_data
        
        invoice_table = extracted_data['invoice_table']
        if not invoice_table:
            return extracted_data
        
        # In strict mode, we only validate but do NOT generate serial numbers
        # The LLM should extract only what exists in the document
        serial_fields = ['sl_no', 'serial_number', 'item_no', 's_no', 'serial']
        has_serials = any(
            any(str(item.get(field, '')).strip() for field in serial_fields)
            for item in invoice_table
        )
        
        if not has_serials:
            logger.info("No serial numbers found in extracted data - this is acceptable in strict mode")
        
        return extracted_data

    def _post_process_tax_data(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates GST data (IGST, CGST, SGST) in STRICT mode.
        - Enforces mutual exclusivity of IGST and CGST/SGST.
        - Does NOT perform calculations or corrections - only validates consistency.
        """
        logger.info("Running strict tax validation (no calculations)...")

        def _safe_float(value: Any) -> float:
            """Safely convert a value to a float, returning 0.0 on failure."""
            if value is None:
                return 0.0
            try:
                # Remove currency symbols, commas, percentages and other non-numeric chars
                cleaned_value = str(value).strip().replace('Rs.', '').replace('₹', '').replace(',', '').replace('%', '')
                if not cleaned_value:
                    return 0.0
                return float(cleaned_value)
            except (ValueError, TypeError):
                return 0.0

        def _process_item(item: Dict[str, Any]):
            """Process a single item (either header_fields or a line_item)."""
            if not isinstance(item, dict):
                return

            # Get tax values
            igst_amount = _safe_float(item.get('igst_amount'))
            cgst_amount = _safe_float(item.get('cgst_amount'))
            sgst_amount = _safe_float(item.get('sgst_amount'))
            igst_rate = _safe_float(item.get('igst_rate'))
            cgst_rate = _safe_float(item.get('cgst_rate'))
            sgst_rate = _safe_float(item.get('sgst_rate'))
            taxable_value = _safe_float(item.get('taxable_value') or item.get('taxable_amount'))

            # --- Rule 1: Mutual Exclusivity --- 
            if igst_amount > 0 and (cgst_amount > 0 or sgst_amount > 0):
                logger.warning(f"Conflict found: IGST ({igst_amount}) and CGST/SGST ({cgst_amount}/{sgst_amount}) present. Prioritizing IGST.")
                item['cgst_amount'] = None
                item['sgst_amount'] = None
                item['cgst_rate'] = None
                item['sgst_rate'] = None

            # --- Rule 2: STRICT MODE - No Calculations, Only Validation ---
            if taxable_value > 0:
                # Validate IGST (log inconsistencies but do NOT correct)
                if igst_rate > 0:
                    expected_igst = round(taxable_value * (igst_rate / 100), 2)
                    if abs(expected_igst - igst_amount) > 0.1: # Tolerance for rounding
                        logger.warning(f"IGST calculation inconsistency detected but NOT corrected. Document shows: {igst_amount}, Expected: {expected_igst}")
                
                # Validate CGST (log inconsistencies but do NOT correct)
                if cgst_rate > 0:
                    expected_cgst = round(taxable_value * (cgst_rate / 100), 2)
                    if abs(expected_cgst - cgst_amount) > 0.1:
                        logger.warning(f"CGST calculation inconsistency detected but NOT corrected. Document shows: {cgst_amount}, Expected: {expected_cgst}")

                # Validate SGST (log inconsistencies but do NOT correct)
                if sgst_rate > 0:
                    expected_sgst = round(taxable_value * (sgst_rate / 100), 2)
                    if abs(expected_sgst - sgst_amount) > 0.1:
                        logger.warning(f"SGST calculation inconsistency detected but NOT corrected. Document shows: {sgst_amount}, Expected: {expected_sgst}")

        # Process header fields
        if 'header_fields' in extracted_data:
            _process_item(extracted_data['header_fields'])

        # Process line items
        if 'invoice_table' in extracted_data and isinstance(extracted_data['invoice_table'], list):
            for line_item in extracted_data['invoice_table']:
                _process_item(line_item)

        logger.info("Strict tax validation complete.")
        return extracted_data
    
    def _check_extraction_completeness(self, extracted_data: Dict[str, Any], original_text: str) -> List[str]:
        """Check for potential missing data in extracted information"""
        warnings = []
        
        # Count pages in original text
        page_count = original_text.count("=== PAGE")
        if page_count == 0:
            page_count = 1  # Single page document
        
        # Check if we processed all pages
        invoice_table = extracted_data.get('invoice_table', [])
        if invoice_table:
            pages_with_items = set()
            for item in invoice_table:
                if 'page_number' in item and item['page_number']:
                    try:
                        pages_with_items.add(int(str(item['page_number']).strip()))
                    except:
                        pass
            
            if len(pages_with_items) < page_count:
                warnings.append(f"Extracted items from {len(pages_with_items)} pages but document has {page_count} pages")
        
        # Check for common missing header fields
        header_fields = extracted_data.get('header_fields', {})
        critical_fields = ['invoice_number', 'invoice_date', 'supplier_name', 'total_amount']
        missing_critical = [field for field in critical_fields if not header_fields.get(field)]
        
        if missing_critical:
            warnings.append(f"Missing critical header fields: {', '.join(missing_critical)}")
        
        # Check for empty line items when text contains table patterns
        table_indicators = ['sl.no', 'sr.no', 'item', 'description', 'qty', 'rate', 'amount', 'total']
        has_table_pattern = any(indicator in original_text.lower() for indicator in table_indicators)
        
        if has_table_pattern and not invoice_table:
            warnings.append("Document appears to contain table data but no line items were extracted")
        
        # Check for incomplete line items
        if invoice_table:
            incomplete_items = 0
            for item in invoice_table:
                essential_fields = ['description', 'quantity', 'rate', 'total_amount']
                missing_fields = sum(1 for field in essential_fields if not item.get(field))
                if missing_fields >= 3:  # More than half missing
                    incomplete_items += 1
            
            if incomplete_items > 0:
                warnings.append(f"{incomplete_items} line items appear incomplete (missing essential data)")
        
        return warnings
    
    def convert_to_target_format(self, extracted_data: Dict[str, Any], process_id: str, filename: str, file_path: str) -> Dict[str, Any]:
        """Convert extracted data to target JSON format"""
        header_fields = extracted_data.get('header_fields', {})
        invoice_table = extracted_data.get('invoice_table', [])
        
        # Create rawtext from extracted data
        rawtext_parts = []
        if header_fields:
            for key, value in header_fields.items():
                if value:
                    rawtext_parts.append(f"{key}: {value}")
        
        if invoice_table:
            for item in invoice_table:
                item_text = ""
                for key, value in item.items():
                    if value:
                        item_text += f"{key}: {value} "
                if item_text.strip():
                    rawtext_parts.append(item_text.strip())
        
        rawtext = " ".join(rawtext_parts)
        
        # Convert header fields to headerInfo format
        header_info = []
        header_mapping = {
            "invoice_number": "Invoice Number",
            "invoice_date": "Invoice Date", 
            "due_date": "Due Date",
            "po_number": "PO No#",
            "supplier_gstin": "GSTIN",
            "subtotal": "Net Amount",
            "total_amount": "Invoice Total Amount"
        }
        
        for field_key, field_value in header_fields.items():
            if field_value:
                header_key = header_mapping.get(field_key, field_key.replace('_', ' ').title())
                header_info.append({
                    "key": header_key,
                    "value": str(field_value),
                    "key_bbox": [[0, 0], [0, 0], [0, 0], [0, 0]],
                    "value_bbox": [[0, 0], [0, 0], [0, 0], [0, 0]],
                    "method": "AI",  # Changed from "right_aligned_pair" to "AI"
                    "doc_text": header_key.replace("#", "").strip(),
                    "closest_distance": 0
                })
        
        # Convert invoice table to lineData format
        line_data = []
        table_info = []
        
        if invoice_table:
            # Create header row - include discount column
            headers = ["#", "Item & Description", "HSN/SAC", "Qty", "Rate", "Discount", "CGST %", "Amt", "SGST %", "Amt", "Amount"]
            line_data.append(headers)
            
            # Create table info
            for i, header in enumerate(headers):
                table_info.append({
                    "key": header,
                    "position": i + 1,
                    "coordinates": [i * 100, (i + 1) * 100]
                })
            
            # Add data rows - include discount column
            for idx, item in enumerate(invoice_table, 1):
                # Extract discount from various possible field names
                discount_value = (item.get('discount') or 
                                item.get('discount_amount') or 
                                item.get('discount_percent') or 
                                item.get('disc') or 
                                item.get('disc_amt') or '')
                
                row = [
                    str(idx),
                    item.get('description') or item.get('item_description') or item.get('product_name') or '',
                    item.get('hsn_sac') or item.get('hsn_code') or item.get('sac_code') or '',
                    item.get('quantity') or item.get('qty') or '',
                    item.get('rate') or item.get('unit_price') or item.get('price') or '',
                    discount_value,  # Add discount column
                    item.get('cgst_rate') or item.get('gst_rate') or '',
                    item.get('cgst_amount') or item.get('gst_amount') or '',
                    item.get('sgst_rate') or item.get('gst_rate') or '',
                    item.get('sgst_amount') or item.get('gst_amount') or '',
                    item.get('total_amount') or item.get('amount') or item.get('line_total') or item.get('net_amount') or ''
                ]
                line_data.append(row)
        
        return {
            "page": 1,
            "identified_doc_type": "INVOICE",
            "rawtext": rawtext,
            "headerInfo": header_info,
            "paymentSts": "UNPAID",
            "incl_Tax": False,
            "lineInfo": {
                "lineData": line_data,
                "tableInfo": table_info,
                "excludeLine": [],
                "tablePosition": [[0, 800], [0, 1200]]
            },
            "pageWiseFilePath": f"app/temp_uploads/{process_id}/local_uploads/page1",
            "pageWisedocPath": f"app/temp_uploads/{process_id}/local_uploads/page1/{filename}"
        }