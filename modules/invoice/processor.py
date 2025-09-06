#!/usr/bin/env python3
"""
Invoice processing module for extracting structured data from invoice documents
"""

import json
import logging
from typing import Dict, Any, List, Tuple

from modules.utils.ai_provider import AIProviderClient, TokenUsage
from modules.utils.document_processor import DocumentProcessor
from utils.prompt_manager import get_prompt_manager

logger = logging.getLogger(__name__)


class InvoiceProcessor:
    def __init__(self):
        self.ai_client = AIProviderClient()
        self.document_processor = DocumentProcessor()
        self.prompt_manager = get_prompt_manager()

    def process_invoice(self, file_content: bytes, filename: str) -> Tuple[List[Dict[str, Any]], TokenUsage, List[str]]:
        """Process invoice document page by page and extract structured data"""
        errors = []
        total_token_usage = TokenUsage()
        all_pages_data = []

        try:
            # Extract text from document, now returns a list of page texts
            if filename.lower().endswith('.pdf'):
                page_texts = self.document_processor.extract_text_from_pdf(file_content, filename)
            else:
                page_texts = self.document_processor.extract_text_from_image(file_content, filename)

            if not page_texts or not any(page.strip() for page in page_texts):
                errors.append("No text could be extracted from document")
                return [], total_token_usage, errors

            for i, text in enumerate(page_texts):
                page_num = i + 1
                logger.info(f"Processing page {page_num}/{len(page_texts)}")
                if not text.strip():
                    logger.info(f"Page {page_num} is empty, skipping.")
                    all_pages_data.append({
                        "page_number": page_num,
                        "header_fields": {},
                        "invoice_table": []
                    })
                    continue

                # Create extraction prompt using template
                prompt = self._create_invoice_extraction_prompt(text)

                # Generate AI response
                response_text, token_usage = self.ai_client.generate_response(prompt)
                total_token_usage.accumulate(token_usage)

                if not response_text:
                    errors.append(f"AI response was empty for page {page_num}")
                    continue

                # Parse JSON response
                try:
                    extracted_data = json.loads(response_text)
                    extracted_data['page_number'] = page_num # Add page number to data
                    
                    # Post-process to ensure serial numbers are present
                    extracted_data = self._ensure_serial_numbers(extracted_data)
                    
                    # Post-process to validate tax data
                    extracted_data = self._post_process_tax_data(extracted_data)

                    # Check extraction completeness and add warnings to errors
                    completeness_warnings = self._check_extraction_completeness(extracted_data, text, page_num)
                    if completeness_warnings:
                        errors.extend([f"Page {page_num} warning: {w}" for w in completeness_warnings])

                    all_pages_data.append(extracted_data)

                except json.JSONDecodeError as e:
                    errors.append(f"Failed to parse AI response for page {page_num} as JSON: {e}")
                    logger.error(f"Invalid JSON response for page {page_num}: {response_text[:500]}...")
                    continue
            
            # Consolidate header fields from all pages
            consolidated_data = self._consolidate_invoice_data(all_pages_data)

            return consolidated_data, total_token_usage, errors

        except Exception as e:
            errors.append(f"Invoice processing error: {str(e)}")
            logger.error(f"Invoice processing error: {e}", exc_info=True)
            return [], total_token_usage, errors

    def _consolidate_invoice_data(self, all_pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Consolidate header fields and keep line items separate per page."""
        consolidated_header = {}
        for page_data in all_pages_data:
            if 'header_fields' in page_data:
                consolidated_header.update(page_data['header_fields'])

        for page_data in all_pages_data:
            page_data['header_fields'] = consolidated_header
            
        return all_pages_data

    def _create_invoice_extraction_prompt(self, text: str) -> str:
        """Create a detailed prompt for AI to extract structured invoice data using template"""
        try:
            return self.prompt_manager.render_template('invoice_extraction.jinja', text=text)
        except Exception as e:
            logger.error(f"CRITICAL: Failed to load invoice extraction template: {e}")
            return f'FINANCIAL DOCUMENT EXTRACTION - STRICT MODE\n\nRULES: \n1. Extract ONLY what is explicitly written\n2. Do NOT calculate or infer anything\n3. Use null for missing fields\n4. Copy text exactly as written\n\nDOCUMENT TEXT:\n{text}\n\nReturn JSON with exact structure: {{"header_fields": {{}}, "invoice_table": []}}'

    def _ensure_serial_numbers(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check for serial numbers but do NOT generate them - strict mode"""
        if not extracted_data or 'invoice_table' not in extracted_data:
            return extracted_data
        
        invoice_table = extracted_data['invoice_table']
        if not invoice_table:
            return extracted_data
        
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
            if value is None: return 0.0
            try:
                cleaned_value = str(value).strip().replace('Rs.', '').replace('₹', '').replace(',', '').replace('%', '')
                return float(cleaned_value) if cleaned_value else 0.0
            except (ValueError, TypeError): return 0.0

        def _process_item(item: Dict[str, Any]):
            if not isinstance(item, dict): return

            igst_amount = _safe_float(item.get('igst_amount'))
            cgst_amount = _safe_float(item.get('cgst_amount'))
            sgst_amount = _safe_float(item.get('sgst_amount'))
            
            if igst_amount > 0 and (cgst_amount > 0 or sgst_amount > 0):
                logger.warning(f"Conflict found: IGST ({igst_amount}) and CGST/SGST ({cgst_amount}/{sgst_amount}) present. Prioritizing IGST.")
                item.update({'cgst_amount': None, 'sgst_amount': None, 'cgst_rate': None, 'sgst_rate': None})

        if 'header_fields' in extracted_data:
            _process_item(extracted_data['header_fields'])
        if 'invoice_table' in extracted_data and isinstance(extracted_data['invoice_table'], list):
            for line_item in extracted_data['invoice_table']:
                _process_item(line_item)

        logger.info("Strict tax validation complete.")
        return extracted_data

    def _check_extraction_completeness(self, extracted_data: Dict[str, Any], original_text: str, page_num: int) -> List[str]:
        """Check for potential missing data in extracted information for a single page"""
        warnings = []
        
        header_fields = extracted_data.get('header_fields', {})
        invoice_table = extracted_data.get('invoice_table', [])

        # Check for common missing header fields only on the first page
        if page_num == 1:
            critical_fields = ['invoice_number', 'invoice_date', 'supplier_name', 'total_amount']
            missing_critical = [field for field in critical_fields if not header_fields.get(field)]
            if missing_critical:
                warnings.append(f"Missing critical header fields: {', '.join(missing_critical)}")
        
        table_indicators = ['sl.no', 'sr.no', 'item', 'description', 'qty', 'rate', 'amount', 'total']
        has_table_pattern = any(indicator in original_text.lower() for indicator in table_indicators)
        
        if has_table_pattern and not invoice_table:
            warnings.append("Document page appears to contain table data but no line items were extracted")
        
        return warnings

    def convert_to_target_format(self, extracted_page_data: Dict[str, Any], process_id: str, filename: str, file_path: str) -> Dict[str, Any]:
        """Convert a single page's extracted data to target JSON format"""
        page_num = extracted_page_data.get('page_number', 1)
        header_fields = extracted_page_data.get('header_fields', {})
        invoice_table = extracted_page_data.get('invoice_table', [])
        
        rawtext_parts = [f"{key}: {value}" for key, value in header_fields.items() if value]
        if invoice_table:
            for item in invoice_table:
                item_text = " ".join([f"{key}: {value}" for key, value in item.items() if value])
                if item_text: rawtext_parts.append(item_text)
        rawtext = " ".join(rawtext_parts)
        
        header_info = []
        header_mapping = {
            "invoice_number": "Invoice Number", "invoice_date": "Invoice Date", "due_date": "Due Date",
            "po_number": "PO No#", "supplier_gstin": "GSTIN", "subtotal": "Net Amount",
            "total_amount": "Invoice Total Amount"
        }
        for field_key, field_value in header_fields.items():
            if field_value:
                header_key = header_mapping.get(field_key, field_key.replace('_', ' ').title())
                header_info.append({
                    "key": header_key, "value": str(field_value), "method": "AI",
                    "key_bbox": [[0,0],[0,0],[0,0],[0,0]], "value_bbox": [[0,0],[0,0],[0,0],[0,0]],
                    "doc_text": header_key.replace("#", "").strip(), "closest_distance": 0
                })
        
        line_data, table_info = [], []
        if invoice_table:
            headers = ["#", "Item & Description", "HSN/SAC", "Qty", "Rate", "Discount", "CGST %", "Amt", "SGST %", "Amt", "Amount"]
            line_data.append(headers)
            table_info = [{"key": h, "position": i + 1, "coordinates": [i * 100, (i + 1) * 100]} for i, h in enumerate(headers)]
            
            for idx, item in enumerate(invoice_table, 1):
                discount_value = item.get('discount') or item.get('discount_amount') or item.get('discount_percent') or item.get('disc') or item.get('disc_amt') or ''
                row = [
                    str(idx),
                    item.get('description') or item.get('item_description') or item.get('product_name') or '',
                    item.get('hsn_sac') or item.get('hsn_code') or item.get('sac_code') or '',
                    item.get('quantity') or item.get('qty') or '',
                    item.get('rate') or item.get('unit_price') or item.get('price') or '',
                    discount_value,
                    item.get('cgst_rate') or item.get('gst_rate') or '',
                    item.get('cgst_amount') or item.get('gst_amount') or '',
                    item.get('sgst_rate') or item.get('gst_rate') or '',
                    item.get('sgst_amount') or item.get('gst_amount') or '',
                    item.get('total_amount') or item.get('amount') or item.get('line_total') or item.get('net_amount') or ''
                ]
                line_data.append(row)
        
        return {
            "page": page_num,
            "identified_doc_type": "INVOICE",
            "rawtext": rawtext,
            "headerInfo": header_info,
            "paymentSts": "UNPAID",
            "incl_Tax": False,
            "lineInfo": {
                "lineData": line_data, "tableInfo": table_info,
                "excludeLine": [], "tablePosition": [[0, 800], [0, 1200]]
            },
            "pageWiseFilePath": f"app/temp_uploads/{process_id}/local_uploads/page{page_num}",
            "pageWisedocPath": f"app/temp_uploads/{process_id}/local_uploads/page{page_num}/{filename}"
        }
