#!/usr/bin/env python3
"""
Invoice processing module for extracting structured data from invoice documents
"""

import json
import logging
from typing import Dict, Any, List, Tuple

from ..utils.ai_provider import AIProviderClient, TokenUsage
from ..utils.document_processor import DocumentProcessor

logger = logging.getLogger(__name__)


class InvoiceProcessor:
    def __init__(self):
        self.ai_client = AIProviderClient()
        self.document_processor = DocumentProcessor()
    
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
            
            # Create extraction prompt
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
            
            # Post-process to validate and reconcile tax data
            extracted_data = self._post_process_tax_data(extracted_data)
            
            return extracted_data, token_usage, errors
                
        except Exception as e:
            errors.append(f"Invoice processing error: {str(e)}")
            logger.error(f"Invoice processing error: {e}")
            return {}, token_usage, errors
    
    def _create_invoice_extraction_prompt(self, text: str) -> str:
        """Create a detailed prompt for AI to extract structured invoice data"""
        return f"""
You are an expert invoice data extraction system. Extract structured information from the following invoice text and return it as valid JSON.

**IMPORTANT: Do not perform any calculations. Only return values that are explicitly present in the document. If a value is not present, return null.**

INVOICE TEXT:
{text}

Please extract the following information and return it as a JSON object with this exact structure:

{{
    "header_fields": {{
        "supplier_name": "string or null",
        "supplier_gstin": "string or null", 
        "supplier_address": "string or null",
        "buyer_name": "string or null",
        "buyer_gstin": "string or null",
        "buyer_address": "string or null",
        "invoice_number": "string or null",
        "invoice_date": "string or null",
        "payment_terms": "string or null",
        "subtotal": "string or null",
        "total_tax": "string or null", 
        "total_amount": "string or null",
        "currency": "string or null",
        "due_date": "string or null",
        "po_number": "string or null"
    }},
    "invoice_table": [
        {{
            "sl_no": "string or null",
            "serial_number": "string or null",
            "item_no": "string or null",
            "description": "string or null",
            "item_description": "string or null", 
            "product_name": "string or null",
            "hsn_sac": "string or null",
            "hsn_code": "string or null",
            "sac_code": "string or null",
            "quantity": "string or null",
            "qty": "string or null",
            "unit": "string or null",
            "uom": "string or null",
            "rate": "string or null",
            "unit_price": "string or null",
            "price": "string or null",
            "discount": "string or null",
            "discount_amount": "string or null",
            "taxable_value": "string or null",
            "taxable_amount": "string or null",
            "cgst_rate": "string or null",
            "cgst_amount": "string or null",
            "sgst_rate": "string or null",
            "sgst_amount": "string or null", 
            "igst_rate": "string or null",
            "igst_amount": "string or null",
            "gst_rate": "string or null",
            "gst_amount": "string or null",
            "tax_rate": "string or null",
            "tax_amount": "string or null",
            "total_amount": "string or null",
            "amount": "string or null",
            "line_total": "string or null",
            "net_amount": "string or null"
        }}
    ]
}}

INSTRUCTIONS:
1. Extract ALL line items from the invoice table/items section
2. CRITICAL: Always look for and extract serial numbers from columns like: S.No, Sl.No, Serial No, Item No, #, Row Number, Line Number, etc.
3. CRITICAL: If there is an Item Description without a specific header, include it in the "description" or "item_description" field
4. CRITICAL: Look for item descriptions that appear in table rows even if they don't have clear column headers
5. CRITICAL: Include any text that describes products/services/items, even if it's not clearly labeled as a description
6. If you find a column with sequential numbers (1, 2, 3...) at the start of each row, that's the serial number - extract it as "sl_no"
7. For header fields, look for: supplier info, buyer info, invoice details, amounts
8. For table data, preserve original column names and extract ALL available fields
9. Include multiple variations of the same field (e.g., both "quantity" and "qty" if present)
10. Use null for any fields not found in the invoice
11. Keep numeric values as strings to preserve formatting
12. If no explicit serial numbers exist in the invoice, generate sequential numbers (1, 2, 3...) for "sl_no" field
13. Pay special attention to: serial numbers, item descriptions, quantities, rates, taxes, totals
14. Return only valid JSON, no additional text or explanations

**INDIAN GST EXTRACTION RULES:**
- **IGST vs. CGST/SGST:**
  - If you see "IGST" mentioned, extract it. IGST is for inter-state sales.
  - If you see "CGST" and "SGST" mentioned, extract them. CGST and SGST are for intra-state sales.
  - **CRITICAL RULE:** An invoice will have EITHER IGST OR a combination of CGST and SGST. They are mutually exclusive. If you find any value for IGST, you MUST set CGST and SGST values to null. If you find values for CGST and SGST, you MUST set IGST to null. Prioritize extracting the tax type that has explicit non-zero amounts listed.
- **Generic Tax:** If you only see a generic "Tax" or "GST" field without specifying the type, try to infer it. If buyer and seller are in different states (if visible), it's likely IGST. If they are in the same state, it's CGST/SGST. If you cannot determine the type, extract the value into `gst_amount` and `gst_rate`.
- **Header vs. Line Items:** Extract taxes from both the header (e.g., "Total IGST") and from each line item in the table.
"""

    def _ensure_serial_numbers(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure serial numbers are present in invoice table data"""
        if not extracted_data or 'invoice_table' not in extracted_data:
            return extracted_data
        
        invoice_table = extracted_data['invoice_table']
        if not invoice_table:
            return extracted_data
        
        # Check if serial numbers already exist
        serial_fields = ['sl_no', 'serial_number', 'item_no', 's_no', 'serial']
        has_serials = any(
            any(str(item.get(field, '')).strip() for field in serial_fields)
            for item in invoice_table
        )
        
        # If no serial numbers found, generate them
        if not has_serials:
            for i, item in enumerate(invoice_table, 1):
                item['sl_no'] = str(i)
        
        return extracted_data

    def _post_process_tax_data(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates and reconciles GST data (IGST, CGST, SGST) in the extracted invoice.
        - Enforces mutual exclusivity of IGST and CGST/SGST.
        - Verifies and corrects tax amounts against rates and taxable value.
        """
        logger.info("Running tax post-processing and verification...")

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

            # --- Rule 2: Verification and Correction ---
            if taxable_value > 0:
                # Verify IGST
                if igst_rate > 0:
                    expected_igst = round(taxable_value * (igst_rate / 100), 2)
                    if abs(expected_igst - igst_amount) > 0.1: # Tolerance for rounding
                        logger.warning(f"Correcting IGST amount. Extracted: {igst_amount}, Calculated: {expected_igst}")
                        item['igst_amount'] = f"{expected_igst:.2f}"
                
                # Verify CGST
                if cgst_rate > 0:
                    expected_cgst = round(taxable_value * (cgst_rate / 100), 2)
                    if abs(expected_cgst - cgst_amount) > 0.1:
                        logger.warning(f"Correcting CGST amount. Extracted: {cgst_amount}, Calculated: {expected_cgst}")
                        item['cgst_amount'] = f"{expected_cgst:.2f}"

                # Verify SGST
                if sgst_rate > 0:
                    expected_sgst = round(taxable_value * (sgst_rate / 100), 2)
                    if abs(expected_sgst - sgst_amount) > 0.1:
                        logger.warning(f"Correcting SGST amount. Extracted: {sgst_amount}, Calculated: {expected_sgst}")
                        item['sgst_amount'] = f"{expected_sgst:.2f}"

        # Process header fields
        if 'header_fields' in extracted_data:
            _process_item(extracted_data['header_fields'])

        # Process line items
        if 'invoice_table' in extracted_data and isinstance(extracted_data['invoice_table'], list):
            for line_item in extracted_data['invoice_table']:
                _process_item(line_item)

        logger.info("Tax post-processing and verification complete.")
        return extracted_data
    
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
            # Create header row
            headers = ["#", "Item & Description", "HSN /SAC", "Qty", "Rate", "CGST %", "Amt", "SGST %", "Amt", "Amount"]
            line_data.append(headers)
            
            # Create table info
            for i, header in enumerate(headers):
                table_info.append({
                    "key": header,
                    "position": i + 1,
                    "coordinates": [i * 100, (i + 1) * 100]
                })
            
            # Add data rows
            for idx, item in enumerate(invoice_table, 1):
                row = [
                    str(idx),
                    item.get('description') or item.get('item_description') or item.get('product_name') or '',
                    item.get('hsn_sac') or item.get('hsn_code') or item.get('sac_code') or '',
                    item.get('quantity') or item.get('qty') or '',
                    item.get('rate') or item.get('unit_price') or item.get('price') or '',
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