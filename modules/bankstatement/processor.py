#!/usr/bin/env python3
"""
Bank statement processing module for extracting structured data from bank statement documents
"""

import json
import logging
from typing import Dict, Any, List, Tuple

from ..utils.ai_provider import AIProviderClient, TokenUsage
from ..utils.document_processor import DocumentProcessor

logger = logging.getLogger(__name__)


class BankStatementProcessor:
    def __init__(self):
        self.ai_client = AIProviderClient()
        self.document_processor = DocumentProcessor()
    
    def process_bankstatement(self, file_content: bytes, filename: str) -> Tuple[Dict[str, Any], TokenUsage, List[str]]:
        """Process bank statement document and extract structured data"""
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
            prompt = self._create_bankstatement_extraction_prompt(text)
            
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
            
            # Post-process data
            extracted_data = self._post_process_bankstatement_data(extracted_data)
            
            return extracted_data, token_usage, errors
                
        except Exception as e:
            errors.append(f"Bank statement processing error: {str(e)}")
            logger.error(f"Bank statement processing error: {e}")
            return {}, token_usage, errors
    
    def _create_bankstatement_extraction_prompt(self, text: str) -> str:
        """Create a detailed prompt for AI to extract structured bank statement data"""
        return f"""
You are an expert bank statement data extraction system. Extract structured information from the following bank statement text and return it as valid JSON.

BANK STATEMENT TEXT:
{text}

Please extract the following information and return it as a JSON object with this exact structure:

{{
    "header_fields": {{
        "account_holder_name": "string or null",
        "account_number": "string or null",
        "bank_name": "string or null",
        "branch_name": "string or null",
        "ifsc_code": "string or null",
        "statement_period_from": "string or null",
        "statement_period_to": "string or null",
        "opening_balance": "string or null",
        "closing_balance": "string or null",
        "currency": "string or null"
    }},
    "transaction_table": [
        {{
            "sl_no": "string or null",
            "transaction_date": "string or null",
            "value_date": "string or null",
            "description": "string or null",
            "transaction_ref": "string or null",
            "debit_amount": "string or null",
            "credit_amount": "string or null",
            "balance": "string or null",
            "transaction_type": "string or null"
        }}
    ]
}}

INSTRUCTIONS:
1. Extract ALL transaction entries from the bank statement
2. CRITICAL: Always extract serial numbers or generate sequential numbers (1, 2, 3...) for "sl_no" field
3. CRITICAL: Look for transaction descriptions, reference numbers, dates, amounts
4. For header fields, extract: account details, bank details, statement period, balances
5. For transaction data, extract: dates, descriptions, amounts (debit/credit), balances
6. Use null for any fields not found in the statement
7. Keep numeric values as strings to preserve formatting
8. Identify transaction type (DEBIT/CREDIT) based on amount columns
9. Return only valid JSON, no additional text or explanations
"""
    
    def _post_process_bankstatement_data(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """Post-process bank statement data"""
        if not extracted_data or 'transaction_table' not in extracted_data:
            return extracted_data
        
        transaction_table = extracted_data['transaction_table']
        if not transaction_table:
            return extracted_data
        
        # Ensure serial numbers are present
        for i, transaction in enumerate(transaction_table, 1):
            if not transaction.get('sl_no'):
                transaction['sl_no'] = str(i)
        
        return extracted_data
    
    def convert_to_target_format(self, extracted_data: Dict[str, Any], process_id: str, filename: str, file_path: str) -> Dict[str, Any]:
        """Convert extracted data to target JSON format for bank statements"""
        header_fields = extracted_data.get('header_fields', {})
        transaction_table = extracted_data.get('transaction_table', [])
        
        # Create rawtext from extracted data
        rawtext_parts = []
        if header_fields:
            for key, value in header_fields.items():
                if value:
                    rawtext_parts.append(f"{key}: {value}")
        
        if transaction_table:
            for item in transaction_table:
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
            "account_holder_name": "Account Holder Name",
            "account_number": "Account Number",
            "bank_name": "Bank Name",
            "statement_period_from": "Statement From",
            "statement_period_to": "Statement To",
            "opening_balance": "Opening Balance",
            "closing_balance": "Closing Balance"
        }
        
        for field_key, field_value in header_fields.items():
            if field_value:
                header_key = header_mapping.get(field_key, field_key.replace('_', ' ').title())
                header_info.append({
                    "key": header_key,
                    "value": str(field_value),
                    "key_bbox": [[0, 0], [0, 0], [0, 0], [0, 0]],
                    "value_bbox": [[0, 0], [0, 0], [0, 0], [0, 0]],
                    "method": "AI",  # Using AI method
                    "doc_text": header_key,
                    "closest_distance": 0
                })
        
        # Convert transaction table to lineData format
        line_data = []
        table_info = []
        
        if transaction_table:
            # Create header row
            headers = ["#", "Date", "Description", "Reference", "Debit", "Credit", "Balance"]
            line_data.append(headers)
            
            # Create table info
            for i, header in enumerate(headers):
                table_info.append({
                    "key": header,
                    "position": i + 1,
                    "coordinates": [i * 100, (i + 1) * 100]
                })
            
            # Add data rows
            for idx, transaction in enumerate(transaction_table, 1):
                row = [
                    str(idx),
                    transaction.get('transaction_date') or transaction.get('value_date') or '',
                    transaction.get('description') or '',
                    transaction.get('transaction_ref') or '',
                    transaction.get('debit_amount') or '',
                    transaction.get('credit_amount') or '',
                    transaction.get('balance') or ''
                ]
                line_data.append(row)
        
        return {
            "page": 1,
            "identified_doc_type": "BANKSTMT",
            "rawtext": rawtext,
            "headerInfo": header_info,
            "paymentSts": "PROCESSED",
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