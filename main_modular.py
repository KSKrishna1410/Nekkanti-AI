#!/usr/bin/env python3
"""
OCR Processing API using FastAPI with modular architecture
Supports both Invoice and Bank Statement processing with SFTP upload
"""

import os
import uuid
import logging
from typing import Optional
from enum import Enum

import uvicorn
import dotenv
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import click

# Import modular components
from modules.utils.sftp_client import SFTPClient
from modules.utils.document_processor import DocumentProcessor
from modules.invoice.processor import InvoiceProcessor
from modules.bankstatement.processor import BankStatementProcessor

# Load environment variables
dotenv.load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Document type enum
class DocumentType(str, Enum):
    INVOICE = "INVOICE"
    BANKSTMT = "BANKSTMT"

# Request/Response models
class Body_ocr_process_file_ocr_process__post(BaseModel):
    file: UploadFile
    output_dir: Optional[str] = None
    doctype: DocumentType

class HTTPValidationError(BaseModel):
    detail: list

# Initialize FastAPI app
app = FastAPI(
    title="OCR Processing API",
    description="OCR processing endpoint for Invoice and Bank Statement documents with SFTP support",
    version="2.0.0"
)

class OCRProcessor:
    def __init__(self):
        # Initialize components
        try:
            self.sftp_client = SFTPClient()
            logger.info("✅ SFTP client initialized successfully")
        except Exception as e:
            logger.warning(f"⚠️ SFTP client initialization failed: {e}")
            self.sftp_client = None
        
        self.document_processor = DocumentProcessor()
        self.invoice_processor = InvoiceProcessor()
        self.bankstatement_processor = BankStatementProcessor()
        
        logger.info("✅ OCR Processor initialized successfully")
    
    def process_document(self, file_content: bytes, filename: str, doctype: DocumentType) -> tuple:
        """Process document based on its type"""
        if doctype == DocumentType.INVOICE:
            return self.invoice_processor.process_invoice(file_content, filename)
        elif doctype == DocumentType.BANKSTMT:
            return self.bankstatement_processor.process_bankstatement(file_content, filename)
        else:
            raise ValueError(f"Unsupported document type: {doctype}")
    
    def convert_to_target_format(self, extracted_data: dict, process_id: str, filename: str, file_path: str, doctype: DocumentType) -> dict:
        """Convert extracted data to target format based on document type"""
        if doctype == DocumentType.INVOICE:
            return self.invoice_processor.convert_to_target_format(extracted_data, process_id, filename, file_path)
        elif doctype == DocumentType.BANKSTMT:
            return self.bankstatement_processor.convert_to_target_format(extracted_data, process_id, filename, file_path)
        else:
            raise ValueError(f"Unsupported document type: {doctype}")

# Global processor instance
try:
    processor = OCRProcessor()
except Exception as e:
    logger.error(f"Failed to initialize OCRProcessor: {e}")
    raise

@app.post("/ocr_process/")
async def ocr_process_file(
    file: UploadFile = File(..., description="Document file to process"),
    output_dir: Optional[str] = Form(None, description="Optional output directory"),
    doctype: DocumentType = Form(..., description="Document type - INVOICE or BANKSTMT")
):
    """
    OCR processing endpoint that matches the original API signature for Java client compatibility.
    Includes detailed debugging for troubleshooting client requests.
    """
    
    # Log request details for debugging
    logger.info(f"🔍 OCR Processing Request:")
    logger.info(f"   - File: {file.filename}")
    logger.info(f"   - Content Type: {file.content_type}")
    logger.info(f"   - Output Dir: {output_dir}")
    logger.info(f"   - Document Type: {doctype}")
    
    # Check supported file types
    is_supported, file_type = processor.document_processor.is_supported_file_type(file.filename)
    if not is_supported:
        allowed_extensions = ['.pdf', '.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif']
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type. Supported formats: {', '.join(allowed_extensions)}"
        )
    
    try:
        # Generate process ID (UUID)
        process_id = str(uuid.uuid4())
        logger.info(f"📋 Generated Process ID: {process_id}")
        
        # Read file content
        file_content = await file.read()
        if not file_content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty file uploaded"
            )
        
        logger.info(f"📄 File size: {len(file_content)} bytes")
        
        # Upload file to SFTP if available
        sftp_file_path = None
        sftp_file_dir = None
        if processor.sftp_client:
            try:
                sftp_file_path, sftp_file_dir = processor.sftp_client.upload_file(
                    file_content, file.filename, process_id
                )
                logger.info(f"📤 File uploaded to SFTP: {sftp_file_path}")
            except Exception as e:
                logger.warning(f"⚠️ SFTP upload failed, continuing without: {e}")
        
        # Process document based on type
        logger.info(f"🔄 Processing {doctype} document...")
        extracted_data, token_usage, errors = processor.process_document(file_content, file.filename, doctype)
        
        if errors:
            logger.warning(f"⚠️ Processing errors: {errors}")
        
        # Construct response in target format
        response_data = {
            "processId": process_id,
            "filePath": sftp_file_path or f"local/{file.filename}",
            "fileDir": sftp_file_dir or "local",
            "document_type": doctype.value,
            "page_cnt": 1,
            "isSingleDoc": True,
            "obj_Type": "SINGLE_DOC_OBJ",
            "fileType": file_type,
            "pageWiseData": [],
            "lineTabulaData": [],
            "sftp_original_file": sftp_file_path or f"local/{file.filename}",
            "sftp_results_file": f"{sftp_file_path}_results.json" if sftp_file_path else f"local/{file.filename}_results.json"
        }
        
        # Convert extracted data to target format
        if extracted_data:
            page_wise_data = processor.convert_to_target_format(
                extracted_data, process_id, file.filename, sftp_file_path or f"local/{file.filename}", doctype
            )
            response_data["pageWiseData"] = [page_wise_data]
        
        # Create final response
        final_response = {
            "status_code": 200,
            "status": "Success",
            "data": response_data
        }
        
        # Upload JSON result to SFTP if available
        if processor.sftp_client:
            try:
                sftp_result_path = processor.sftp_client.upload_json_result(
                    final_response, process_id, file.filename
                )
                response_data["sftp_results_file"] = sftp_result_path
                logger.info(f"📤 JSON result uploaded to SFTP: {sftp_result_path}")
            except Exception as e:
                logger.warning(f"⚠️ SFTP JSON upload failed: {e}")
        
        logger.info(f"✅ OCR processing completed successfully for {doctype}")
        logger.info(f"   - Token Usage: Input={token_usage.input_tokens}, Output={token_usage.output_tokens}, Cost=${token_usage.total_cost_usd:.4f}")
        
        return JSONResponse(
            status_code=200,
            content=final_response
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error during OCR processing: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "ocr-processing-api", "version": "2.0.0"}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "OCR Processing API",
        "version": "2.0.0",
        "supported_document_types": ["INVOICE", "BANKSTMT"],
        "endpoints": {
            "POST /ocr_process/": "Process document with OCR",
            "GET /health": "Health check",
            "GET /docs": "API documentation"
        }
    }

# CLI Interface
@click.group()
def cli():
    """OCR Processing API CLI"""
    pass

@cli.command()
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.option('--port', default=8000, help='Port to bind to')
@click.option('--reload', is_flag=True, help='Enable auto-reload')
def serve(host, port, reload):
    """Start the API server"""
    uvicorn.run("main_modular:app", host=host, port=port, reload=reload)

@cli.command()
@click.argument('file_path')
@click.argument('doctype', type=click.Choice(['INVOICE', 'BANKSTMT']))
@click.option('--output', '-o', help='Output JSON file path')
def process(file_path, doctype, output):
    """Process a document file directly"""
    try:
        with open(file_path, 'rb') as f:
            file_content = f.read()
        
        doc_type = DocumentType(doctype)
        extracted_data, token_usage, errors = processor.process_document(file_content, os.path.basename(file_path), doc_type)
        
        result = {
            "extracted_data": extracted_data,
            "token_usage": {
                "input_tokens": token_usage.input_tokens,
                "output_tokens": token_usage.output_tokens,
                "total_cost_usd": token_usage.total_cost_usd,
                "provider": token_usage.provider,
                "model": token_usage.model
            },
            "errors": errors
        }
        
        if output:
            import json
            with open(output, 'w') as f:
                json.dump(result, f, indent=2)
            click.echo(f"Results saved to {output}")
        else:
            import json
            click.echo(json.dumps(result, indent=2))
            
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        raise click.Abort()

if __name__ == "__main__":
    cli()