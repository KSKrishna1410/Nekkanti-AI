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
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel
import click
import re

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

# Middleware to fix Java client boundary format issues
@app.middleware("http")
async def debug_requests(request: Request, call_next):
    # Fix Content-Type boundary format for Java client compatibility
    if request.url.path == "/ocr_process/" and request.method == "POST":
        content_type = request.headers.get("content-type", "")
        if "multipart/form-data" in content_type and "boundary====" in content_type:
            # Read the body first to modify it
            body = await request.body()
            
            # Extract the original problematic boundary
            boundary_match = re.search(r'boundary=(=+.*?=+)(?:;|$)', content_type)
            if boundary_match:
                original_boundary = boundary_match.group(1)
                # Create a clean boundary (just the middle numeric part)
                clean_boundary_match = re.search(r'=*(\d+)=*', original_boundary)
                if clean_boundary_match:
                    clean_boundary = f"boundary{clean_boundary_match.group(1)}"
                    
                    # Fix the Content-Type header
                    new_content_type = f"multipart/form-data; boundary={clean_boundary}"
                    
                    # Fix the body by replacing boundary markers (binary-safe)
                    # Replace --===NUMERIC=== with --boundaryNUMERIC
                    old_boundary_marker = f"--{original_boundary}".encode('utf-8')
                    new_boundary_marker = f"--{clean_boundary}".encode('utf-8')
                    fixed_body = body.replace(old_boundary_marker, new_boundary_marker)
                    
                    # Also fix the ending boundary marker
                    old_end_marker = f"--{original_boundary}--".encode('utf-8')
                    new_end_marker = f"--{clean_boundary}--".encode('utf-8')
                    fixed_body = fixed_body.replace(old_end_marker, new_end_marker)
                    
                    # Update headers
                    fixed_headers = []
                    for name_bytes, value_bytes in request.scope["headers"]:
                        name = name_bytes.decode()
                        if name.lower() == "content-type":
                            fixed_headers.append((b"content-type", new_content_type.encode()))
                        elif name.lower() == "content-length":
                            fixed_headers.append((b"content-length", str(len(fixed_body)).encode()))
                        else:
                            fixed_headers.append((name_bytes, value_bytes))
                    
                    request.scope["headers"] = fixed_headers
                    
                    # Create a new receive callable with the fixed body
                    async def receive():
                        return {"type": "http.request", "body": fixed_body}
                    
                    request._receive = receive
                    
                    # Continue processing with fixed request
                    response = await call_next(request)
                    return response
    
    # For non-OCR requests or requests that don't need fixing, use original flow
    body = await request.body()
    
    # Create a new request with the body for the endpoint to use
    async def receive():
        return {"type": "http.request", "body": body}
    
    request._receive = receive
    
    # Continue to the actual endpoint
    response = await call_next(request)
    return response

# Add exception handler for all HTTP exceptions
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

# Add general exception handler for any unhandled exceptions
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

# Add exception handler for validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()}
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
    
    def cleanup(self):
        """Clean up temporary files from all processors"""
        try:
            # Cleanup bank statement processor
            if hasattr(self.bankstatement_processor, 'cleanup'):
                self.bankstatement_processor.cleanup()
        except Exception as e:
            logger.warning(f"⚠️ Cleanup failed: {e}")
    
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
    
    logger.info(f"🔍 OCR Processing Request:")
    logger.info(f"   - File: {file.filename}, Content-Type: {file.content_type}")
    logger.info(f"   - Output Dir: {output_dir}, Document Type: {doctype}")

    is_supported, file_type = processor.document_processor.is_supported_file_type(file.filename)
    if not is_supported:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type. Supported formats: .pdf, .png, .jpg, .jpeg, .bmp, .tiff, .tif"
        )
    
    try:
        process_id = str(uuid.uuid4())
        logger.info(f"📋 Generated Process ID: {process_id}")
        
        file_content = await file.read()
        if not file_content:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty file uploaded")
        
        logger.info(f"📄 File size: {len(file_content)} bytes")
        
        sftp_file_path, sftp_file_dir = None, None
        if processor.sftp_client:
            try:
                sftp_file_path, sftp_file_dir = processor.sftp_client.upload_file(file_content, file.filename, process_id)
                logger.info(f"📤 File uploaded to SFTP: {sftp_file_path}")
            except Exception as e:
                logger.warning(f"⚠️ SFTP upload failed, continuing without: {e}")
        
        logger.info(f"🔄 Processing {doctype} document...")
        all_pages_extracted_data, token_usage, errors = processor.process_document(file_content, file.filename, doctype)
        
        if errors:
            logger.warning(f"⚠️ Processing errors: {errors}")

        page_wise_data = []
        if all_pages_extracted_data:
            for page_data in all_pages_extracted_data:
                formatted_page = processor.convert_to_target_format(
                    page_data, process_id, file.filename, sftp_file_path or f"local/{file.filename}", doctype
                )
                page_wise_data.append(formatted_page)

        response_data = {
            "processId": process_id,
            "filePath": sftp_file_path or f"local/{file.filename}",
            "fileDir": sftp_file_dir or "local",
            "document_type": doctype.value,
            "page_cnt": len(page_wise_data),
            "isSingleDoc": len(page_wise_data) <= 1,
            "obj_Type": "SINGLE_DOC_OBJ" if len(page_wise_data) <= 1 else "MULTI_DOC_OBJ",
            "fileType": file_type,
            "pageWiseData": page_wise_data,
            "lineTabulaData": [],
            "sftp_original_file": sftp_file_path or f"local/{file.filename}",
            "sftp_results_file": f"{sftp_file_path}_results.json" if sftp_file_path else f"local/{file.filename}_results.json"
        }
        
        final_response = {"status_code": 200, "status": "Success", "data": response_data}
        
        if processor.sftp_client:
            try:
                sftp_result_path = processor.sftp_client.upload_json_result(final_response, process_id, file.filename)
                response_data["sftp_results_file"] = sftp_result_path
                logger.info(f"📤 JSON result uploaded to SFTP: {sftp_result_path}")
            except Exception as e:
                logger.warning(f"⚠️ SFTP JSON upload failed: {e}")
        
        logger.info(f"✅ OCR processing completed successfully for {doctype}")
        logger.info(f"   - Token Usage: Input={token_usage.input_tokens}, Output={token_usage.output_tokens}, Cost=${token_usage.total_cost_usd:.4f}")
        
        processor.cleanup()
        
        return JSONResponse(status_code=200, content=final_response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error during OCR processing: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "ocr-processing-api", "version": "2.0.0"}

@app.get("/test")
async def test_endpoint():
    """Test endpoint for quick verification"""
    return {"status": "OK", "message": "OCR API is running", "version": "2.0.0"}

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