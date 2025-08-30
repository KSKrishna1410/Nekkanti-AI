#!/usr/bin/env python3
"""
Streamlit UI for Invoice Extraction API
Displays extracted invoice data in structured format with download capabilities
"""

import streamlit as st
import pandas as pd
import json
import io
import os
import tempfile
import requests
from datetime import datetime
from typing import Optional, Dict, Any
import base64
from PIL import Image

# API configuration
API_BASE_URL = "http://localhost:8000"  # Default FastAPI server URL

# Page configuration
st.set_page_config(
    page_title="Nekkanti OCR",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for invoice styling
st.markdown("""
<style>
.invoice-header {
    background-color: #f8f9fa;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
    border: 1px solid #dee2e6;
}
.invoice-title {
    color: #2c3e50;
    font-size: 24px;
    font-weight: bold;
    text-align: center;
    margin-bottom: 20px;
}
.info-box {
    background-color: #e3f2fd;
    padding: 15px;
    border-radius: 5px;
    border-left: 4px solid #2196f3;
    margin: 10px 0;
}
.supplier-box {
    background-color: #f3e5f5;
    padding: 15px;
    border-radius: 5px;
    border-left: 4px solid #9c27b0;
}
.buyer-box {
    background-color: #e8f5e8;
    padding: 15px;
    border-radius: 5px;
    border-left: 4px solid #4caf50;
}
.cost-box {
    background-color: #fff3e0;
    padding: 15px;
    border-radius: 5px;
    border-left: 4px solid #ff9800;
}
.metric-container {
    text-align: center;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'extracted_data' not in st.session_state:
        st.session_state.extracted_data = None
    if 'api_status' not in st.session_state:
        st.session_state.api_status = None

def check_api_status() -> bool:
    """Check if the FastAPI server is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def call_extraction_api(file_bytes: bytes, filename: str, doctype: str = "INVOICE") -> Dict[str, Any]:
    """Call the FastAPI OCR processing endpoint"""
    try:
        # Prepare the file for upload
        files = {'file': (filename, file_bytes, 'application/octet-stream')}
        
        # Prepare form data
        data = {
            'doctype': doctype,
            'output_dir': None
        }
        
        # Make the API call to new modular endpoint
        response = requests.post(
            f"{API_BASE_URL}/ocr_process/",
            files=files,
            data=data,
            timeout=120  # 2 minutes timeout for processing
        )
        
        if response.status_code == 200:
            api_response = response.json()
            # Extract from the new response format
            if api_response.get("status") == "Success" and api_response.get("data"):
                page_wise_data = api_response["data"].get("pageWiseData", [])
                if page_wise_data and len(page_wise_data) > 0:
                    page_data = page_wise_data[0]
                    
                    # Convert back to expected format for Streamlit
                    header_fields = {}
                    invoice_table = []
                    
                    # Extract header info
                    header_info = page_data.get("headerInfo", [])
                    for item in header_info:
                        key = item.get("key", "")
                        value = item.get("value", "")
                        # Convert back to original key format
                        if key == "Invoice Number":
                            header_fields["invoice_number"] = value
                        elif key == "Invoice Date":
                            header_fields["invoice_date"] = value
                        elif key == "Due Date":
                            header_fields["due_date"] = value
                        elif key == "PO No#":
                            header_fields["po_number"] = value
                        elif key == "GSTIN":
                            header_fields["supplier_gstin"] = value
                        elif key == "Net Amount":
                            header_fields["subtotal"] = value
                        elif key == "Invoice Total Amount":
                            header_fields["total_amount"] = value
                        else:
                            # Use cleaned key
                            clean_key = key.lower().replace(" ", "_").replace("#", "")
                            header_fields[clean_key] = value
                    
                    # Extract line data
                    line_info = page_data.get("lineInfo", {})
                    line_data = line_info.get("lineData", [])
                    
                    if len(line_data) > 1:  # Skip header row
                        headers = line_data[0] if line_data else []
                        for row in line_data[1:]:
                            if len(row) > 0:  # Skip empty rows
                                item = {}
                                for i, value in enumerate(row):
                                    if i < len(headers) and value:
                                        header = headers[i]
                                        # Map to standard field names
                                        if header == "#":
                                            item["sl_no"] = value
                                        elif "Item" in header or "Description" in header:
                                            item["description"] = value
                                        elif "HSN" in header or "SAC" in header:
                                            item["hsn_sac"] = value
                                        elif "Qty" in header:
                                            item["quantity"] = value
                                        elif "Rate" in header:
                                            item["rate"] = value
                                        elif "CGST" in header and "%" in header:
                                            item["cgst_rate"] = value
                                        elif "CGST" in header and "Amt" in header:
                                            item["cgst_amount"] = value
                                        elif "SGST" in header and "%" in header:
                                            item["sgst_rate"] = value
                                        elif "SGST" in header and "Amt" in header:
                                            item["sgst_amount"] = value
                                        elif "Amount" in header:
                                            item["total_amount"] = value
                                        else:
                                            # Use cleaned header name
                                            clean_header = header.lower().replace(" ", "_").replace("%", "rate").replace("#", "")
                                            item[clean_header] = value
                                
                                if item:  # Only add if item has data
                                    invoice_table.append(item)
                    
                    return {
                        "header_fields": header_fields,
                        "invoice_table": invoice_table,
                        "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_cost_usd": 0.0, "provider": "", "model": ""},
                        "errors": []
                    }
            
            return {
                "header_fields": {},
                "invoice_table": [],
                "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_cost_usd": 0.0, "provider": "", "model": ""},
                "errors": [f"Unexpected API response format"]
            }
        else:
            return {
                "header_fields": {},
                "invoice_table": [],
                "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_cost_usd": 0.0, "provider": "", "model": ""},
                "errors": [f"API Error {response.status_code}: {response.text}"]
            }
    except requests.exceptions.Timeout:
        return {
            "header_fields": {},
            "invoice_table": [],
            "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_cost_usd": 0.0, "provider": "", "model": ""},
            "errors": ["Request timeout - the file might be too large or complex"]
        }
    except Exception as e:
        return {
            "header_fields": {},
            "invoice_table": [],
            "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_cost_usd": 0.0, "provider": "", "model": ""},
            "errors": [f"API call failed: {str(e)}"]
        }

def display_file_preview(file_bytes: bytes, filename: str) -> None:
    """Display file preview in Streamlit (PDF or Image)"""
    st.subheader("📄 File Preview")
    
    file_extension = filename.lower().split('.')[-1] if '.' in filename else ''
    
    try:
        if file_extension == 'pdf':
            try:
                # First try: Use PyMuPDF to convert PDF pages to images
                import fitz  # PyMuPDF
                
                # Create a temporary file to handle the PDF
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                    temp_file.write(file_bytes)
                    temp_path = temp_file.name
                
                try:
                    # Open the PDF
                    pdf_document = fitz.open(temp_path)
                    
                    # Get number of pages
                    num_pages = len(pdf_document)
                    
                    if num_pages > 0:
                        # Create tabs for multiple pages
                        if num_pages > 1:
                            tabs = st.tabs([f"Page {i+1}" for i in range(num_pages)])
                        else:
                            tabs = [st.container()]  # Single container for one page
                        
                        # Display each page
                        for page_num in range(num_pages):
                            with tabs[page_num]:
                                # Get the page
                                page = pdf_document[page_num]
                                
                                # Convert page to image
                                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
                                
                                # Convert to PIL Image
                                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                                
                                # Display the image
                                st.image(img, caption=f"Page {page_num + 1}")
                                
                                # Add download button for each page
                                img_bytes = io.BytesIO()
                                img.save(img_bytes, format='PNG')
                                st.download_button(
                                    label=f"📥 Download Page {page_num + 1} as Image",
                                    data=img_bytes.getvalue(),
                                    file_name=f"{filename}_page_{page_num + 1}.png",
                                    mime="image/png"
                                )
                    else:
                        st.warning("No pages found in the PDF")
                    
                    pdf_document.close()
                    
                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
                        
            except Exception as pdf_error:
                st.warning(f"Could not render PDF preview: {str(pdf_error)}")
                st.info("Falling back to basic PDF preview...")
                
                # Fallback: Basic PDF display with iframe
                base64_pdf = base64.b64encode(file_bytes).decode('utf-8')
                pdf_display = f"""
                <embed
                    src="data:application/pdf;base64,{base64_pdf}"
                    width="100%"
                    height="800px"
                    type="application/pdf"
                >
                """
                st.markdown(pdf_display, unsafe_allow_html=True)
            
            # Always provide download option
            st.download_button(
                label="📥 Download Original PDF",
                data=file_bytes,
                file_name=filename,
                mime="application/pdf"
            )
            
        elif file_extension in ['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif']:
            # Display Image
            try:
                # Convert bytes to PIL Image
                image = Image.open(io.BytesIO(file_bytes))
                
                # Display image with improved quality
                st.image(image, caption=filename)
                
                # Add image info
                st.info(f"🖼️ Image Info: {image.size[0]}x{image.size[1]} pixels, Mode: {image.mode}")
                
                # Add download button
                img_bytes = io.BytesIO()
                image.save(img_bytes, format=image.format or 'PNG')
                st.download_button(
                    label="📥 Download Image",
                    data=img_bytes.getvalue(),
                    file_name=filename,
                    mime=f"image/{image.format.lower() if image.format else 'png'}"
                )
            except Exception as img_error:
                st.error(f"Could not display image preview: {str(img_error)}")
                st.warning("Providing download option instead")
                st.download_button(
                    label="📥 Download Image",
                    data=file_bytes,
                    file_name=filename,
                    mime=f"image/{file_extension}"
                )
            
        else:
            # Fallback for unknown types
            st.warning(f"Preview not available for {file_extension.upper()} files")
            st.download_button(
                label="📥 Download file to view",
                data=file_bytes,
                file_name=filename,
                mime="application/octet-stream"
            )
        
    except Exception as e:
        st.error(f"Could not display file preview: {str(e)}")
        
        # Fallback: provide download button
        mime_type = "application/pdf" if file_extension == 'pdf' else f"image/{file_extension}"
        st.download_button(
            label="📥 Download file to view",
            data=file_bytes,
            file_name=filename,
            mime=mime_type
        )

def create_download_link(data, filename, file_format):
    """Create download link for data"""
    if file_format == 'json':
        json_str = json.dumps(data, indent=2, default=str)
        b64 = base64.b64encode(json_str.encode()).decode()
        href = f'<a href="data:application/json;base64,{b64}" download="{filename}.json">Download JSON</a>'
    elif file_format == 'csv':
        output = io.StringIO()
        data.to_csv(output, index=False)
        csv_str = output.getvalue()
        b64 = base64.b64encode(csv_str.encode()).decode()
        href = f'<a href="data:text/csv;base64,{b64}" download="{filename}.csv">Download CSV</a>'
    return href

def display_header_info_dict(header_fields, doc_type="INVOICE"):
    """Display document header information from dictionary format"""
    if doc_type == "INVOICE":
        st.markdown('<div class="invoice-title">📄 INVOICE DETAILS</div>', unsafe_allow_html=True)
        title_icon = "🏢"
        buyer_icon = "👤"
    else:  # BANKSTMT
        st.markdown('<div class="invoice-title">🏦 BANK STATEMENT DETAILS</div>', unsafe_allow_html=True)
        title_icon = "🏦"
        buyer_icon = "👤"
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="supplier-box">', unsafe_allow_html=True)
        if doc_type == "INVOICE":
            st.markdown(f"**{title_icon} SUPPLIER INFORMATION**")
            if header_fields.get('supplier_name'):
                st.write(f"**Name:** {header_fields['supplier_name']}")
            if header_fields.get('supplier_gstin'):
                st.write(f"**GSTIN:** {header_fields['supplier_gstin']}")
            if header_fields.get('supplier_address'):
                st.write(f"**Address:** {header_fields['supplier_address']}")
        else:  # BANKSTMT
            st.markdown(f"**{title_icon} BANK INFORMATION**")
            if header_fields.get('bank_name'):
                st.write(f"**Bank Name:** {header_fields['bank_name']}")
            if header_fields.get('branch_name'):
                st.write(f"**Branch:** {header_fields['branch_name']}")
            if header_fields.get('ifsc_code'):
                st.write(f"**IFSC Code:** {header_fields['ifsc_code']}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="buyer-box">', unsafe_allow_html=True)
        if doc_type == "INVOICE":
            st.markdown(f"**{buyer_icon} BUYER INFORMATION**")
            if header_fields.get('buyer_name'):
                st.write(f"**Name:** {header_fields['buyer_name']}")
            if header_fields.get('buyer_gstin'):
                st.write(f"**GSTIN:** {header_fields['buyer_gstin']}")
            if header_fields.get('buyer_address'):
                st.write(f"**Address:** {header_fields['buyer_address']}")
        else:  # BANKSTMT
            st.markdown(f"**{buyer_icon} ACCOUNT INFORMATION**")
            if header_fields.get('account_holder_name'):
                st.write(f"**Account Holder:** {header_fields['account_holder_name']}")
            if header_fields.get('account_number'):
                st.write(f"**Account Number:** {header_fields['account_number']}")
            if header_fields.get('statement_period_from') and header_fields.get('statement_period_to'):
                st.write(f"**Period:** {header_fields['statement_period_from']} to {header_fields['statement_period_to']}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Invoice details
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("**📋 INVOICE DETAILS**")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if header_fields.get('invoice_number'):
            st.metric("Invoice Number", header_fields['invoice_number'])
    with col2:
        if header_fields.get('invoice_date'):
            st.metric("Invoice Date", header_fields['invoice_date'])
    with col3:
        if header_fields.get('po_number'):
            st.metric("PO Number", header_fields['po_number'])
    with col4:
        if header_fields.get('due_date'):
            st.metric("Due Date", header_fields['due_date'])
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Financial summary
    if any([header_fields.get('subtotal'), header_fields.get('total_tax'), header_fields.get('total_amount')]):
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("**💰 FINANCIAL SUMMARY**")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if header_fields.get('currency'):
                st.metric("Currency", header_fields['currency'])
        with col2:
            if header_fields.get('subtotal'):
                st.metric("Subtotal", f"₹ {header_fields['subtotal']}")
        with col3:
            if header_fields.get('total_tax'):
                st.metric("Total Tax", f"₹ {header_fields['total_tax']}")
        with col4:
            if header_fields.get('total_amount'):
                st.metric("Total Amount", f"₹ {header_fields['total_amount']}")
        
        st.markdown('</div>', unsafe_allow_html=True)

def display_header_info(header_fields):
    """Display invoice header information in structured format"""
    st.markdown('<div class="invoice-title">📄 INVOICE DETAILS</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="supplier-box">', unsafe_allow_html=True)
        st.markdown("**🏢 SUPPLIER INFORMATION**")
        if header_fields.supplier_name:
            st.write(f"**Name:** {header_fields.supplier_name}")
        if header_fields.supplier_gstin:
            st.write(f"**GSTIN:** {header_fields.supplier_gstin}")
        if header_fields.supplier_address:
            st.write(f"**Address:** {header_fields.supplier_address}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="buyer-box">', unsafe_allow_html=True)
        st.markdown("**👤 BUYER INFORMATION**")
        if header_fields.buyer_name:
            st.write(f"**Name:** {header_fields.buyer_name}")
        if header_fields.buyer_gstin:
            st.write(f"**GSTIN:** {header_fields.buyer_gstin}")
        if header_fields.buyer_address:
            st.write(f"**Address:** {header_fields.buyer_address}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Invoice details
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("**📋 INVOICE DETAILS**")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if header_fields.invoice_number:
            st.metric("Invoice Number", header_fields.invoice_number)
    with col2:
        if header_fields.invoice_date:
            st.metric("Invoice Date", header_fields.invoice_date)
    with col3:
        if header_fields.po_number:
            st.metric("PO Number", header_fields.po_number)
    with col4:
        if header_fields.due_date:
            st.metric("Due Date", header_fields.due_date)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Financial summary
    if any([header_fields.subtotal, header_fields.total_tax, header_fields.total_amount]):
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("**💰 FINANCIAL SUMMARY**")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if header_fields.currency:
                st.metric("Currency", header_fields.currency)
        with col2:
            if header_fields.subtotal:
                st.metric("Subtotal", f"₹ {header_fields.subtotal}")
        with col3:
            if header_fields.total_tax:
                st.metric("Total Tax", f"₹ {header_fields.total_tax}")
        with col4:
            if header_fields.total_amount:
                st.metric("Total Amount", f"₹ {header_fields.total_amount}")
        
        st.markdown('</div>', unsafe_allow_html=True)

def display_line_items(line_items):
    """Display invoice line items in table format with dynamic headers"""
    if not line_items:
        st.warning("No line items found in the invoice")
        return None
    
    st.markdown("### 📊 INVOICE LINE ITEMS")
    
    # Convert line items to DataFrame using actual data structure
    # This preserves the original column names from the extracted invoice
    df = pd.DataFrame(line_items)
    
    # Clean up the DataFrame - remove empty columns and handle missing values
    # Remove columns that are completely empty
    df = df.dropna(axis=1, how='all')
    
    # Fill NaN values with empty strings for better display
    df = df.fillna('')
    
    # Reorder columns to put serial numbers first
    serial_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['sl_no', 'serial', 'item_no', 's.no', 'sl.no', '#', 's_no'])]
    other_cols = [col for col in df.columns if col not in serial_cols]
    
    if serial_cols:
        # Put serial number columns first
        df = df[serial_cols + other_cols]
    

    st.info(f"**Columns:** {', '.join(df.columns.tolist())}")
    
    # Create dynamic column configuration based on data types and content
    column_config = {}
    for col in df.columns:
        col_lower = col.lower()
        
        # Serial number columns - make them small and prominent
        if any(keyword in col_lower for keyword in ['sl_no', 'serial', 'item_no', 's.no', 'sl.no', '#']):
            column_config[col] = st.column_config.NumberColumn(
                col,
                width="small",
                help="Serial Number"
            )
        # Numeric columns
        elif any(keyword in col_lower for keyword in ['quantity', 'qty', 'rate', 'amount', 'total', 'price', 'value', 'cost']):
            column_config[col] = st.column_config.NumberColumn(
                col, 
                width="small" if 'qty' in col_lower or 'quantity' in col_lower else "medium",
                format="%.2f"
            )
        # Description columns - make them large
        elif any(keyword in col_lower for keyword in ['description', 'item_description', 'product_name', 'product', 'item']):
            column_config[col] = st.column_config.TextColumn(col, width="large")
        # HSN/SAC columns
        elif any(keyword in col_lower for keyword in ['hsn', 'sac', 'code']):
            column_config[col] = st.column_config.TextColumn(col, width="small")
        # Default text columns
        else:
            column_config[col] = st.column_config.TextColumn(col, width="medium")
    
    # Display the table with dynamic configuration
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config=column_config
    )
    
    # Show column summary
    with st.expander("📊 Column Details"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Available Columns:**")
            for col in df.columns:
                non_empty_count = df[col].astype(str).str.strip().ne('').sum()
                st.write(f"• **{col}**: {non_empty_count}/{len(df)} filled")
        
        with col2:
            st.write("**Data Types:**")
            for col in df.columns:
                # Try to infer if numeric
                try:
                    pd.to_numeric(df[col], errors='raise')
                    dtype = "Numeric"
                except:
                    dtype = "Text"
                st.write(f"• **{col}**: {dtype}")
    
    return df

def display_token_usage(token_usage):
    """Display token usage and cost information"""
    st.markdown('<div class="cost-box">', unsafe_allow_html=True)
    st.markdown("**🔢 API USAGE & COST**")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Input Tokens", 
            f"{token_usage.get('input_tokens', 0):,}",
            help="Number of tokens sent to AI provider"
        )
    with col2:
        st.metric(
            "Output Tokens", 
            f"{token_usage.get('output_tokens', 0):,}",
            help="Number of tokens generated by AI provider"
        )
    with col3:
        st.metric(
            "Total Cost", 
            f"${token_usage.get('total_cost_usd', 0.0):.4f}",
            help="Total cost for this API call"
        )
        
    # Show provider info if available
    if token_usage.get('provider') or token_usage.get('model'):
        st.caption(f"Provider: {token_usage.get('provider', 'Unknown')} | Model: {token_usage.get('model', 'Unknown')}")
    
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Main Streamlit application"""
    initialize_session_state()
    
    # Sidebar
    with st.sidebar:
        st.title("📄 OCR Document Processor")
        st.markdown("---")
        
        # Document Type Selection
        doc_type_choice = st.selectbox(
            "Choose Document Type",
            ["Invoice", "Bank Statement"],
            help="Select the type of document to process"
        )
        
        # Azure OpenAI API Key status check
        azure_openai_key = os.getenv("AZURE_OPENAI_API_KEY")
        
        if azure_openai_key:
            st.success("✅ Azure OpenAI API Key configured")
        else:
            st.error("❌ Azure OpenAI API Key not found")

        
        # API Server Configuration
        st.markdown("**⚙️ Server Configuration**")
        st.info(f"API Server: {API_BASE_URL}")
        
        # Check API server status
        if st.button("🔄 Check Server Status"):
            st.session_state.api_status = None  # Reset to recheck
            
        if st.session_state.api_status is None:
            with st.spinner("Checking API server..."):
                api_status = check_api_status()
                st.session_state.api_status = "online" if api_status else "offline"
        
        if st.session_state.api_status == "online":
            st.success("✅ API Server is running")
        else:
            st.error("❌ API Server is not reachable")
            st.info("Start the server: `python main_modular.py serve`")
        
        st.markdown("---")
        st.markdown("**📋 Instructions:**")
        st.markdown("1. Select document type")
        st.markdown("2. Upload PDF or image")
        st.markdown("3. Wait for AI extraction")
        st.markdown("4. View structured data")
        st.markdown("5. Download as CSV/JSON")
    
    # Main content
    st.title("🧾 Nekkanti OCR - AI")
    st.markdown("Upload a PDF or image document to extract structured data using AI (supports invoices and bank statements)")
    
    # File upload
    document_type_text = doc_type_choice.lower() if 'doc_type_choice' in locals() else 'document'
    uploaded_file = st.file_uploader(
        f"Choose a PDF or image file ({document_type_text})",
        type=['pdf', 'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif'],
        help=f"Upload a {document_type_text} PDF or image to extract structured data (supports scanned documents)"
    )
    
    if uploaded_file is not None:
        # Display file info
        st.info(f"📁 File: {uploaded_file.name} ({uploaded_file.size} bytes)")
        
        # Check API server status first
        if st.session_state.api_status == "offline":
            st.error("🚨 API Server is offline. Please start the server first.")
            st.info("🚀 To start the server: `python main.py serve --host 0.0.0.0 --port 8000`")
            return
        
        # Read file content for preview and processing
        file_content = uploaded_file.read()
        
        # Display file preview
        with st.expander("👁️ File Preview", expanded=True):
            display_file_preview(file_content, uploaded_file.name)
        
        # Map document type choice to API parameter
        doctype_map = {
            "Invoice": "INVOICE",
            "Bank Statement": "BANKSTMT"
        }
        selected_doctype = doctype_map.get(doc_type_choice, "INVOICE")
        
        # Extract data button
        button_text = "🚀 Extract Invoice Data" if selected_doctype == "INVOICE" else "🏦 Extract Bank Statement Data"
        if st.button(button_text, type="primary"):
            spinner_text = f"Extracting data from {doc_type_choice.lower()}..."
            with st.spinner(spinner_text):
                try:
                    # Call the API for extraction
                    api_response = call_extraction_api(file_content, uploaded_file.name, selected_doctype)
                    
                    # Store in session state (simplified - no need for class conversion)
                    st.session_state.extracted_data = {
                        'header_fields': api_response.get("header_fields", {}),
                        'invoice_table': api_response.get("invoice_table", []),
                        'token_usage': api_response.get("token_usage", {"input_tokens": 0, "output_tokens": 0, "total_cost_usd": 0.0, "provider": "", "model": ""}),
                        'errors': api_response.get("errors", []),
                        'filename': uploaded_file.name
                    }
                    
                    if api_response.get("errors"):
                        for error in api_response["errors"]:
                            st.error(f"  {error}")
                    else:
                        st.success("✅ Data extracted successfully!")
                
                except Exception as e:
                    st.error(f"❌ Extraction failed: {str(e)}")
                    return
    
    # Display extracted data
    if st.session_state.extracted_data is not None:
        st.markdown("---")
        
        # Display header information
        display_header_info_dict(st.session_state.extracted_data['header_fields'])
        
        # Display line items
        df = display_line_items(st.session_state.extracted_data['invoice_table'])
        
        # Display token usage
        display_token_usage(st.session_state.extracted_data['token_usage'])
        
        # Download section
        st.markdown("---")
        st.markdown("### 💾 Download Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Download Raw JSON"):
                # Prepare complete data for download
                complete_data = {
                    'header_fields': st.session_state.extracted_data['header_fields'],
                    'invoice_table': st.session_state.extracted_data['invoice_table'],
                    'token_usage': st.session_state.extracted_data['token_usage'],
                    'errors': st.session_state.extracted_data['errors'],
                    'extracted_at': datetime.now().isoformat()
                }
                
                json_str = json.dumps(complete_data, indent=2, default=str)
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name=f"{st.session_state.extracted_data['filename']}_extracted.json",
                    mime="application/json"
                )
        
        with col2:
            if df is not None and st.button("📊 Download Line Items CSV"):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"{st.session_state.extracted_data['filename']}_line_items.csv",
                    mime="text/csv"
                )
        
        with col3:
            if st.button("📋 Download Header Data CSV"):
                # Convert header fields to DataFrame
                header_dict = st.session_state.extracted_data['header_fields']
                header_df = pd.DataFrame([header_dict])
                
                csv = header_df.to_csv(index=False)
                st.download_button(
                    label="Download Header CSV",
                    data=csv,
                    file_name=f"{st.session_state.extracted_data['filename']}_header.csv",
                    mime="text/csv"
                )
        
        # Summary statistics
        if df is not None:
            st.markdown("---")
            st.markdown("### 📈 Summary Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Items", len(df))
            with col2:
                # Check if serial numbers are present
                serial_fields = ['sl_no', 'serial_number', 'item_no', 's_no', 'serial']
                has_serials = any(
                    any(item.get(field) for field in serial_fields) 
                    for item in st.session_state.extracted_data['invoice_table']
                )
                st.metric("Serial Numbers", "✓ Present" if has_serials else "✗ Missing")
            with col3:
                # Dynamic tax field detection
                tax_fields = ['cgst_amount', 'sgst_amount', 'igst_amount', 'tax_amount', 'gst_amount', 'tax']
                tax_count = 0
                for item in st.session_state.extracted_data['invoice_table']:
                    if any(item.get(field) for field in tax_fields):
                        tax_count += 1
                st.metric("Items with Tax Info", tax_count)
            with col4:
                # Dynamic total field detection
                total_fields = ['total_amount', 'amount', 'total', 'line_total', 'item_total']
                total_count = 0
                for item in st.session_state.extracted_data['invoice_table']:
                    if any(item.get(field) for field in total_fields):
                        total_count += 1
                st.metric("Items with Amounts", total_count)

if __name__ == "__main__":
    main()