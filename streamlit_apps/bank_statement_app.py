#!/usr/bin/env python3
"""
Streamlit UI for Bank Statement OCR Processing

This app provides a user-friendly interface to upload bank statements,
process them through the OCR API, and display results in both raw and
structured formats.
"""

import streamlit as st
import requests
import json
import pandas as pd
from typing import Optional, Dict, Any
import io
import time
import base64
from datetime import datetime
import os
import tempfile
from PIL import Image

# Authentication configuration
AUTH_CREDENTIALS = {
    "admin": os.getenv("ADMIN_PASSWORD", "Welcome!23"),
    "user": os.getenv("USER_PASSWORD", "user123"),
    "demo": os.getenv("DEMO_PASSWORD", "demo123")
}

def check_authentication():
    """Check if user is authenticated"""
    return st.session_state.get("authenticated", False)

def login_form():
    """Display login form"""
    st.title("🔐 Nekkanti OCR - Login")
    st.markdown("---")
    
    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("### Please Login to Continue")
            
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Enter username")
                password = st.text_input("Password", type="password", placeholder="Enter password")
                submit_button = st.form_submit_button("Login")
                
                if submit_button:
                    if username in AUTH_CREDENTIALS and AUTH_CREDENTIALS[username] == password:
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.success("✅ Login successful!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password")
            
            # # Demo credentials info
            # with st.expander("📋 Demo Credentials"):
            #     st.markdown("""
            #     **Available Demo Accounts:**
            #     - Username: `admin`, Password: `Welcome!23`
            #     - Username: `user`, Password: `user123`  
            #     - Username: `demo`, Password: `demo123`
            #     """)

def logout():
    """Logout user"""
    st.session_state.authenticated = False
    st.session_state.username = None
    st.rerun()

# Page configuration
st.set_page_config(
    page_title="Nekkanti OCR - Bank Statement Extractor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_HOST = os.getenv("API_HOST", "localhost")
API_PORT = os.getenv("API_PORT", "8000")
API_BASE_URL = f"http://{API_HOST}:{API_PORT}"
API_ENDPOINT = f"{API_BASE_URL}/ocr_process/"

def process_document(file_bytes: bytes, filename: str, doc_type: str) -> Optional[Dict[Any, Any]]:
    """Process document through the API with specified document type"""
    try:
        file_size_mb = len(file_bytes) / (1024 * 1024)
        st.write(f"📄 File size: {file_size_mb:.1f} MB")
        
        if file_size_mb > 50:
            st.warning(f"⚠️ Large file detected ({file_size_mb:.1f} MB). Processing may take longer.")
            
        files = {'file': (filename, io.BytesIO(file_bytes), 'application/pdf')}
        data = {
            'output_dir': '',
            'doctype': doc_type
        }
        
        with st.spinner(f'Processing {doc_type} through OCR API... This may take several minutes for large files.'):
            try:
                session = requests.Session()
                session.trust_env = False
                
                timeout = max(600, int(file_size_mb * 30))
                st.write(f"⏱️ Setting timeout to {timeout} seconds based on file size")
                
                response = session.post(
                    API_ENDPOINT,
                    files=files,
                    data=data,
                    timeout=timeout,
                    verify=False,
                    headers={
                        'Accept': 'application/json',
                        'User-Agent': 'Streamlit/1.0',
                        'X-File-Size': str(file_size_mb)
                    }
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    st.error(f"API Error {response.status_code}")
                    st.write("Response details:")
                    st.write(f"• Status: {response.status_code}")
                    st.write(f"• Headers: {dict(response.headers)}")
                    try:
                        st.write(f"• Body: {response.text}")
                    except:
                        st.write("• Body: Could not decode response body")
                    return None
                    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                return None
                
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

def display_table_data(table_data: list) -> None:
    """Display table data in a structured format"""
    st.subheader("📊 Transaction Table")
    
    if not table_data:
        st.warning("No transaction data found in the document")
        return
    
    # Convert to DataFrame and clean up null values
    df = pd.DataFrame(table_data)
    df = df.replace('null', None)  # Replace 'null' strings with None
    df = df.dropna(how='all')  # Drop rows where all values are None
    
    if df.empty:
        st.warning("Transaction table is empty")
        return
    
    # Display the table
    st.dataframe(df, use_container_width=True, height=400)
    
    # Download button with unique key
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download as CSV",
        data=csv,
        file_name="bank_statement_transactions.csv",
        mime="text/csv",
        key="download_bank_statement_table"
    )

def display_summary_metrics(response_data: dict) -> None:
    """Display summary metrics"""
    data = response_data.get('data', {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Document Type", data.get('document_type', 'N/A'))
    
    with col2:
        st.metric("Page Count", data.get('page_cnt', 'N/A'))
    
    with col3:
        page_wise_data = data.get('pageWiseData', [])
        headers_count = len(page_wise_data[0].get('headerInfo', [])) if page_wise_data else 0
        st.metric("Headers Found", headers_count)
    
    with col4:
        # Get line items count from the first page's lineInfo
        page_wise_data = data.get('pageWiseData', [])
        if page_wise_data and 'lineInfo' in page_wise_data[0] and 'lineData' in page_wise_data[0]['lineInfo']:
            table_rows = len(page_wise_data[0]['lineInfo']['lineData'])
        else:
            table_rows = 0
        st.metric("Table Rows", table_rows)

def display_headers(headers_data):
    """Display headers in a structured format"""
    st.subheader("📋 Extracted Headers")
    
    if not headers_data:
        st.warning("No headers found in the document")
        return
    
    # Group headers by category
    account_info = []
    bank_info = []
    statement_info = []
    balance_info = []
    
    for header in headers_data:
        if not isinstance(header, dict) or 'key' not in header or 'value' not in header:
            continue
            
        field_name = header['key']
        value = header['value']
        
        # Handle value if it's a dictionary string
        try:
            if isinstance(value, str) and value.startswith('{') and value.endswith('}'):
                import ast
                value_dict = ast.literal_eval(value)
                header_value = value_dict.get('value', '')
                key_text = value_dict.get('key_text', '')
                method = value_dict.get('method', '')
                confidence = value_dict.get('confidence', 0)
                validation_score = value_dict.get('validation_score', 0)
                spatial_score = value_dict.get('spatial_score', 0)
                data_type = value_dict.get('data_type', '')
                field_type = value_dict.get('field_type', '')
            else:
                header_value = value
                key_text = header.get('doc_text', '')
                method = header.get('method', '')
                confidence = header.get('confidence', 0)
                validation_score = header.get('validation_score', 0)
                spatial_score = 0
                data_type = ''
                field_type = ''
        except:
            # Fallback if parsing fails
            header_value = value
            key_text = header.get('doc_text', '')
            method = header.get('method', '')
            confidence = header.get('confidence', 0)
            validation_score = header.get('validation_score', 0)
            spatial_score = 0
            data_type = ''
            field_type = ''
            
        if not header_value or header_value == 'null':  # Skip empty or null values
            continue
            
        header_data = {
            'Field': field_name,
            'Value': header_value,
            'Key Text': key_text,
            'Method': method,
            'Data Type': data_type,
            'Field Type': field_type,
            'Confidence': f"{confidence:.2%}" if confidence > 0 else "N/A",
            'Validation': f"{validation_score:.2%}" if validation_score > 0 else "N/A",
            'Spatial Score': f"{spatial_score:.2%}" if spatial_score > 0 else "N/A"
        }
        
        # Categorize headers
        if field_name in ['Account Number', 'IFSC Code']:
            account_info.append(header_data)
        elif field_name.startswith('Bank'):
            bank_info.append(header_data)
        elif 'Date' in field_name:
            statement_info.append(header_data)
        elif 'Balance' in field_name:
            balance_info.append(header_data)
    
    # Display in columns
    col1, col2 = st.columns(2)
    
    # Column configuration for all dataframes
    column_config = {
        "Field": st.column_config.TextColumn("Field", width="medium"),
        "Value": st.column_config.TextColumn("Value", width="medium"),
        "Key Text": st.column_config.TextColumn("Key Text", width="medium"),
        "Method": st.column_config.TextColumn("Method", width="small"),
        "Data Type": st.column_config.TextColumn("Data Type", width="small"),
        "Field Type": st.column_config.TextColumn("Field Type", width="small"),
        "Confidence": st.column_config.TextColumn("Confidence", width="small"),
        "Validation": st.column_config.TextColumn("Validation", width="small"),
        "Spatial Score": st.column_config.TextColumn("Spatial Score", width="small")
    }
    
    with col1:
        if account_info:
            st.markdown("**🏦 Account Information**")
            df = pd.DataFrame(account_info)
            st.dataframe(
                df,
                column_config=column_config,
                use_container_width=True,
                hide_index=True
            )
        
        if statement_info:
            st.markdown("**📅 Statement Information**")
            df = pd.DataFrame(statement_info)
            st.dataframe(
                df,
                column_config=column_config,
                use_container_width=True,
                hide_index=True
            )
    
    with col2:
        if bank_info:
            st.markdown("**🏢 Bank Information**")
            df = pd.DataFrame(bank_info)
            st.dataframe(
                df,
                column_config=column_config,
                use_container_width=True,
                hide_index=True
            )
        
        if balance_info:
            st.markdown("**💰 Balance Information**")
            df = pd.DataFrame(balance_info)
            st.dataframe(
                df,
                column_config=column_config,
                use_container_width=True,
                hide_index=True
            )
            
    # Add download button for all header data
    all_headers = account_info + bank_info + statement_info + balance_info
    if all_headers:
        df_all = pd.DataFrame(all_headers)
        csv = df_all.to_csv(index=False)
        st.download_button(
            label="📥 Download All Headers as CSV",
            data=csv,
            file_name="bank_statement_headers.csv",
            mime="text/csv",
            key="download_all_headers"
        )

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
                                st.image(img, caption=f"Page {page_num + 1}", use_container_width=True)
                                
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
                st.image(image, caption=filename, use_container_width=True)
                
                # Add image info
                st.info(f"📊 Image Info: {image.size[0]}x{image.size[1]} pixels, Mode: {image.mode}")
                
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

# Add new function for displaying invoice data
def display_invoice_data(response_data: dict) -> None:
    """Display invoice data in a structured format"""
    if not response_data or 'data' not in response_data:
        st.warning("No invoice data found")
        return
        
    data = response_data['data']
    
    # Display page-wise data
    for page_num, page_data in enumerate(data.get('pageWiseData', [])):
        st.subheader(f"📄 Page {page_data.get('page', 'N/A')}")
        
        # Display header information
        if 'headerInfo' in page_data:
            st.markdown("### 📋 Invoice Details")
            header_data = []
            
            for header in page_data['headerInfo']:
                if isinstance(header, dict):
                    value = header.get('value', '')
                    if isinstance(value, dict):
                        value = value.get('value', '')
                    header_data.append({
                        'Field': header.get('key', ''),
                        'Value': value,
                        'Confidence': header.get('confidence', 'N/A'),
                        'Method': header.get('method', 'N/A')
                    })
            
            if header_data:
                df = pd.DataFrame(header_data)
                st.dataframe(
                    df,
                    column_config={
                        "Field": st.column_config.TextColumn("Field", width="medium"),
                        "Value": st.column_config.TextColumn("Value", width="medium"),
                        "Confidence": st.column_config.TextColumn("Confidence", width="small"),
                        "Method": st.column_config.TextColumn("Method", width="small")
                    },
                    use_container_width=True,
                    hide_index=True
                )
                
                # Download button for header data with unique key
                csv_header = df.to_csv(index=False)
                st.download_button(
                    label=f"📥 Download Page {page_num + 1} Headers as CSV",
                    data=csv_header,
                    file_name=f"invoice_headers_page_{page_num + 1}.csv",
                    mime="text/csv",
                    key=f"download_headers_page_{page_num + 1}"
                )
        
        # Display line items/table data
        if 'lineInfo' in page_data and 'lineData' in page_data['lineInfo']:
            st.markdown("### 📊 Line Items")
            line_data = page_data['lineInfo']['lineData']
            if line_data:
                df = pd.DataFrame(line_data)
                st.dataframe(df, use_container_width=True)
                
                # Download button for line items with unique key
                csv = df.to_csv(index=False)
                st.download_button(
                    label=f"📥 Download Page {page_num + 1} Line Items as CSV",
                    data=csv,
                    file_name=f"invoice_line_items_page_{page_num + 1}.csv",
                    mime="text/csv",
                    key=f"download_line_items_page_{page_num + 1}"
                )

def display_raw_json_response(response_data: dict, doc_type: str) -> None:
    """Display and provide download option for raw JSON response"""
    # with st.expander("🔍 View Raw JSON Response"):
        # Pretty print the JSON
    st.json(response_data)
    
    # Convert to pretty JSON string
    json_str = json.dumps(response_data, indent=2)
    
    # Add download button
    st.download_button(
        label="📥 Download Raw JSON",
        data=json_str,
        file_name=f"{doc_type.lower()}_response.json",
        mime="application/json",
        key=f"download_{doc_type.lower()}_json"
    )

def main():
    """Main application"""
    
    # Authentication check
    if not check_authentication():
        login_form()
        return
    
    # Sidebar with user info and logout
    with st.sidebar:
        st.markdown(f"👤 **Welcome, {st.session_state.get('username', 'User')}!**")
        if st.button("🚪 Logout"):
            logout()
        st.markdown("---")

    # Header
    st.markdown('<div class="main-header">🏢 Nekkanti OCR - Document Processor</div>', unsafe_allow_html=True)
    
    # Create tabs for different document types
    doc_type_tab1, doc_type_tab2 = st.tabs(["🏦 Bank Statement", "📄 Invoice"])
    
    with doc_type_tab1:
        st.header("Bank Statement Processing")
        with st.sidebar:
            st.header("📋 Bank Statement Instructions")
            st.markdown("""
            1. **Upload** your bank statement (PDF or image file)
            2. **Wait** while it's processed through our OCR engine
            3. **View** extracted headers and tables
            4. **Download** results in CSV format
            """)
        
        # Initialize session state for bank statement
        if 'current_bank_statement' not in st.session_state:
            st.session_state.current_bank_statement = None
            st.session_state.bank_statement_processed = False
            st.session_state.bank_statement_result = None
        
        uploaded_file = st.file_uploader(
            "Choose a Bank Statement file",
            type=['pdf', 'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif'],
            help="Upload your bank statement in PDF or image format",
            key="bank_statement_uploader"
        )
        
        if uploaded_file is not None:
            # Check if this is a new file
            if st.session_state.current_bank_statement != uploaded_file.name:
                st.session_state.current_bank_statement = uploaded_file.name
                st.session_state.bank_statement_processed = False
                st.session_state.bank_statement_result = None
                
            file_bytes = uploaded_file.read()
            display_file_preview(file_bytes, uploaded_file.name)
            
            if not st.session_state.bank_statement_processed:
                with st.spinner("🔄 Processing bank statement..."):
                    result = process_document(file_bytes, uploaded_file.name, "BANKSTMT")
                    if result:
                        st.session_state.bank_statement_result = result
                        st.session_state.bank_statement_processed = True
                        st.success("✅ Bank statement processed successfully!")
                        st.rerun()
            
            if st.session_state.bank_statement_result:
                st.header("📊 Bank Statement Results")
                
                # Add a refresh button
                if st.button("🔄 Process Again", key="refresh_bank_statement"):
                    st.session_state.bank_statement_processed = False
                    st.session_state.bank_statement_result = None
                    st.rerun()
                
                display_summary_metrics(st.session_state.bank_statement_result)
                
                # Add raw JSON response viewer and download
                # display_raw_json_response(st.session_state.bank_statement_result, "BANKSTMT")
                
                tab1, tab2 = st.tabs(["🎯 Structured View", "📄 Raw Response"])
                
                with tab1:
                    data = st.session_state.bank_statement_result.get('data', {})
                    page_wise_data = data.get('pageWiseData', [])
                    if page_wise_data:
                        for page_data in page_wise_data:
                            st.subheader(f"📄 Page {page_data.get('page', 'N/A')}")
                            # Display headers from the page
                            headers_data = page_data.get('headerInfo', {})
                            display_headers(headers_data)
                            st.markdown("---")
                            
                            # Display table data from the page's lineInfo
                            if 'lineInfo' in page_data and 'lineData' in page_data['lineInfo']:
                                table_data = page_data['lineInfo']['lineData']
                                display_table_data(table_data)
                
                with tab2:
                    display_raw_json_response(st.session_state.bank_statement_result, "BANKSTMT")
    
    with doc_type_tab2:
        st.header("Invoice Processing")
        with st.sidebar:
            st.header("📋 Invoice Instructions")
            st.markdown("""
            1. **Upload** your invoice (PDF or image file)
            2. **Wait** while it's processed through our OCR engine
            3. **View** extracted details and line items
            4. **Download** results in CSV format
            """)
        
        # Initialize session state for invoice
        if 'current_invoice' not in st.session_state:
            st.session_state.current_invoice = None
            st.session_state.invoice_processed = False
            st.session_state.invoice_result = None
        
        uploaded_invoice = st.file_uploader(
            "Choose an Invoice file",
            type=['pdf', 'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif'],
            help="Upload your invoice in PDF or image format",
            key="invoice_uploader"
        )
        
        if uploaded_invoice is not None:
            # Check if this is a new file
            if st.session_state.current_invoice != uploaded_invoice.name:
                st.session_state.current_invoice = uploaded_invoice.name
                st.session_state.invoice_processed = False
                st.session_state.invoice_result = None
                
            invoice_bytes = uploaded_invoice.read()
            display_file_preview(invoice_bytes, uploaded_invoice.name)
            
            if not st.session_state.invoice_processed:
                with st.spinner("🔄 Processing invoice..."):
                    result = process_document(invoice_bytes, uploaded_invoice.name, "INVOICE")
                    if result:
                        st.session_state.invoice_result = result
                        st.session_state.invoice_processed = True
                        st.success("✅ Invoice processed successfully!")
                        st.rerun()
            
            if st.session_state.invoice_result:
                st.header("📊 Invoice Results")
                
                # Add a refresh button
                if st.button("🔄 Process Again", key="refresh_invoice"):
                    st.session_state.invoice_processed = False
                    st.session_state.invoice_result = None
                    st.rerun()
                
                # Add raw JSON response viewer and download
                # display_raw_json_response(st.session_state.invoice_result, "INVOICE")
                
                tab1, tab2 = st.tabs(["🎯 Structured View", "📄 Raw Response"])
                
                with tab1:
                    display_invoice_data(st.session_state.invoice_result)
                
                with tab2:
                    display_raw_json_response(st.session_state.invoice_result, "INVOICE")

if __name__ == "__main__":
    main() 