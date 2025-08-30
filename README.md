# Invoice Extraction API

A FastAPI-based service that extracts structured data from PDF invoices and images using Google's Gemini AI or OpenAI GPT-4o, with advanced OCR support and token usage tracking.

## Features

- **Multi-AI Provider Support**: Choose between Gemini AI and OpenAI GPT-4o
- **Multi-Format Support**: Upload PDF invoices, images (PNG, JPG, TIFF, BMP) via REST API
- **Scanned Document OCR**: Handles scanned PDFs and images using PaddleOCR
- **Structured Data Extraction**: Extracts header fields and line items in JSON format
- **GST Invoice Support**: Optimized for Indian GST invoices with GSTIN, HSN/SAC codes
- **Token Usage Tracking**: Monitors AI API usage and calculates costs for both providers
- **Beautiful Streamlit UI**: Interactive web interface with provider selection and invoice-style display
- **CLI Interface**: Command-line tools for direct file processing
- **Error Handling**: Comprehensive error handling and validation

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd invoice-extraction-api
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your Gemini API key:
```bash
export GEMINI_API_KEY="your-gemini-api-key-here"
```

## Usage

### Option 1: Streamlit UI (Recommended)

Launch the interactive web interface:

```bash
# Using the launcher script
python run_streamlit.py run

# Or directly with streamlit
streamlit run streamlit_app.py
```

The Streamlit UI provides:
- 📁 Drag-and-drop PDF upload
- 🎨 Beautiful invoice-style data display
- 📊 Interactive data tables
- 💾 Download as CSV or JSON
- 📈 Usage statistics and cost tracking

### Option 2: API Server

```bash
# Using the CLI
python main.py serve --host 0.0.0.0 --port 8000

# Or directly with uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000
```

### API Endpoints

#### POST /extract-invoice
Upload a PDF invoice for data extraction.

**Request**: Multipart form data with PDF file
**Response**: JSON with extracted data and token usage

```json
{
  "header_fields": {
    "supplier_name": "ABC Company Ltd",
    "supplier_gstin": "29ABCDE1234F1Z5",
    "invoice_number": "INV-2024-001",
    "invoice_date": "2024-01-15",
    "total_amount": "1180.00",
    ...
  },
  "invoice_table": [
    {
      "description": "Product A",
      "hsn_sac": "1234",
      "quantity": "2",
      "rate": "500.00",
      "total_amount": "1000.00",
      ...
    }
  ],
  "token_usage": {
    "input_tokens": 1500,
    "output_tokens": 800,
    "total_cost_usd": 0.0775
  },
  "errors": []
}
```

#### GET /health
Health check endpoint

#### GET /docs
Interactive API documentation (Swagger UI)

### CLI Usage

#### Extract data from a PDF or image file:
```bash
python main.py extract invoice.pdf --output result.json
python main.py extract scanned_invoice.png --output result.json
```

#### Start the server with custom options:
```bash
python main.py serve --host 127.0.0.1 --port 9000 --reload
```

## Configuration

### Environment Variables

- `GEMINI_API_KEY`: Your Google Gemini API key (required)

### Token Pricing

Current token pricing (configurable in `main.py`):
- Input tokens: $0.000025 per token
- Output tokens: $0.000050 per token

## API Response Structure

### Header Fields
- `supplier_name`: Name of the supplier/vendor
- `supplier_gstin`: Supplier's GSTIN number
- `supplier_address`: Supplier's address
- `buyer_name`: Name of the buyer
- `buyer_gstin`: Buyer's GSTIN number
- `buyer_address`: Buyer's address
- `invoice_number`: Invoice number
- `invoice_date`: Invoice date
- `subtotal`: Subtotal amount
- `total_tax`: Total tax amount
- `total_amount`: Final total amount
- `currency`: Currency code
- `due_date`: Payment due date
- `po_number`: Purchase order number

### Line Items
- `description`: Item/service description
- `hsn_sac`: HSN/SAC code
- `quantity`: Quantity
- `unit`: Unit of measurement
- `rate`: Unit rate
- `discount`: Discount amount
- `taxable_value`: Taxable value
- `cgst_rate`: CGST rate percentage
- `cgst_amount`: CGST amount
- `sgst_rate`: SGST rate percentage
- `sgst_amount`: SGST amount
- `igst_rate`: IGST rate percentage
- `igst_amount`: IGST amount
- `total_amount`: Line item total

## Error Handling

The API handles various error conditions:
- Invalid file formats (non-PDF)
- Empty files
- PDF parsing errors
- Gemini API errors
- JSON parsing errors

Errors are returned in the `errors` array of the response.

## Testing

Test the API using curl:

```bash
curl -X POST "http://localhost:8000/extract-invoice" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your-invoice.pdf"
```

## Dependencies

### Core Dependencies
- **FastAPI**: Web framework for API
- **Streamlit**: Interactive web UI framework
- **PyPDF2**: PDF text extraction for readable PDFs
- **google-generativeai**: Gemini AI integration
- **Pydantic**: Data validation and serialization
- **Pandas**: Data manipulation and CSV export
- **Click**: CLI interface
- **Uvicorn**: ASGI server

### OCR Dependencies (for scanned documents)
- **PaddleOCR**: Advanced OCR engine
- **PaddlePaddle**: Deep learning framework
- **OpenCV**: Computer vision library
- **PyMuPDF**: PDF processing
- **ReportLab**: PDF generation
- **Pillow**: Image processing

## Troubleshooting

### Token Usage Showing 0
The application has been updated to use `gemini-1.5-pro` model for better token tracking. If you still see 0 tokens:
1. Ensure you're using a valid Gemini API key
2. Check your API quota and billing settings
3. Try with a smaller document first

### OCR Dependencies Installation Issues
If you encounter issues installing OCR dependencies:
```bash
# For macOS with Apple Silicon
pip install paddlepaddle -i https://pypi.tuna.tsinghua.edu.cn/simple

# For Windows/Linux
pip install paddlepaddle

# If OpenCV installation fails
pip install opencv-python-headless
```

## License

MIT License