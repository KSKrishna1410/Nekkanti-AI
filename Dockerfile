# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for production
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    procps \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with optimizations
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set ownership to non-root user
RUN chown -R appuser:appuser /app

# Create directories for temporary files with proper permissions
RUN mkdir -p /tmp/ocr_temp /tmp/bankstmt_temp /tmp/ocr_outputs_reconstructed && \
    chmod 755 /tmp/ocr_temp /tmp/bankstmt_temp /tmp/ocr_outputs_reconstructed && \
    chown -R appuser:appuser /tmp/ocr_temp /tmp/bankstmt_temp /tmp/ocr_outputs_reconstructed

# Expose port 8888
EXPOSE 8888

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8888/health || exit 1

# Command to run the application
CMD ["python", "main_modular.py", "serve", "--host", "0.0.0.0", "--port", "8888"]