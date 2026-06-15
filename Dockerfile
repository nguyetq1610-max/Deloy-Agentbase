FROM python:3.11-slim

LABEL maintainer="ZaloPay CS Team"
LABEL description="Z-Agent One — AI Customer Support Agent for ZaloPay"

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    tesseract-ocr \
    tesseract-ocr-vie \
    tesseract-ocr-eng \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY agent.py .
COPY .env.example .env.example

# Data and history directories
RUN mkdir -p data history

# Copy canned responses if present
COPY data/ data/

EXPOSE 8080

ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO
ENV CSV_PATH=data/canned_responses.csv
ENV HISTORY_DIR=history

# Default: API server mode
CMD ["python", "agent.py", "--server", "--port", "8080"]
