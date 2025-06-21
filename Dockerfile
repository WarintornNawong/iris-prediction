FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY predict.py .
COPY rf_iris.onnx .

# Expose FastAPI default port
EXPOSE 8080

# Run the FastAPI app
CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8080"]
