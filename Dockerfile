FROM python:3.10-slim

# Install system dependencies, including those for lxml
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libxml2-dev \
        libxslt-dev \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy only requirements first for layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Default command
CMD ["python", "main.py"]