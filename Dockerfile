FROM python:3.11-slim

WORKDIR /app

# System libraries required by OpenCV, EasyOCR and image processing
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Increase download timeout because PyTorch packages are large.
# Install CPU-only PyTorch first so Docker does not download CUDA packages.
RUN python -m pip install --upgrade pip \
    && python -m pip install \
        --default-timeout=1000 \
        --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision \
    && python -m pip install \
        --default-timeout=1000 \
        --no-cache-dir \
        -r requirements.txt

COPY app ./app
COPY data ./data
COPY README.md .
COPY requirements.txt .

RUN mkdir -p /app/data/photos /app/credentials

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]