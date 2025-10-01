# CUDA-enabled runtime to allow GPU for generator if available
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TRANSFORMERS_NO_TORCHVISION=1

# System deps + Python 3.10 (Ubuntu 22.04 ships Python 3.10)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    ca-certificates \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

# Install PyTorch CUDA 12.1 wheels explicitly; FAISS will be CPU
# Adjust torch version if needed to match your environment.
RUN pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.5.1

# Core Python deps for the API
RUN pip install \
    transformers==4.45.2 \
    sentence-transformers==2.6.1 \
    faiss-cpu==1.8.0 \
    fastapi==0.115.0 \
    uvicorn[standard]==0.30.6 \
    pandas==2.2.2 \
    numpy==1.26.4 \
    pydantic[dotenv]==2.8.2 \
    argostranslate==1.9.6 \
    safetensors==0.4.5 \
    bert-score==0.3.13 \
    rouge-score==0.1.2 \
    nltk==3.9.1 \
    pyarrow==16.1.0 \
    fastparquet==2024.5.0

# Create app directory and copy source
WORKDIR /app
COPY app/ /app/app/

# Expose volumes via symlinks: app expects /app/data and /app/models; users will mount /data and /models
RUN mkdir -p /app/data /app/models \
    && ln -sfn /data /app/data \
    && ln -sfn /models /app/models

# Expose API port
EXPOSE 8080

# Work inside module dir so 'uvicorn main:app' works
WORKDIR /app/app

# Default CMD per requirement
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
