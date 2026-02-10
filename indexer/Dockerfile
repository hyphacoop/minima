# GPU-enabled minima indexer
# Uses NVIDIA CUDA runtime for GPU inference
FROM nvidia/cuda:12.1.1-base-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11 and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

WORKDIR /usr/src/app

# Install PyTorch with CUDA support before other requirements
# This ensures GPU-enabled torch is used instead of CPU-only
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121

# Pre-download the embedding model into the image
ARG EMBEDDING_MODEL_ID
RUN pip install --no-cache-dir huggingface_hub && \
    python -c "from huggingface_hub import snapshot_download; snapshot_download('$EMBEDDING_MODEL_ID')"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    apt-get purge -y --auto-remove python3.11-dev && \
    rm -rf /var/lib/apt/lists/*
COPY . .

ENV PORT=8000
ENV CURRENT_HOST=0.0.0.0
ENV WORKERS=1

CMD ["sh", "-c", "uvicorn app:app --loop asyncio --reload --workers ${WORKERS} --host $CURRENT_HOST --port $PORT --proxy-headers"]
