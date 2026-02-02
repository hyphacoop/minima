# GPU-enabled minima indexer
# Uses NVIDIA CUDA runtime for GPU inference
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11 and dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

WORKDIR /usr/src/app
RUN pip install --upgrade pip

# Install PyTorch with CUDA support before other requirements
# This ensures GPU-enabled torch is used instead of CPU-only
RUN pip install torch --index-url https://download.pytorch.org/whl/cu121

ARG EMBEDDING_MODEL_ID

RUN pip install huggingface_hub
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('$EMBEDDING_MODEL_ID')"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

ENV PORT 8000
ENV CURRENT_HOST 0.0.0.0
ENV WORKERS 1

CMD ["sh", "-c", "uvicorn app:app --loop asyncio --reload --workers ${WORKERS} --host $CURRENT_HOST --port $PORT --proxy-headers"]
