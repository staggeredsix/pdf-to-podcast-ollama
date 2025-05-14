FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=0

# System packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev \
        ffmpeg libsndfile1 \
        git curl unzip build-essential && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip and install wheel
RUN pip install --upgrade pip wheel setuptools

# Install PyTorch with CUDA support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install core dependencies with versions
RUN pip install \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    redis==5.0.1 \
    httpx==0.25.2 \
    soundfile==0.12.1 \
    numpy==1.24.3

# Install OpenTelemetry packages with versions
RUN pip install \
    opentelemetry-api==1.21.0 \
    opentelemetry-sdk==1.21.0 \
    opentelemetry-exporter-otlp==1.21.0 \
    opentelemetry-instrumentation==0.42b0 \
    opentelemetry-instrumentation-fastapi==0.42b0 \
    opentelemetry-instrumentation-redis==0.42b0 \
    opentelemetry-instrumentation-httpx==0.42b0 \
    opentelemetry-instrumentation-requests==0.42b0 \
    opentelemetry-instrumentation-urllib3==0.42b0 \
    opentelemetry-util-http==0.42b0

# Install Dia dependencies separately
RUN pip install \
    transformers>=4.30.0 \
    accelerate>=0.20.0 \
    scikit-learn \
    datasets \
    scipy \
    einops \
    safetensors \
    huggingface-hub

# Clone and install Dia
RUN git clone https://github.com/nari-labs/dia.git /tmp/dia && \
    cd /tmp/dia && \
    pip install -e . && \
    rm -rf /tmp/dia

# Add shared package
COPY shared /shared
RUN pip install /shared

# Copy service code
WORKDIR /app
COPY services/TTSService /app

# Expose port
EXPOSE 8889

# Start the TTS service
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8889"]