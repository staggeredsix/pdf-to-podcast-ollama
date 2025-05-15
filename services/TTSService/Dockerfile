FROM nvidia/cuda:12.6.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=0

# System packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev python3-venv \
        ffmpeg libsndfile1 \
        git curl unzip build-essential \
        pkg-config libffi-dev && \
    rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN pip3 install --upgrade pip wheel setuptools

# Install PyTorch with CUDA 12.8 support for Blackwell
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Verify PyTorch installation
RUN python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Install core dependencies
RUN pip3 install \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    redis==5.0.1 \
    httpx==0.25.2 \
    soundfile==0.12.1 \
    numpy==1.24.3

# Install OpenTelemetry packages
RUN pip3 install \
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

# Install Dia dependencies (without torch)
RUN pip3 install \
    transformers>=4.30.0 \
    accelerate>=0.20.0 \
    scikit-learn \
    datasets \
    scipy \
    einops \
    safetensors \
    huggingface-hub \
    librosa \
    descript-audio-codec

# Install Dia while ignoring version conflicts - single command approach
RUN pip3 install git+https://github.com/nari-labs/dia.git \
    --force-reinstall \
    --no-deps \
    --upgrade \
    --disable-pip-version-check

# Verify installation
RUN python3 -c "import dia; print('Dia imported successfully')" && \
    python3 -c "from dia.model import Dia; print('Dia.model imported successfully')" && \
    python3 -c "import torch; print(f'Final PyTorch version: {torch.__version__}')"

# Add shared package
COPY shared /shared
RUN pip3 install /shared

# Copy service code
WORKDIR /app
COPY services/TTSService /app

# Expose port
EXPOSE 8889

# Start the TTS service
CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8889"]