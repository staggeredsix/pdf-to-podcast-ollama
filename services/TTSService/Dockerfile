FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=0

# System packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev python3-venv \
        ffmpeg libsndfile1 \
        git curl unzip build-essential \
        pkg-config libffi-dev \
        git-lfs && \
    rm -rf /var/lib/apt/lists/*

# Initialize git-lfs
RUN git lfs install

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

# Install compatible versions of protobuf and related packages
RUN pip3 install protobuf==4.25.3 grpcio==1.59.2 grpcio-status==1.59.2 googleapis-common-protos==1.61.0

# Verify protobuf installation
RUN python3 -c "import google.protobuf; print('Protobuf version:', google.protobuf.__version__)"

# Install OpenTelemetry packages
RUN pip3 install \
    opentelemetry-api==1.21.0 \
    opentelemetry-sdk==1.21.0 \
    opentelemetry-proto==1.21.0 \
    opentelemetry-exporter-otlp-proto-grpc==1.21.0

# Install OpenTelemetry instrumentation packages
RUN pip3 install \
    opentelemetry-instrumentation==0.42b0 \
    opentelemetry-instrumentation-fastapi==0.42b0 \
    opentelemetry-instrumentation-redis==0.42b0 \
    opentelemetry-instrumentation-httpx==0.42b0 \
    opentelemetry-instrumentation-requests==0.42b0 \
    opentelemetry-instrumentation-urllib3==0.42b0 \
    opentelemetry-util-http==0.42b0

# Test OpenTelemetry import
RUN python3 -c "from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter; print('OpenTelemetry import successful')"

# Install ML dependencies
RUN pip3 install \
    transformers>=4.30.0 \
    accelerate>=0.20.0 \
    safetensors \
    huggingface-hub \
    descript-audio-codec \
    librosa

# Clone the Hugging Face model repository
WORKDIR /tmp
RUN git clone https://huggingface.co/nari-labs/Dia-1.6B dia-model || echo "HF clone failed"

# Clone the GitHub repository as backup
RUN git clone https://github.com/nari-labs/dia.git dia-github || echo "GitHub clone failed"

# Create application directory
WORKDIR /app
RUN mkdir -p /app/dia_model /app/dia_hf_repo /app/dia_github

# Copy HuggingFace repository if it exists
RUN if [ -d "/tmp/dia-model" ]; then \
        cp -r /tmp/dia-model/* /app/dia_hf_repo/; \
        echo "Copied HuggingFace repo"; \
    fi

# Copy GitHub repository if it exists
RUN if [ -d "/tmp/dia-github" ]; then \
        cp -r /tmp/dia-github/* /app/dia_github/; \
        if [ -d "/tmp/dia-github/dia" ]; then \
            cp -r /tmp/dia-github/dia/* /app/dia_model/; \
        fi; \
        echo "Copied GitHub repo"; \
    fi

# Create __init__.py files
RUN touch /app/dia_model/__init__.py
RUN touch /app/dia_hf_repo/__init__.py

# Try installing Dia package (but don't fail if it doesn't work)
RUN pip3 install git+https://github.com/nari-labs/dia.git --no-deps --force-reinstall || echo "Package install failed, using extracted files"

# List what we have
RUN echo "=== Available files ===" && \
    ls -la /app/dia_model/ && \
    ls -la /app/dia_hf_repo/ && \
    ls -la /app/dia_github/

# Add shared package
COPY shared /shared
RUN pip3 install /shared

# Copy service code
COPY services/TTSService /app/

# Set Python path
ENV PYTHONPATH="/app/dia_model:/app/dia_hf_repo:/app/dia_github:/app:$PYTHONPATH"

# Clean up temp files
RUN rm -rf /tmp/dia-model /tmp/dia-github

# Expose port
EXPOSE 8889

# Start the TTS service
CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8889"]