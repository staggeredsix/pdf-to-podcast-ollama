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

# Install core dependencies with exact versions to prevent conflicts
RUN pip3 install \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    redis==5.0.1 \
    httpx==0.25.2 \
    soundfile==0.12.1 \
    numpy==2.2.4

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

# Install Dia dependencies while avoiding torch reinstallation
RUN pip3 install \
    transformers>=4.30.0 \
    accelerate>=0.20.0 \
    scikit-learn \
    datasets \
    scipy \
    einops \
    safetensors \
    huggingface-hub \
    librosa

# Create constraints file to prevent torch reinstallation
RUN echo "torch==$(python3 -c 'import torch; print(torch.__version__)')" > /tmp/constraints.txt && \
    echo "torchaudio==$(python3 -c 'import torchaudio; print(torchaudio.__version__)')" >> /tmp/constraints.txt && \
    echo "torchvision==$(python3 -c 'import torchvision; print(torchvision.__version__)')" >> /tmp/constraints.txt && \
    echo "numpy==$(python3 -c 'import numpy; print(numpy.__version__)')" >> /tmp/constraints.txt && \
    cat /tmp/constraints.txt

# Try Method 1: Direct pip install from GitHub with constraints
RUN pip3 install git+https://github.com/nari-labs/dia.git --constraint /tmp/constraints.txt && \
    python3 -c "import dia; from dia.model import Dia; print('SUCCESS: Method 1 worked')" || \
    echo "Method 1 failed, trying method 2..."

# Method 2: Clone and manual install if Method 1 failed
RUN if ! python3 -c "import dia" 2>/dev/null; then \
        echo "Trying method 2: Clone and install manually" && \
        git clone https://github.com/nari-labs/dia.git /tmp/dia && \
        cd /tmp/dia && \
        pip3 install -e . --constraint /tmp/constraints.txt && \
        rm -rf /tmp/dia; \
    fi

# Method 3: Copy source directly if Methods 1 and 2 failed
RUN if ! python3 -c "import dia" 2>/dev/null; then \
        echo "Trying method 3: Direct copy" && \
        git clone https://github.com/nari-labs/dia.git /opt/dia && \
        SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])") && \
        if [ -d "/opt/dia/dia" ]; then \
            cp -r /opt/dia/dia $SITE_PACKAGES/ && \
            chmod -R 755 $SITE_PACKAGES/dia; \
        fi && \
        export PYTHONPATH="/opt/dia:$PYTHONPATH"; \
    fi
    
RUN if ! python3 -c "import dia" 2>/dev/null; then \
        echo "Trying method 4: Sledgehammer Install" && \
        pip3 install git+https://github.com/nari-labs/dia.git \
        --force-reinstall \
        --no-deps \
        --upgrade \
        --disable-pip-version-check

# Final verification
RUN python3 -c "import dia; print('Final check: dia imported successfully')" && \
    python3 -c "from dia.model import Dia; print('Final check: Dia class imported successfully')"

# Add shared package
COPY shared /shared
RUN pip3 install /shared

# Copy service code
WORKDIR /app
COPY services/TTSService /app

# Set environment variables
ENV PYTHONPATH="/opt/dia:$PYTHONPATH"

# Expose port
EXPOSE 8889

# Start the TTS service
CMD ["python3", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8889"]