# Base image: PyTorch 2.5.1 + CUDA 12.4 developer tools
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive

# CUDA compilation environment for GroundingDINO custom ops
ENV CUDA_HOME=/usr/local/cuda
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
ENV MAX_JOBS=4

WORKDIR /workspace/adfusionR&D

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    zip \
    unzip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install uv
RUN pip install --no-cache-dir --upgrade pip uv

# Install core Python dependencies first (cacheable layer)
COPY pyproject.toml requirements.txt ./
RUN uv pip install --no-cache-dir --system -r requirements.txt

# Copy full project (including third_party/)
COPY . .

# Install SAM2 from third_party in editable mode
RUN uv pip install --no-build-isolation --no-cache-dir --system -e third_party/sam2

# Install GroundingDINO from third_party in editable mode (compiles CUDA ops)
RUN uv pip install --no-build-isolation --no-cache-dir --system -e third_party/GroundingDINO

# Expose Jupyter and SSH ports
EXPOSE 8888 22

# Default: run environment diagnostics
CMD ["python", "utils/verify_env.py"]
