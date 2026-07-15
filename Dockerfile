# Base image with CUDA, CUDNN, and PyTorch pre-installed
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# Set non-interactive mode for apt
ENV DEBIAN_FRONTEND=noninteractive

# Set environment variables for compilation and CUDA
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
ENV MAX_JOBS=4

# Set working directory inside container
WORKDIR /workspace/adfusionR&D

# Install system dependencies (FFmpeg, OpenCV dependencies, development tools)
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

# Upgrade pip and install uv for ultra-fast python package installation
RUN pip install --no-cache-dir --upgrade pip uv

# Copy only the configuration files first to cache dependencies layers
COPY pyproject.toml requirements.txt ./

# Install python dependencies globally inside the container
RUN uv pip install --no-cache-dir --system -r requirements.txt

# Copy the rest of the application files (including third_party code)
COPY . .

# Install the third_party sam3 package in editable mode
RUN uv pip install --no-cache-dir --system -e third_party/sam3

# Install optional optimized GPU kernels (ninja, einops, flash-attn)
# Note: flash-attn can take time to build, we pre-install setup tools
RUN uv pip install --no-cache-dir --system ninja einops

# Expose Jupyter notebook port and SSH port
EXPOSE 8888 22

# Default action is to run system verification
CMD ["python", "utils/verify_env.py"]
