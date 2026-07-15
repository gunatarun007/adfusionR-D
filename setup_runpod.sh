#!/bin/bash
# ==============================================================================
# ADFUSION RUNPOD INITIALIZATION & SETUP SCRIPT (Idempotent)
# Backend: GroundingDINO + SAM2 — fully open-source, no HF token required
# ==============================================================================

set -eo pipefail

echo "=============================================================="
echo "       STARTING RUNPOD PLATFORM INITIALIZATION"
echo "       Backend: GroundingDINO + SAM2"
echo "=============================================================="

# 1. System packages
echo "[*] Updating apt repositories..."
apt-get update -y

echo "[*] Installing system dependencies..."
apt-get install -y --no-install-recommends \
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
    unzip

# 2. Set CUDA_HOME for GroundingDINO CUDA extension compilation
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
echo "[*] CUDA_HOME = $CUDA_HOME"

# 3. Upgrade pip / install uv
echo "[*] Upgrading pip..."
python -m pip install --upgrade --no-cache-dir pip --break-system-packages || \
    python -m pip install --upgrade --no-cache-dir pip

echo "[*] Installing uv..."
python -m pip install --no-cache-dir uv --break-system-packages || \
    python -m pip install --no-cache-dir uv

# 4. Initialise third_party directories
echo "[*] Initialising third_party layout..."
mkdir -p third_party/depth_anything3
mkdir -p third_party/sam3d
mkdir -p third_party/wan22_vace

# 5. Core python dependencies
if command -v uv &> /dev/null; then
    echo "[*] Installing requirements via uv..."
    uv pip install --system --no-cache-dir -r requirements.txt
else
    echo "[*] Installing requirements via pip..."
    pip install --no-cache-dir -r requirements.txt \
        --break-system-packages || \
        pip install --no-cache-dir -r requirements.txt
fi

# 6. Install third_party/sam2 in editable mode
echo "[*] Installing SAM2 from third_party/sam2..."
if command -v uv &> /dev/null; then
    uv pip install --system --no-build-isolation --no-cache-dir -e third_party/sam2
else
    pip install --no-build-isolation --no-cache-dir -e third_party/sam2 \
        --break-system-packages || \
        pip install --no-build-isolation --no-cache-dir -e third_party/sam2
fi

# 7. Install third_party/GroundingDINO in editable mode
# Requires CUDA_HOME set correctly to compile custom CUDA extensions
echo "[*] Installing GroundingDINO from third_party/GroundingDINO..."
if command -v uv &> /dev/null; then
    uv pip install --system --no-build-isolation --no-cache-dir -e third_party/GroundingDINO
else
    pip install --no-build-isolation --no-cache-dir -e third_party/GroundingDINO \
        --break-system-packages || \
        pip install --no-build-isolation --no-cache-dir -e third_party/GroundingDINO
fi

# 8. Download model checkpoints (public URLs, no token needed)
echo "[*] Downloading model checkpoints..."
python utils/download_models.py

# 9. Verify environment
echo "[*] Running environment diagnostics..."
python utils/verify_env.py

echo "=============================================================="
echo "    RUNPOD INITIALIZATION COMPLETE. READY FOR PIPELINE."
echo "=============================================================="
