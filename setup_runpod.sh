#!/bin/bash
# ==============================================================================
# ADFUSION RUNPOD INITIALIZATION & SETUP SCRIPT (Idempotent)
# This script sets up a fresh RunPod Linux GPU instance for VPP execution.
# ==============================================================================

set -eo pipefail

echo "=============================================================="
echo "          STARTING RUNPOD PLATFORM INITIALIZATION"
echo "=============================================================="

# 1. Update Apt-Get & Install System Packages
echo "[*] Updating apt repositories..."
apt-get update -y

echo "[*] Installing critical system dependencies (FFmpeg, OpenCV libs)..."
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

# 2. Upgrade Python Package Installer Stack
echo "[*] Upgrading pip..."
python -m pip install --upgrade --no-cache-dir pip --break-system-packages || python -m pip install --upgrade --no-cache-dir pip

echo "[*] Installing uv for ultra-fast pip execution..."
python -m pip install --no-cache-dir uv --break-system-packages || python -m pip install --no-cache-dir uv

# 3. Initialize third_party directories if not already populated
echo "[*] Initializing third_party framework layout..."
mkdir -p third_party/depth_anything3
mkdir -p third_party/sam3d
mkdir -p third_party/wan22_vace

# 4. Install python dependencies with uv if available, falling back to standard pip
if command -v uv &> /dev/null; then
    echo "[*] Installing dependencies from requirements.txt via uv..."
    uv pip install --system --no-cache-dir -r requirements.txt
    
    echo "[*] Installing third_party/sam3 in editable mode via uv..."
    uv pip install --system --no-cache-dir -e third_party/sam3
else
    echo "[*] Installing dependencies from requirements.txt via pip..."
    pip install --no-cache-dir -r requirements.txt --break-system-packages || pip install --no-cache-dir -r requirements.txt
    
    echo "[*] Installing third_party/sam3 in editable mode via pip..."
    pip install --no-cache-dir -e third_party/sam3 --break-system-packages || pip install --no-cache-dir -e third_party/sam3
fi

# 5. Verify System Environment
echo "[*] Running system diagnostics..."
python utils/verify_env.py

# 6. Model Download Check
echo "[*] Running model downloader to verify checkpoints..."
python utils/download_models.py

echo "=============================================================="
echo "     RUNPOD INITIALIZATION COMPLETE. READY FOR PIPELINE."
echo "=============================================================="
