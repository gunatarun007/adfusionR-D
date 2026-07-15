#!/bin/bash
# ==============================================================================
# ADFUSION AUTOMATED SPRINT 1 VALIDATION SCRIPT
# GroundingDINO + SAM2 backend — no HF token required
# ==============================================================================

set -eo pipefail

echo "=============================================================="
echo "       STARTING AUTOMATED SPRINT 1 VALIDATION"
echo "       Backend: GroundingDINO + SAM2"
echo "=============================================================="

# 1. Environment verification
echo "[*] Step 1/4: Running Environment Verification..."
python utils/verify_env.py

# 2. Checkpoint downloads
echo "[*] Step 2/4: Verifying / Downloading Checkpoints..."
python utils/download_models.py

# 3. Object detection
echo "[*] Step 3/4: Running GroundingDINO + SAM2 Object Detection..."
python main.py --stage detect

# 4. Object tracking
echo "[*] Step 4/4: Running SAM2 Video Tracking..."
python main.py --stage track

echo "=============================================================="
echo "   AUTOMATED SPRINT 1 VALIDATION COMPLETED SUCCESSFULLY"
echo "=============================================================="
