#!/bin/bash
# ==============================================================================
# ADFUSION AUTOMATED SPRINT 1 VALIDATION SCRIPT
# This script runs the entire diagnostics, download, and SAM 3 test sequence.
# ==============================================================================

set -eo pipefail

echo "=============================================================="
echo "          STARTING AUTOMATED SPRINT 1 VALIDATION"
echo "=============================================================="

# 1. Environment Verification
echo "[*] Step 1/4: Running Environment Verification..."
python utils/verify_env.py

# 2. Checkpoints Verification
echo "[*] Step 2/4: Verifying Checkpoints..."
python utils/download_models.py

# 3. Object Detection Execution
echo "[*] Step 3/4: Running SAM 3 Object Detection (detect stage)..."
python main.py --stage detect

# 4. Object Tracking Execution
echo "[*] Step 4/4: Running SAM 3.1 Object Tracking (track stage)..."
python main.py --stage track

echo "=============================================================="
echo "      AUTOMATED SPRINT 1 VALIDATION COMPLETED SUCCESSFULLY"
echo "=============================================================="
