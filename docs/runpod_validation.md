# RunPod Validation Guide (Sprint 1)

This document provides a step-by-step procedure to validate the AdFusion VPP research pipeline on a real NVIDIA Linux GPU instance on RunPod.

---

## 1. Environment Requirements & Recommendations

### A. Recommended RunPod GPU
*   **NVIDIA L40S (48GB VRAM)**: Best balance of availability, cost, and memory. Easily handles torch compiles and tracking memory buckets.
*   **NVIDIA A100 SXM4 (80GB VRAM)**: Recommended for high-throughput batching, training, and multi-object tracking.
*   **NVIDIA RTX 4090 (24GB VRAM)**: Smallest consumer GPU model recommended.
*   *Note*: Ensure you select a **secure cloud** or **community cloud** instance with a CUDA-enabled base system.

### B. Recommended Docker Image
*   **Official PyTorch Dev Image**: Use the official PyTorch devel template on RunPod:
    `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel`
    *(This image contains pre-installed CUDA compilers required for compiling Triton and optimized attention kernels).*

### C. Recommended Disk Space
*   **Container Disk**: 15 GB
*   **Volume Disk**: 40 GB (required for storing large image frames, checkpoints, and output files).

---

## 2. Environment Variables & Hugging Face Gated Access

Because Meta SAM 3 and SAM 3.1 model checkpoints are **gated** on Hugging Face, you must accept Meta's terms and authenticate:

1.  Log in to your Hugging Face account.
2.  Visit [huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3) and click **Accept Terms / Access Repository**.
3.  Visit [huggingface.co/facebook/sam3.1](https://huggingface.co/facebook/sam3.1) and accept the terms there as well.
4.  Generate a **Read Access Token** from your HF Settings -> Access Tokens page.
5.  When launching your RunPod GPU instance, configure the following environment variable:
    ```bash
    export HF_TOKEN="your_hf_access_token_here"
    ```

---

## 3. Command Execution Sequence

Once inside the RunPod terminal (accessible via SSH or the Jupyter web terminal):

```bash
# 1. Clone the project repository
git clone https://github.com/your-org/adfusionR&D.git
cd adfusionR&D

# 2. Export Hugging Face token
export HF_TOKEN="your_huggingface_read_token"

# 3. Mark scripts as executable
chmod +x setup_runpod.sh
chmod +x scripts/run_validation.sh

# 4. Initialize and install system/python packages
./setup_runpod.sh

# 5. Run the automated validation sequence
./scripts/run_validation.sh
```

---

## 4. Expected Outputs & Verification Checkpoints

Upon successful execution of `./scripts/run_validation.sh`, verify the following outputs:

### A. Environment Check
`verify_env.py` output should display `[SUCCESS] Environment verification PASSED` with active GPU configurations and Triton loaded.

### B. Checkpoint Downloads
The directory `models/` must contain:
*   `sam3.pt` (3.45 GB)
*   `sam3.1_multiplex.pt` (3.50 GB)

### C. Detection Stage Output
Running `python main.py --stage detect` should output:
*   `cache/masks/mask_0001.png` to `mask_0150.png` (8-bit binary masks showing target segments).
*   Green overlay images in `cache/debug/detect_0001.png` to `detect_0150.png`.

### D. Tracking Stage Output
Running `python main.py --stage track` should output:
*   `cache/tracks/tracks.json` containing object tracking trajectories.
*   Updated binary masks in `cache/masks/`.
*   Blue-ish boundary tracking overlays in `cache/debug/track_0001.png` to `track_0150.png`.

---

## 5. Common Failure Cases & Recovery Steps

### A. Error: `ModuleNotFoundError: No module named 'triton'`
*   **Cause**: Running on Windows or an incompatible Linux setup without CUDA developer tools.
*   **Recovery**: Verify that you are running inside a Linux Docker container built from the PyTorch devel template. If needed, reinstall `triton` manually using `pip install triton --break-system-packages`.

### B. Error: `GatedRepoError` or `401 Unauthorized` during model download
*   **Cause**: Gated terms not accepted on Hugging Face, or `HF_TOKEN` was omitted or invalid.
*   **Recovery**: Visit the Hugging Face links above, click accept, and verify that `export HF_TOKEN="your_token"` is executed in the active terminal session. Run `python utils/download_models.py` again to check.

### C. Error: `CUDA Out Of Memory` (OOM) during video tracking
*   **Cause**: High batch sizes or large inputs on small VRAM GPUs.
*   **Recovery**: By default, SAM 3.1 Multiplexing operates inside a 4.8 GB VRAM envelope. If an OOM occurs, ensure no other processes are consuming VRAM (run `nvidia-smi` to audit active processes) or reduce `max_num_objects` in `config.yaml`.
