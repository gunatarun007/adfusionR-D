# AdFusion VPP Research Pipeline

AdFusion is a modular, high-performance research framework for AI-powered Virtual Product Placement (VPP). Designed as an "AI VFX Operating System," it allows rapid prototyping and evaluation of state-of-the-art computer vision models (such as SAM 3, Depth Anything 3, SAM3D, and video generator models like Wan2.2).

---

## 1. Linux-First Architecture Decision

To optimize training and inference performance on state-of-the-art models, AdFusion follows a **Linux-first execution model**. 

*   **Windows Environment**: Development workstation only. Used for code editing, version control, documentation, and running unit tests that do not depend on heavy deep learning models.
*   **RunPod Linux GPU Environment**: Target runtime execution environment. Used for PyTorch, Triton GPU, CUDA kernel compilation, and full end-to-end pipeline execution (including SAM 3.0/3.1, Depth Anything 3, and Wan2.2).

---

## 2. RunPod Platform Workflow

Deploying and running experiments follows an automated lifecycle:

1.  **Develop Locally**: Write and modify pipeline stages on a local workstation.
2.  **Push to GitHub**: Commit code changes and push to the remote repository.
3.  **Launch Pod**: Spin up a RunPod GPU instance (NVIDIA L40S, A100, or H100) running the PyTorch CUDA 12.4 template.
4.  **Clone & Authenticate**: Clone the repository and export a valid Hugging Face gated token (`HF_TOKEN`).
5.  **Provision**: Run `./setup_runpod.sh` to install all system dependencies, python libraries, and cache model weights.
6.  **Run Pipeline**: Run `./scripts/run_validation.sh` to execute environment checks and run detection/tracking test frames.
7.  **Download Metrics & Video**: Inspect execution timing logs and output videos (`output/final.mp4`).
8.  **Terminate Pod**: Shut down the instance to save cloud costs.

---

## 3. Repository Structure

```
adfusionR&D/
├── Dockerfile                  # Base PyTorch devel container for RunPod execution
├── setup_runpod.sh             # Main automated setup entry point for fresh pods
├── requirements.txt            # Python library pins
├── pyproject.toml              # Project packaging configurations
├── main.py                     # Central CLI Orchestrator
├── adfusion/                   # Core application package
│   ├── config/config.yaml      # Pipeline and stage parameter configurations
│   ├── utils/                  # Multi-destination logging and performance profiling
│   └── pipeline/               # Hot-swappable stages (extract, detect, track, depth, etc.)
├── third_party/                # Relocated upstream model packages
│   └── sam3/                   # Unmodified facebookresearch/sam3 codebase
├── utils/                      # Shell helpers and environment check scripts
│   ├── verify_env.py           # Automated GPU and dependency diagnostic script
│   └── download_models.py      # Hugging Face checkpoint cache utility
├── scripts/                    # Validation scripts
│   └── run_validation.sh       # Single-command automated R&D validation script
├── docs/                       # Validation documentation
│   ├── runpod_validation.md    # Standard operations procedures
│   └── verification_report_template.md # Template report for run results
└── models/                     # Cache directory for gating weights (.pt files)
```

---

## 4. Getting Started on RunPod

### Prerequisites
*   **Hugging Face Account**: Access must be accepted on [facebook/sam3](https://huggingface.co/facebook/sam3) and [facebook/sam3.1](https://huggingface.co/facebook/sam3.1).
*   **HF Token**: Retrieve a Read Token from your Hugging Face settings page.

### Provisioning Command Sequence
```bash
# Clone the repository
git clone https://github.com/gunatarun007/adfusionR-D.git
cd adfusionR-D

# Set HF credentials
export HF_TOKEN="your_hf_access_token"

# Mark installers as executable
chmod +x setup_runpod.sh
chmod +x scripts/run_validation.sh

# Run provisioning (installs system tools, packages, and downloads models)
./setup_runpod.sh
```

---

## 5. Automated Validation Workflow

To execute Sprint 1 verification in a single command, run:
```bash
./scripts/run_validation.sh
```
This script sequentially runs:
1.  **Environment Diagnostics (`verify_env.py`)**: Checks Python, PyTorch, CUDA compilation, GPU properties, Triton availability, FFmpeg, OpenCV, and disk spaces.
2.  **Model Downloads (`download_models.py`)**: Ensures `sam3.pt` and `sam3.1_multiplex.pt` are cached to `models/`.
3.  **SAM 3 Object Detection (`main.py --stage detect`)**: Performs open-vocabulary text query segmentations (PCS), writing masks to `cache/masks/` and green overlays to `cache/debug/`.
4.  **SAM 3.1 Object Tracking (`main.py --stage track`)**: Performs temporal object propagation using the multiplex tracker, outputting `cache/tracks/tracks.json` and blue-ish overlays to `cache/debug/`.

---

## 6. Sprint Status & Known Limitations

*   **Sprint 1 Status**: **Provisionally Complete** (Infrastructure and SAM 3.0/3.1 predictor/tracker implementation finalized. Awaiting real-world RunPod GPU validation).
*   **Known Limitations**:
    *   *Triton dependency*: Triton is Linux-only, so the pipeline cannot be run end-to-end on Windows workstations.
    *   *Gated access*: RunPod environments must have access to the public internet and valid Hugging Face tokens to pull weights during initial provisioning.
