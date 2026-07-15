# AdFusion VPP Research Pipeline

AdFusion is a modular, high-performance research framework for AI-powered Virtual Product Placement (VPP). Designed as an "AI VFX Operating System," it allows rapid prototyping and evaluation of state-of-the-art computer vision models.

**Sprint 1 Backend**: [IDEA-Research GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) + [Meta SAM2](https://github.com/facebookresearch/sam2). Fully open-source — no Hugging Face gating, no approval required, no token needed.

---

## 1. Linux-First Architecture Decision

AdFusion follows a **Linux-first execution model**:

*   **Windows**: Development workstation — code editing, Git, documentation, unit tests.
*   **RunPod Linux GPU**: Target runtime — PyTorch GPU, CUDA compilation, full pipeline execution.

---

## 2. Detection & Tracking Pipeline

```
Video Frame
  → GroundingDINO (text prompt)     → bounding boxes
  → SAM2 ImagePredictor (box prompt) → binary masks    [cache/masks/]
  → SAM2 VideoPredictor              → temporal tracks [cache/tracks/tracks.json]
```

**Models (public download, no token):**

| Model | Checkpoint | Size | Source |
|---|---|---|---|
| GroundingDINO SwinB | `groundingdino_swinb_cogcoor.pth` | ~340 MB | GitHub releases |
| SAM2.1 Hiera-Large | `sam2.1_hiera_large.pt` | ~860 MB | Meta public CDN |

---

## 3. Repository Structure

```
adfusionR&D/
├── Dockerfile                  # PyTorch 2.5.1 + CUDA 12.4 container
├── setup_runpod.sh             # Idempotent one-command pod provisioner
├── requirements.txt            # Python dependencies
├── main.py                     # CLI Orchestrator (10 stages)
├── adfusion/
│   ├── config/config.yaml      # Pipeline config (detection: block)
│   └── pipeline/
│       ├── detect_objects.py   # Stage 2: GroundingDINO + SAM2 Image
│       └── track_objects.py    # Stage 3: SAM2 Video Predictor
├── third_party/
│   ├── sam2/                   # facebookresearch/sam2 (unmodified)
│   └── GroundingDINO/          # IDEA-Research/GroundingDINO (unmodified)
├── utils/
│   ├── verify_env.py           # System + package diagnostics
│   └── download_models.py      # Public checkpoint downloader (no token)
├── scripts/
│   └── run_validation.sh       # Single-command Sprint 1 validation
└── docs/
    ├── runpod_validation.md    # RunPod SOP
    └── verification_report_template.md
```

---

## 4. RunPod Provisioning

```bash
# 1. Clone repository
git clone https://github.com/gunatarun007/adfusionR-D.git
cd adfusionR-D

# 2. Mark scripts executable
chmod +x setup_runpod.sh scripts/run_validation.sh

# 3. Provision environment (installs deps, downloads models)
./setup_runpod.sh
```

No `HF_TOKEN` needed. No gating approval needed.

---

## 5. Running Validation

```bash
# Run full Sprint 1 validation sequence
./scripts/run_validation.sh

# Or run stages individually
python main.py --stage detect   # generates cache/masks/
python main.py --stage track    # generates cache/tracks/tracks.json
```

---

## 6. Sprint Status & Known Limitations

*   **Sprint 1 Status**: Infrastructure complete. Awaiting RunPod GPU runtime validation.
*   **Known Limitations**:
    *   GroundingDINO compiles custom CUDA ops on first install — requires `CUDA_HOME` to be set.
    *   SAM2 VideoPredictor converts PNG frames to temporary JPEGs at runtime — negligible overhead.
    *   CPU fallback is available but inference will be significantly slower.
