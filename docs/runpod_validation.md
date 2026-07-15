# RunPod Validation Guide (Sprint 1)

Backend: **GroundingDINO + SAM2** — fully open-source, no Hugging Face token required.

---

## 1. Recommended RunPod Configuration

| Setting | Recommendation |
|---|---|
| **GPU** | NVIDIA L40S (48 GB) or A100 (80 GB) |
| **Docker Image** | `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel` |
| **Container Disk** | 15 GB |
| **Volume Disk** | 40 GB |
| **Environment Variables** | None required |

---

## 2. No Token Required

All model checkpoints download from **public URLs**:

| Checkpoint | Source |
|---|---|
| `sam2.1_hiera_large.pt` | `dl.fbaipublicfiles.com` (Meta CDN) |
| `groundingdino_swinb_cogcoor.pth` | GitHub Releases (IDEA-Research) |

No Hugging Face login. No Meta gating approval. No `HF_TOKEN`.

---

## 3. Exact Command Sequence

```bash
# Clone repository
git clone https://github.com/gunatarun007/adfusionR-D.git
cd adfusionR-D

# Make scripts executable
chmod +x setup_runpod.sh scripts/run_validation.sh

# Provision the environment
# This installs system packages, SAM2, GroundingDINO, downloads checkpoints
./setup_runpod.sh

# Run full automated validation
./scripts/run_validation.sh
```

Or run stages individually:
```bash
python main.py --stage detect   # GroundingDINO + SAM2 Image → cache/masks/
python main.py --stage track    # SAM2 Video → cache/tracks/tracks.json
```

---

## 4. Expected Outputs

### Environment Check (`verify_env.py`)
```
[*] Python Version: 3.11.x ... OK
[*] PyTorch Version: 2.5.1 ... OK
[*] GPU: NVIDIA L40S (1 GPU(s))
[*] VRAM: 48.00 GB
[*] GroundingDINO package: importable ... OK
[*] SAM2 package: importable ... OK
[*] Checkpoint sam2.1_hiera_large.pt: 860 MB ... OK
[*] Checkpoint groundingdino_swinb_cogcoor.pth: 340 MB ... OK
[SUCCESS] Environment verification PASSED.
```

### Detection Stage
- `cache/masks/mask_0001.png` → `mask_NNNN.png` — binary segmentation masks
- `cache/debug/detect_0001.png` → `detect_NNNN.png` — green overlay debug frames

### Tracking Stage
- `cache/tracks/tracks.json` — object trajectory data per frame
- `cache/debug/track_0001.png` → `track_NNNN.png` — blue overlay debug frames

---

## 5. Common Failure Cases

### `ImportError: No module named 'groundingdino'`
GroundingDINO was not installed. Run:
```bash
pip install -e third_party/GroundingDINO
```
Ensure `CUDA_HOME` is set: `export CUDA_HOME=/usr/local/cuda`

### `ImportError: No module named 'sam2'`
SAM2 was not installed. Run:
```bash
pip install -e third_party/sam2
```

### `FileNotFoundError: models/sam2.1_hiera_large.pt`
Checkpoint missing. Run:
```bash
python utils/download_models.py
```

### `CUDA Out of Memory`
Reduce the video resolution in `adfusion/config/config.yaml` or switch to a GPU with more VRAM (L40S recommended).

### GroundingDINO CUDA compilation fails
```bash
export CUDA_HOME=/usr/local/cuda
export FORCE_CUDA=1
pip install -e third_party/GroundingDINO
```
