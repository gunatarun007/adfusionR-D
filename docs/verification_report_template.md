# RunPod Sprint 1 Validation Report

Use this template to document the runtime verification of the AdFusion VPP R&D pipeline on a RunPod Linux GPU.

---

## 1. Hardware & System Diagnostics

Record the output of the environment diagnostics step:

| Diagnostic Target | System Value / Version | Verification Status (Pass/Fail) |
| :--- | :--- | :--- |
| **GPU Name** | | |
| **VRAM Capacity (GB)** | | |
| **CUDA Driver Version**| | |
| **PyTorch Version** | | |
| **Triton Version** | | |
| **FFmpeg Binary** | | |
| **OpenCV Version** | | |

---

## 2. Model Checkpoint Caching Audit

Verify that checkpoints are correctly cached in `models/` without redownloading:

| Checkpoint Name | File Size (GB) | Status (Cached / Downloaded / Failed) |
| :--- | :--- | :--- |
| **`sam3.pt`** | | |
| **`sam3.1_multiplex.pt`**| | |

---

## 3. Pipeline Runtime Benchmarks

Record execution timings and metrics returned by `experiments/experiment_XXX/metrics.json` or console prints:

| Stage Name | Execution Duration (s) | Throughput (FPS) | Peak GPU VRAM (MB) | Status (Success/Failure) |
| :--- | :--- | :--- | :--- | :--- |
| **`object_detection`**| | | | |
| **`track_objects`** | | | | |

---

## 4. Pipeline Outputs & Artifact Check

| Output Target | File Path | Confirmed Present (Yes/No) | Visual Check Notes |
| :--- | :--- | :--- | :--- |
| **Detection Masks** | `cache/masks/mask_0001.png` | | |
| **Tracking JSON** | `cache/tracks/tracks.json` | | |
| **Detection Overlay**| `cache/debug/detect_0001.png`| | (Green Overlay boundaries) |
| **Tracking Overlay** | `cache/debug/track_0001.png` | | (Blue Boundary overlay) |

---

## 5. Console Execution Logs

Copy and paste the terminal logs from executing `./scripts/run_validation.sh` here:

```
[Paste Console Logs Here]
```

---

## 6. Screenshots & Outputs

Provide links or embeds to output frames and diagnostics:
*   *Detection Overlay Sample*:
*   *Tracking Overlay Sample*:

---

## 7. Known Issues & Future R&D Improvements

Document any runtime anomalies, warnings, or compile performance details noticed during validation:
1.  
2.  

---

## 8. Verification Checklist

*   [ ] Fresh RunPod instance booted successfully.
*   [ ] System requirements and CUDA devel images verified.
*   [ ] `./setup_runpod.sh` completed successfully.
*   [ ] Model downloader resolved gated HF checkpoint authentication.
*   [ ] Automated validation execution completed without crash or OOM.
*   [ ] Downstream VPP stages remained unaffected and compatible.
