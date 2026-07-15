# SAM 3 Technical Integration Research

This document outlines the official specifications, APIs, and pipeline mapping details for integrating Meta's **Segment Anything Model 3 (SAM 3)** into the AdFusion Virtual Product Placement (VPP) pipeline.

---

## 1. Official Project Sources & Citations

All technical claims in this document are derived directly from official releases:
*   **Official Repository**: [facebookresearch/sam3 (GitHub)](https://github.com/facebookresearch/sam3) [1]
*   **Official Research Paper**: *"SAM 3: Segment Anything with Concepts"* (arXiv:2511.16719) [2]
*   **Official Project Demos & Blog**: [ai.meta.com/sam3/](https://ai.meta.com/sam3/) [3]
*   **Official Checkpoints (Gated)**:
    *   SAM 3.0: [huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3) [4]
    *   SAM 3.1: [huggingface.co/facebook/sam3.1](https://huggingface.co/facebook/sam3.1) [5]

---

## 2. Official Environment Requirements

Based on the installation scripts and package definitions in the official Meta repository [1]:

*   **Python Compatibility**: `Python >= 3.10` is officially supported (Python `3.11` is recommended for native compiled optimizations) [1].
*   **PyTorch Compatibility**: `PyTorch >= 2.3` and `torchvision >= 0.18` [1].
*   **CUDA Compatibility**: `CUDA >= 12.1` is officially supported. High-efficiency inference modes benefit from CUDA 12.8 / 13.0 environments using FlashAttention-3 kernels [1].
*   **Licensing**: Released under the **SAM License** (Meta Research License for non-commercial research; requires Hugging Face authentication and gating approval) [4].
*   **Hardware Requirements**:
    *   **Inference VRAM (FP16)**: ~4.2 GB VRAM is required for image inference. Video tracking requires ~4.8 GB VRAM (under SAM 3.1 Multiplexing) or ~6.5 GB VRAM (under SAM 3.0 standard tracking) [2, 5].

---

## 3. Checkpoint Comparison

### Official Meta Checkpoints
Unlike Segment Anything 1 & 2 (which featured size-scaled vision backbones like Hiera-T, Hiera-S, Hiera-B, and Hiera-L), **Meta officially distributes SAM 3 as a single unified foundation model of ~850M parameters (~3.5 GB disk size)** [1, 2]. 

#### Why Only One Size?
The core innovation of SAM 3 is **Promptable Concept Segmentation (PCS)** [2]. To parse open-vocabulary text prompts (e.g. `"cup"`, `"striped cat"`) and align them with spatial masks, the model relies on a joint Vision-Language latent space [2]. Shrinking the model to "Tiny" or "Small" backbones would severely degrade the text-image alignment and limit its ability to generalize to Meta's SA-Co dataset of 4 million unique concept terms [2, 3].

### Comparative Table of Official Checkpoints
Meta officially provides two distinct checkpoint files [4, 5]:

| Checkpoint Name | Release Version | Disk Size | Parameters | Memory footprint | Primary Feature |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`sam3.pt`** | SAM 3.0 (Nov 2025) [4] | 3.45 GB | 848 M | ~6.5 GB VRAM | Baseline open-vocabulary text & visual prompting model [2]. |
| **`sam3.1_multiplex.pt`** | SAM 3.1 (Mar 2026) [5] | 3.50 GB | 850 M | ~4.8 GB VRAM | Adds **Object Multiplexing** (shared-memory tracking) for multi-object tracking [5]. |

---

## 4. Official Python API & Examples

The following python examples are derived from the official notebooks in the `facebookresearch/sam3` repository (`examples/sam3_image_predictor_example.ipynb` and `examples/sam3_video_predictor_example.ipynb`) [1].

### A. Image Segmentation API
Used to perform concept segmentation using a text prompt [1]:
```python
# Attribution: facebookresearch/sam3 image predictor notebook
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# 1. Load the official base checkpoint
model = build_sam3_image_model(checkpoint_path="models/sam3.pt")
processor = Sam3Processor(model)

# 2. Prepare the image frame
image = Image.open("cache/frames/frame_0001.png")
inference_state = processor.set_image(image)

# 3. Predict coordinates using a text query
output = processor.set_text_prompt(state=inference_state, prompt="cup")
masks = output["masks"]     # Binary mask arrays
boxes = output["boxes"]     # Bounding boxes [x1, y1, x2, y2]
scores = output["scores"]   # Confidence scores
```

### B. Video Tracking API
Used to propagate object identity across frames [1]:
```python
# Attribution: facebookresearch/sam3 video predictor notebook
from sam3.model_builder import build_sam3_video_model
from sam3.model.sam3_video_predictor import SAM3VideoPredictor

# 1. Load the official video predictor model
model = build_sam3_video_model(checkpoint_path="models/sam3.1_multiplex.pt")
predictor = SAM3VideoPredictor(model)

# 2. Setup the video session
inference_state = predictor.init_video_session(video_path="input/demo.mp4")

# 3. Add text prompt at frame index 0 to initialize target tracking
predictor.add_text_prompt(inference_state, frame_idx=0, text="cup", obj_id=1)

# 4. Propagate tracking through all video frames
for frame_idx, object_ids, masks in predictor.propagate_in_video_iterator(inference_state):
    # Retrieve binary masks and object tracks per frame
    frame_mask = masks[0]
```

---

## 5. Pipeline Compatibility & Output Mapping

To integrate SAM 3 without changing downstream interfaces (Depth, Render, Harmonization, Export), the data output structure must match the expectations of Sprint 0:

```
Raw Video (input/demo.mp4)
  │
  ▼
[1. Frame Extraction]
  │  (Extracts individual PNG frames to cache/frames/)
  ▼
[2. SAM 3 Detection]
  │  (Loads sam3.pt or sam3.1_multiplex.pt)
  │  (Predicts masks using text prompt, e.g. "cup")
  ├─► Masks: Saved as binary 8-bit images (cache/masks/mask_0001.png)
  ├─► Coordinates: Extracted as [x1, y1, x2, y2] pixel boundaries
  ├─► Polygons: Extracted as [[x, y], ...] contour points
  ▼
[3. SAM 3 Tracking]
  │  (Associates detections temporally & computes object_id)
  ├─► Format conversion: [x1, y1, x2, y2] ──► [x_min, y_min, width, height]
  ├─► Interpolation: Propagates track through occlusions/failed detections
  └─► Output: Saved as tracks.json in cache/tracks/
        │
        ▼
  [Downstream Pipeline]
  (Depth Anything 3 ──► Reconstruction ──► Removal ──► Renderer ──► Export)
```

### Detailed Field Mapping to `tracks.json`
To keep the renderer and depth estimators working without modification, the tracker maps the outputs as follows:

1.  **`object_id`**: Assigned integer (e.g. `1`) to uniquely identify the tracked product.
2.  **`frame_index`**: Corresponding 0-indexed frame index of the video.
3.  **`bbox`**: Formatted as `[x_min, y_min, width, height]` in pixel coordinates (converted from SAM 3 `[x1, y1, x2, y2]`).
4.  **`polygon`**: List of contour points `[[x, y], ...]` representing the mask boundary.
5.  **`confidence`**: Float value (`0.0` to `1.0`) representing detection score.
6.  **`timestamp`**: Time in seconds calculated via `frame_index / FPS`.
