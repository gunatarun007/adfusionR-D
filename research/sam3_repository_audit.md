# SAM 3 Repository Audit Document

This document presents a technical audit of the official Meta **Segment Anything Model 3 (SAM 3)** repository based on a local inspection of the cloned codebase [1].

---

## 1. Repository Tree & Key Files

The official `facebookresearch/sam3` repository is structured as follows:

```
sam3/
├── pyproject.toml              # Project dependencies, build-system & optional dependencies [1]
├── README.md                   # Getting started & model checkpoints documentation [1]
├── RELEASE_SAM3p1.md           # Documentation for SAM 3.1 Object Multiplexing release [1]
├── examples/                   # Official Jupyter Notebook tutorials [1]
│   ├── sam3_image_predictor_example.ipynb
│   ├── sam3_video_predictor_example.ipynb
│   └── sam3.1_video_predictor_example.ipynb
├── scripts/                    # Speed measurement and qualitative test scripts [1]
│   ├── measure_speed.py
│   └── qualitative_test.py
└── sam3/                       # Core python package folder [1]
    ├── __init__.py
    ├── model_builder.py        # Entry points for initializing and building all SAM 3/3.1 models [1]
    ├── logger.py               # Package-wide logging utility [1]
    ├── visualization_utils.py  # Utilities for drawing overlay masks, boxes, and text on images [1]
    ├── sam/                    # Original SAM-based modular network layers (rope, transformer, mask_decoder) [1]
    ├── perflib/                # Triton & custom optimized kernels (nms, iou, fa3, fused, masks_ops) [1]
    └── model/                  # Target inference modules, predictors, and trackers [1]
        ├── decoder.py          # Decoupled transformer decoder layers [1]
        ├── encoder.py          # Vision-Language transformer encoder layers [1]
        ├── text_encoder_ve.py  # Text encoder wrapping CLIP/SigLIP transformer [1]
        ├── tokenizer_ve.py     # Noun/concept phrase text tokenizer [1]
        ├── sam3_image.py       # Image model wrapper class [1]
        ├── sam3_image_processor.py # Processors for static image inference [1]
        ├── sam3_video_base.py  # Core video sequence inference layers [1]
        ├── sam3_video_inference.py # Video tracking and segmentation with interactivity [1]
        ├── sam3_video_predictor.py # Video inference executor on single/multi-GPUs [1]
        ├── sam3_base_predictor.py # Unified base class exposing the handle_request/handle_stream_request APIs [1]
        ├── sam3_multiplex_base.py # Baseline shared-memory object tracking layers [1]
        ├── sam3_multiplex_detector.py # Multiplex concept detector model [1]
        └── sam3_multiplex_tracking.py # Interactive tracking class supporting multiplex buckets [1]
```

---

## 2. Official Python API & Core Classes

Based on `sam3/sam3/model_builder.py` and package imports, the official Meta APIs are built around the following classes [1]:

### A. Model Builders (in `sam3/sam3/model_builder.py`)
*   `build_sam3_image_model(checkpoint_path, **kwargs)`: Builds the baseline `Sam3Image` model for promptable static image segmentation [1].
*   `build_sam3_video_predictor(checkpoint_path, bpe_path, compile, **kwargs)`: Recommended entry point to build `Sam3VideoPredictorMultiGPU` for base SAM 3.0 video tracking [1].
*   `build_sam3_multiplex_video_predictor(checkpoint_path, bpe_path, max_num_objects, multiplex_count, compile, **kwargs)`: Recommended entry point to build the `Sam3MultiplexVideoPredictor` for SAM 3.1 video tracking [1].
*   `build_sam3_predictor(checkpoint_path, bpe_path, version, compile, **kwargs)`: Unified entry point. Accepts `version="sam3"` (loads SAM 3.0 predictor) or `version="sam3.1"` (loads SAM 3.1 multiplex predictor) [1].

### B. Core Predictor / Processor Classes
*   `Sam3Processor`: Image inference orchestrator located in `sam3/sam3/model/sam3_image_processor.py` [1].
*   `Sam3VideoPredictorMultiGPU` (inherits from `Sam3VideoInferenceWithInstanceInteractivity`): Exposes standard tracking execution loops [1].
*   `Sam3MultiplexVideoPredictor`: Handles multi-object shared-memory inference loops [1].
*   `Sam3BasePredictor` (located in `sam3/sam3/model/sam3_base_predictor.py`): Base class for video tracking. It is the target class that exposes the unified request interface [1]:
    *   `handle_request(request: dict)`: Dispatches synchronous actions (`"start_session"`, `"add_prompt"`, `"remove_object"`, `"reset_session"`, `"close_session"`) [1].
    *   `handle_stream_request(request: dict)`: Dispatches generator streams for frame tracking (`"propagate_in_video"`) [1].

---

## 3. Official Checkpoint Inventory

Meta AI distributes two official checkpoints on Hugging Face (requiring non-commercial license approval) [4, 5]:

1.  **`sam3.pt`** [4]
    *   *Repository*: `facebook/sam3`
    *   *Size*: 3.45 GB
    *   *Purpose*: Pretrained base model for static image Promptable Concept Segmentation (PCS) and single-object video tracking [2, 4].
2.  **`sam3.1_multiplex.pt`** [5]
    *   *Repository*: `facebook/sam3.1`
    *   *Size*: 3.50 GB
    *   *Purpose*: Optimized tracking model with Object Multiplexing, allowing fast, parallel tracking of up to 16 objects with reduced VRAM [5].

---

## 4. Official Video Tracking Request Flow

The official example tracking pipeline (traced directly from `examples/sam3.1_video_predictor_example.ipynb`) follows this lifecycle:

```
[1. Load Predictor]
predictor = build_sam3_predictor(version="sam3.1", compile=False)
      │
      ▼
[2. Initialize Session]
response = predictor.handle_request({
    "type": "start_session",
    "resource_path": "cache/frames",  # Path to directory containing frame images
})
session_id = response["session_id"]
      │
      ▼
[3. Add Text Prompt (Concept Registration)]
predictor.handle_request({
    "type": "add_prompt",
    "session_id": session_id,
    "frame_index": 0,                 # Frame index to register the prompt
    "text": "cup",                    # Target noun phrase
    "obj_id": 1,                      # Custom tracking ID assigned to this concept
})
      │
      ▼
[4. Propagate Tracking (Stream Generator)]
stream = predictor.handle_stream_request({
    "type": "propagate_in_video",
    "session_id": session_id,
    "propagation_direction": "forward"
})

for frame_data in stream:
    frame_idx = frame_data["frame_index"]
    outputs = frame_data["outputs"]  # Dictionary containing binary masks, scores, and boxes
      │
      ▼
[5. Close Session]
predictor.handle_request({
    "type": "close_session",
    "session_id": session_id
})
```

---

## 5. Output Objects & Specifications

Inside the tracking generator loop, the returned `outputs` dictionary (post-processed via `_postprocess_output` in `sam3_video_inference.py`) has the following keys and shapes [1]:

*   **`out_obj_ids`**: `np.ndarray` of shape `(N,)` (dtype: `int64`), listing the IDs of active objects tracked in this frame [1].
*   **`out_probs`**: `np.ndarray` of shape `(N,)` (dtype: `float32`), listing the probability confidence score for each tracked object [1].
*   **`out_boxes_xywh`**: `np.ndarray` of shape `(N, 4)` (dtype: `float32`).
    *   *Warning*: Coordinates are normalized relative to image boundaries and are in **Center-Width-Height format** [1]:
        `[(center_x / W), (center_y / H), (width / W), (height / H)]` [1].
*   **`out_binary_masks`**: `np.ndarray` of shape `(N, H_video, W_video)` (dtype: `bool`), containing the dense segmentation masks for each object [1].

---

## 6. Pipeline Integration Plan (No Architecture Changes)

To replace our placeholder stages using the official Meta API:

### A. Stage 2: Object Detection (`adfusion/pipeline/detect_objects.py`)
In our pipeline, the detection stage parses prompts and registers masks frame-by-frame. We will initialize the official model and processor:
1.  Verify `models/sam3.pt` (auto-download via `download_ckpt_from_hf` inside `model_builder.py` if missing) [1].
2.  Set image using `processor.set_image(image)`.
3.  Predict using `state = processor.set_text_prompt(state, prompt=target_class)`.
4.  Extract the highest confidence mask from `state['masks']` and save it to `cache/masks/mask_XXXX.png` [1].
5.  Save the bounding box coordinates, polygon contours, and confidence in the stage's execution metadata.

### B. Stage 3: Video Tracking (`adfusion/pipeline/track_objects.py`)
1.  Read the directory of frames `cache/frames` [1].
2.  Build the official video predictor: `predictor = build_sam3_predictor(version="sam3.1", checkpoint_path="models/sam3.1_multiplex.pt")`.
3.  Initialize the session: `start_session` on `cache/frames`.
4.  Add the text prompt at frame 0 using `add_prompt` with `text=target_class`.
5.  Call `propagate_in_video` to generate the tracks.
6.  For each frame:
    *   Read `out_boxes_xywh` and convert to absolute pixel `[x_min, y_min, width, height]` format:
        ```python
        cx, cy, nw, nh = out_boxes_xywh[i]
        w = nw * W_video
        h = nh * H_video
        x_min = (cx - nw / 2) * W_video
        y_min = (cy - nh / 2) * H_video
        bbox_xywh = [x_min, y_min, w, h]
        ```
    *   Save `tracks.json` with the assigned `object_id`, `frame_index`, converted `bbox`, contour polygons, and calculated timestamp.
7.  Generate debug visual overlays under `cache/debug/track_XXXX.png`.
8.  Close session.

---

## 7. Official Dependencies

Direct dependencies specified in `sam3/pyproject.toml` are [1]:
*   `timm>=1.0.17`
*   `numpy>=1.26,<2`
*   `tqdm`
*   `ftfy==6.1.1`
*   `regex`
*   `iopath>=0.1.10`
*   `typing_extensions`
*   `huggingface_hub`

Recommended optional dependencies for optimized CUDA performance:
*   `einops`
*   `ninja`
*   `flash-attn-3` (FlashAttention-3 for high speed throughput on Hopper/Blackwell) [1].
