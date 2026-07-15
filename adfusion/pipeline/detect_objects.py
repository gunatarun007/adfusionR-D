import os
import time
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, List
from adfusion.pipeline.base import BaseStage, PipelineContext, StageResult


class ObjectDetectionStage(BaseStage):
    """Stage 2: Object Detection.

    Uses IDEA-Research GroundingDINO for open-vocabulary text-prompted bounding
    box detection, then feeds each box into Meta SAM2 ImagePredictor to generate
    binary segmentation masks. Both models are fully open-source with public
    (non-gated) checkpoint downloads — no Hugging Face token required.

    Outputs:
        cache/masks/mask_XXXX.png  — 8-bit binary masks
        cache/debug/detect_XXXX.png — green overlay visualisations
    """

    def __init__(self) -> None:
        super().__init__("object_detection")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_gdino(self, config_path: Path, ckpt_path: Path, device: str):
        """Load GroundingDINO model from local third_party installation."""
        from groundingdino.util.inference import load_model
        model = load_model(str(config_path), str(ckpt_path))
        model = model.to(device)
        model.eval()
        return model

    def _load_sam2_image(self, ckpt_path: Path, cfg_path: str, device: str):
        """Build SAM2 ImagePredictor from local third_party installation."""
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        sam2_model = build_sam2(cfg_path, str(ckpt_path), device=device)
        predictor = SAM2ImagePredictor(sam2_model)
        return predictor

    def _gdino_predict(self, gdino_model, image_path: Path, caption: str,
                       box_threshold: float, text_threshold: float, device: str):
        """Run GroundingDINO inference; returns boxes in [x1,y1,x2,y2] pixel coords."""
        import groundingdino.datasets.transforms as T
        from groundingdino.util.inference import predict
        from PIL import Image as PILImage

        # GroundingDINO expects a normalised tensor image
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        pil_img = PILImage.open(str(image_path)).convert("RGB")
        W, H = pil_img.size
        image_tensor, _ = transform(pil_img, None)  # returns (tensor, None)

        # caption must end with "."
        caption_clean = caption.strip()
        if not caption_clean.endswith("."):
            caption_clean += "."

        boxes_norm, logits, phrases = predict(
            model=gdino_model,
            image=image_tensor,
            caption=caption_clean,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=device,
        )

        # Convert normalised CX,CY,W,H → absolute x1,y1,x2,y2
        if len(boxes_norm) == 0:
            return [], []

        boxes_abs = []
        for box in boxes_norm:
            cx, cy, bw, bh = box.tolist()
            x1 = (cx - bw / 2) * W
            y1 = (cy - bh / 2) * H
            x2 = (cx + bw / 2) * W
            y2 = (cy + bh / 2) * H
            boxes_abs.append([x1, y1, x2, y2])

        scores = logits.tolist()
        return boxes_abs, scores

    # ------------------------------------------------------------------
    # Stage interface
    # ------------------------------------------------------------------

    def run(self, context: PipelineContext) -> StageResult:
        start_time = time.perf_counter()
        context.logger.info("=== Starting GroundingDINO + SAM2 Object Detection ===")

        # ── Configuration ──────────────────────────────────────────────
        det_cfg = context.config.get("detection", {})
        gdino_ckpt_rel   = det_cfg.get("gdino_checkpoint", "models/groundingdino_swinb_cogcoor.pth")
        sam2_ckpt_rel    = det_cfg.get("sam2_checkpoint",  "models/sam2.1_hiera_large.pt")
        sam2_cfg_name    = det_cfg.get("sam2_config",      "configs/sam2.1/sam2.1_hiera_l.yaml")
        box_threshold    = float(det_cfg.get("box_threshold",  0.35))
        text_threshold   = float(det_cfg.get("text_threshold", 0.25))
        device_cfg       = det_cfg.get("device", "cuda")

        target_class = (context.config
                        .get("stages", {})
                        .get("object_detection", {})
                        .get("target_class", "person"))

        gdino_ckpt = context.workspace_dir / gdino_ckpt_rel
        sam2_ckpt  = context.workspace_dir / sam2_ckpt_rel

        # GroundingDINO config lives inside third_party/GroundingDINO
        gdino_config = (context.workspace_dir
                        / "third_party" / "GroundingDINO"
                        / "groundingdino" / "config"
                        / "GroundingDINO_SwinB_cfg.py")

        # ── Device ─────────────────────────────────────────────────────
        device = "cpu"
        if device_cfg == "cuda" and torch.cuda.is_available():
            device = "cuda"
        else:
            if device_cfg == "cuda":
                context.logger.warning("CUDA not available; falling back to CPU.")

        # ── Validate checkpoint files exist ────────────────────────────
        for label, path in [("GroundingDINO checkpoint", gdino_ckpt),
                            ("SAM2 checkpoint", sam2_ckpt),
                            ("GroundingDINO config", gdino_config)]:
            if not path.exists():
                return StageResult(
                    stage_name=self.name, success=False,
                    runtime_seconds=time.perf_counter() - start_time,
                    error_message=f"{label} not found at {path}. "
                                  f"Run: python utils/download_models.py"
                )

        # ── Load models ────────────────────────────────────────────────
        context.logger.info("Loading GroundingDINO model…")
        try:
            gdino_model = self._load_gdino(gdino_config, gdino_ckpt, device)
        except Exception as e:
            return StageResult(stage_name=self.name, success=False,
                               runtime_seconds=time.perf_counter() - start_time,
                               error_message=f"GroundingDINO load failed: {e}")

        context.logger.info("Loading SAM2 ImagePredictor…")
        try:
            sam2_predictor = self._load_sam2_image(sam2_ckpt, sam2_cfg_name, device)
        except Exception as e:
            return StageResult(stage_name=self.name, success=False,
                               runtime_seconds=time.perf_counter() - start_time,
                               error_message=f"SAM2 ImagePredictor load failed: {e}")

        # ── Directories ────────────────────────────────────────────────
        extraction_meta = context.stage_outputs.get("frame_extraction", {}).get("metadata", {})
        width  = extraction_meta.get("width",  1280)
        height = extraction_meta.get("height", 720)

        frames_dir = context.cache_dir / "frames"
        masks_dir  = context.cache_dir / "masks"
        debug_dir  = context.cache_dir / "debug"
        masks_dir.mkdir(parents=True, exist_ok=True)
        debug_dir.mkdir(parents=True, exist_ok=True)

        frame_files = sorted(frames_dir.glob("frame_*.png"))
        if not frame_files:
            return StageResult(stage_name=self.name, success=False,
                               runtime_seconds=time.perf_counter() - start_time,
                               error_message="No frames found. Run frame extraction first.")

        output_files: List[Path] = []
        masks_meta:  List[Dict[str, Any]] = []

        context.logger.info(
            f"Processing {len(frame_files)} frames | prompt: '{target_class}' | device: {device}"
        )

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        # ── Per-frame inference ────────────────────────────────────────
        for idx, frame_file in enumerate(frame_files):
            frame_bgr = cv2.imread(str(frame_file))
            if frame_bgr is None or frame_bgr.size == 0:
                context.logger.warning(f"Skipping corrupted frame: {frame_file.name}")
                mask = np.zeros((height, width), dtype=np.uint8)
                bbox, polygon = [0, 0, 0, 0], []
                cx, cy, radius, confidence, detected = width // 2, height // 2, 0, 0.0, False
            else:
                # 1. GroundingDINO → bounding boxes
                boxes, scores = self._gdino_predict(
                    gdino_model, frame_file, target_class,
                    box_threshold, text_threshold, device
                )

                if boxes:
                    # Pick highest-scoring box
                    best_i = int(np.argmax(scores))
                    bbox   = boxes[best_i]          # [x1, y1, x2, y2]
                    confidence = float(scores[best_i])

                    # 2. SAM2 → mask from box prompt
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    with torch.inference_mode():
                        sam2_predictor.set_image(frame_rgb)
                        box_tensor = np.array(bbox, dtype=np.float32)
                        masks_out, _, _ = sam2_predictor.predict(
                            box=box_tensor,
                            multimask_output=False,
                        )
                    # masks_out shape: (N, H, W) bool
                    mask = (masks_out[0].astype(np.uint8)) * 255
                    if mask.shape[:2] != (height, width):
                        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    polygon = (max(contours, key=cv2.contourArea).reshape(-1, 2).tolist()
                               if contours else [])
                    x1, y1, x2, y2 = bbox
                    cx     = int((x1 + x2) / 2)
                    cy     = int((y1 + y2) / 2)
                    radius = int(max(x2 - x1, y2 - y1) / 2)
                    detected = True
                else:
                    mask = np.zeros((height, width), dtype=np.uint8)
                    bbox, polygon = [0, 0, 0, 0], []
                    cx, cy = width // 2, height // 2
                    radius, confidence, detected = 0, 0.0, False

            # ── Write outputs ──────────────────────────────────────────
            mask_path = masks_dir / f"mask_{idx + 1:04d}.png"
            cv2.imwrite(str(mask_path), mask)
            output_files.append(mask_path)

            masks_meta.append({
                "frame_idx":  idx,
                "center":     [cx, cy],
                "radius":     radius,
                "bbox":       bbox,
                "polygon":    polygon,
                "confidence": confidence,
                "detected":   detected,
            })

            # ── Debug overlay (green mask + bbox) ─────────────────────
            debug_frame = frame_bgr.copy() if frame_bgr is not None else np.zeros((height, width, 3), dtype=np.uint8)
            if detected:
                colour_mask = np.zeros_like(debug_frame)
                colour_mask[mask > 0] = [0, 255, 0]
                cv2.addWeighted(debug_frame, 1.0, colour_mask, 0.4, 0, debug_frame)
                cv2.rectangle(debug_frame,
                              (int(bbox[0]), int(bbox[1])),
                              (int(bbox[2]), int(bbox[3])),
                              (0, 255, 0), 2)
                cv2.putText(debug_frame,
                            f"DET: {target_class} ({confidence:.2f})",
                            (int(bbox[0]), max(int(bbox[1]) - 10, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imwrite(str(debug_dir / f"detect_{idx + 1:04d}.png"), debug_frame)

            if (idx + 1) % 30 == 0 or (idx + 1) == len(frame_files):
                context.logger.info(f"Processed frame {idx + 1}/{len(frame_files)}")

        # ── Metrics ───────────────────────────────────────────────────
        peak_vram_mb = 0.0
        if device == "cuda":
            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            context.logger.info(f"Peak VRAM: {peak_vram_mb:.1f} MB")

        runtime = time.perf_counter() - start_time
        fps     = len(frame_files) / runtime if runtime > 0 else 0.0
        context.logger.info(
            f"Detection stage complete. Frames: {len(frame_files)} | "
            f"FPS: {fps:.2f} | Runtime: {runtime:.2f}s"
        )

        return StageResult(
            stage_name=self.name,
            success=True,
            runtime_seconds=runtime,
            output_files=output_files,
            metadata={
                "masks_metadata":    masks_meta,
                "target_class":      target_class,
                "device":            device,
                "peak_vram_mb":      peak_vram_mb,
                "average_fps":       fps,
                "checkpoint_version": "GroundingDINO SwinB + SAM2.1 Hiera-L",
            },
        )

    def is_cached(self, context: PipelineContext) -> bool:
        masks_dir = context.cache_dir / "masks"
        if not masks_dir.exists():
            return False
        masks = list(masks_dir.glob("mask_*.png"))
        if not masks:
            return False
        if any(m.stat().st_size == 0 for m in masks):
            return False
        return True
