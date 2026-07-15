import json
import shutil
import tempfile
import time
import cv2
import numpy as np
from pathlib import Path
from typing import Any, Dict, List
from adfusion.pipeline.base import BaseStage, PipelineContext, StageResult


class VideoTrackingStage(BaseStage):
    """Stage 3: Video Tracking.

    Uses the Meta SAM2 VideoPredictor to propagate segmentation masks across all
    frames. The first-frame bounding box comes from the upstream detection stage.

    SAM2 VideoPredictor requires JPEG frames in a flat directory named as
    zero-padded integers (e.g. 00000.jpg, 00001.jpg …). This stage creates a
    temporary directory, writes JPEG copies of the PNG frames, runs the predictor,
    then cleans up. No modifications to upstream stages are required.

    Outputs:
        cache/masks/mask_XXXX.png   — refined per-frame binary masks
        cache/tracks/tracks.json    — object trajectory data
        cache/debug/track_XXXX.png  — blue overlay visualisations
    """

    def __init__(self) -> None:
        super().__init__("track_objects")

    # ------------------------------------------------------------------
    # Stage interface
    # ------------------------------------------------------------------

    def run(self, context: PipelineContext) -> StageResult:
        start_time = time.perf_counter()
        context.logger.info("=== Starting SAM2 Video Tracking ===")

        # ── Configuration ──────────────────────────────────────────────
        det_cfg     = context.config.get("detection", {})
        sam2_ckpt_rel = det_cfg.get("sam2_checkpoint", "models/sam2.1_hiera_large.pt")
        sam2_cfg_name = det_cfg.get("sam2_config",     "configs/sam2.1/sam2.1_hiera_l.yaml")
        device_cfg    = det_cfg.get("device", "cuda")

        sam2_ckpt = context.workspace_dir / sam2_ckpt_rel

        # ── Device ─────────────────────────────────────────────────────
        import torch
        device = "cpu"
        if device_cfg == "cuda" and torch.cuda.is_available():
            device = "cuda"
        else:
            if device_cfg == "cuda":
                context.logger.warning("CUDA not available; falling back to CPU.")

        # ── Validate checkpoint ─────────────────────────────────────────
        if not sam2_ckpt.exists():
            return StageResult(
                stage_name=self.name, success=False,
                runtime_seconds=time.perf_counter() - start_time,
                error_message=f"SAM2 checkpoint not found at {sam2_ckpt}. "
                              f"Run: python utils/download_models.py"
            )

        # ── Retrieve upstream outputs ───────────────────────────────────
        detection_meta = (context.stage_outputs
                          .get("object_detection", {})
                          .get("metadata", {}))
        label        = detection_meta.get("target_class", "person")
        masks_meta   = detection_meta.get("masks_metadata", [])

        extraction_meta = (context.stage_outputs
                           .get("frame_extraction", {})
                           .get("metadata", {}))
        video_fps = extraction_meta.get("fps",    30.0)
        width     = extraction_meta.get("width",  1280)
        height    = extraction_meta.get("height", 720)

        # ── Paths ───────────────────────────────────────────────────────
        frames_dir = context.cache_dir / "frames"
        masks_dir  = context.cache_dir / "masks"
        tracks_dir = context.cache_dir / "tracks"
        debug_dir  = context.cache_dir / "debug"

        tracks_dir.mkdir(parents=True, exist_ok=True)
        debug_dir.mkdir(parents=True, exist_ok=True)

        frame_files = sorted(frames_dir.glob("frame_*.png"))
        if not frame_files:
            return StageResult(
                stage_name=self.name, success=False,
                runtime_seconds=time.perf_counter() - start_time,
                error_message="No frames found. Run frame extraction first."
            )

        # ── Find the first detected bounding box ────────────────────────
        first_box = None
        first_box_frame_idx = 0
        for fm in masks_meta:
            if fm.get("detected") and fm.get("bbox") and fm["bbox"] != [0, 0, 0, 0]:
                first_box = fm["bbox"]          # [x1, y1, x2, y2] absolute pixels
                first_box_frame_idx = fm["frame_idx"]
                break

        if first_box is None:
            context.logger.warning(
                "No detected object found in detection stage. "
                "Falling back to centre-region box for tracking initialisation."
            )
            cx, cy = width // 2, height // 2
            bw, bh = width // 4, height // 4
            first_box = [cx - bw // 2, cy - bh // 2,
                         cx + bw // 2, cy + bh // 2]

        context.logger.info(
            f"Initialising SAM2 tracker from frame {first_box_frame_idx} "
            f"box {[round(v, 1) for v in first_box]}"
        )

        # ── Build a temporary JPEG frame directory for SAM2 ─────────────
        # SAM2 VideoPredictor.init_state() requires files named
        # 00000.jpg, 00001.jpg, … (zero-padded, zero-indexed integers)
        tmp_dir = Path(tempfile.mkdtemp(prefix="adfusion_sam2_frames_"))
        try:
            context.logger.info(f"Writing JPEG frames to temp dir: {tmp_dir}")
            for i, fp in enumerate(frame_files):
                bgr = cv2.imread(str(fp))
                if bgr is None:
                    bgr = np.zeros((height, width, 3), dtype=np.uint8)
                jpeg_path = tmp_dir / f"{i:05d}.jpg"
                cv2.imwrite(str(jpeg_path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # ── Load SAM2 VideoPredictor ────────────────────────────────
            context.logger.info("Loading SAM2 VideoPredictor…")
            from sam2.build_sam import build_sam2_video_predictor
            predictor = build_sam2_video_predictor(sam2_cfg_name, str(sam2_ckpt),
                                                   device=device)

            tracks_data: Dict[str, Any] = {
                "track_id": 1,
                "label":    label,
                "frames":   [],
            }

            with torch.inference_mode():
                # init_state accepts a directory of JPEG frames
                state = predictor.init_state(video_path=str(tmp_dir))

                # Register the first-frame box prompt for object ID 1
                box_np = np.array(first_box, dtype=np.float32)
                predictor.add_new_points_or_box(
                    state,
                    frame_idx=first_box_frame_idx,
                    obj_id=1,
                    box=box_np,
                )

                context.logger.info("Propagating masks through video…")

                for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
                    # mask_logits: Tensor (N, 1, H, W) — one channel per object
                    target_pos = (obj_ids.index(1)
                                  if isinstance(obj_ids, list) and 1 in obj_ids
                                  else (0 if len(obj_ids) > 0 else None))

                    if target_pos is not None:
                        mask_bool = (mask_logits[target_pos, 0] > 0).cpu().numpy()
                        mask = mask_bool.astype(np.uint8) * 255
                        if mask.shape != (height, width):
                            mask = cv2.resize(mask, (width, height),
                                              interpolation=cv2.INTER_NEAREST)

                        # Derive bounding box from mask pixels
                        ys, xs = np.where(mask > 0)
                        if len(xs):
                            x1, x2 = int(xs.min()), int(xs.max())
                            y1, y2 = int(ys.min()), int(ys.max())
                            bw = float(x2 - x1)
                            bh = float(y2 - y1)
                            bbox_xywh = [float(x1), float(y1), bw, bh]
                            cx = int(x1 + bw / 2)
                            cy = int(y1 + bh / 2)
                        else:
                            bbox_xywh = [0.0, 0.0, 0.0, 0.0]
                            cx, cy = width // 2, height // 2

                        contours, _ = cv2.findContours(
                            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )
                        polygon = (max(contours, key=cv2.contourArea)
                                   .reshape(-1, 2).tolist() if contours else [])
                        # Confidence from logit magnitude (proxy)
                        confidence = float(
                            torch.sigmoid(mask_logits[target_pos, 0]).mean().item()
                        )
                        detected = True
                    else:
                        mask = np.zeros((height, width), dtype=np.uint8)
                        bbox_xywh = [0.0, 0.0, 0.0, 0.0]
                        polygon, confidence = [], 0.0
                        cx, cy = width // 2, height // 2
                        detected = False

                    # Overwrite mask file in cache/masks/
                    mask_path = masks_dir / f"mask_{frame_idx + 1:04d}.png"
                    cv2.imwrite(str(mask_path), mask)

                    tracks_data["frames"].append({
                        "object_id": 1,
                        "frame_idx": frame_idx,
                        "center":    [cx, cy],
                        "bbox":      bbox_xywh,
                        "polygon":   polygon,
                        "confidence": confidence,
                        "timestamp": frame_idx / video_fps if video_fps > 0 else 0.0,
                        "detected":  detected,
                    })

                    # Debug overlay (blue-ish)
                    orig_file = frames_dir / f"frame_{frame_idx + 1:04d}.png"
                    frame_bgr = cv2.imread(str(orig_file))
                    if frame_bgr is not None:
                        if detected:
                            colour_mask = np.zeros_like(frame_bgr)
                            colour_mask[mask > 0] = [235, 140, 50]
                            cv2.addWeighted(frame_bgr, 1.0, colour_mask, 0.4, 0, frame_bgr)
                            x1_d = int(bbox_xywh[0])
                            y1_d = int(bbox_xywh[1])
                            x2_d = int(x1_d + bbox_xywh[2])
                            y2_d = int(y1_d + bbox_xywh[3])
                            cv2.rectangle(frame_bgr, (x1_d, y1_d), (x2_d, y2_d), (235, 140, 50), 2)
                            cv2.putText(
                                frame_bgr,
                                f"TRACK ID: 1 - {label} ({confidence:.2f})",
                                (x1_d, max(y1_d - 10, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 140, 50), 2,
                            )
                        cv2.imwrite(str(debug_dir / f"track_{frame_idx + 1:04d}.png"), frame_bgr)

                    if (frame_idx + 1) % 30 == 0 or (frame_idx + 1) == len(frame_files):
                        context.logger.info(
                            f"Propagated frame {frame_idx + 1}/{len(frame_files)}"
                        )

        finally:
            # Always clean up temp JPEG directory
            shutil.rmtree(tmp_dir, ignore_errors=True)

        # ── Save tracks.json ───────────────────────────────────────────
        tracks_file = tracks_dir / "tracks.json"
        with open(tracks_file, "w", encoding="utf-8") as f:
            json.dump(tracks_data, f, indent=4)

        runtime  = time.perf_counter() - start_time
        fps_rate = len(frame_files) / runtime if runtime > 0 else 0.0
        context.logger.info(
            f"SAM2 Tracking complete. Frames: {len(frame_files)} | "
            f"FPS: {fps_rate:.2f} | Runtime: {runtime:.2f}s"
        )

        return StageResult(
            stage_name=self.name,
            success=True,
            runtime_seconds=runtime,
            output_files=[tracks_file],
            metadata={
                "tracks":              tracks_data,
                "num_tracked_objects": 1,
                "average_fps":         fps_rate,
            },
        )

    def is_cached(self, context: PipelineContext) -> bool:
        tracks_file = context.cache_dir / "tracks" / "tracks.json"
        if not tracks_file.exists():
            return False
        try:
            with open(tracks_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return "track_id" in data and len(data.get("frames", [])) > 0
        except Exception:
            return False
