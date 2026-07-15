import json
import time
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from adfusion.pipeline.base import BaseStage, PipelineContext, StageResult

class VideoTrackingStage(BaseStage):
    """Stage 3: Video Tracking.
    
    Loads detection outputs and associates bounding boxes across frames using
    an Intersection-over-Union (IoU) tracker. Outputs tracks.json documenting
    object trajectories and saves tracking debug visualizations.
    """

    def __init__(self) -> None:
        super().__init__("track_objects")

    def _get_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculates Intersection over Union (IoU) of two bounding boxes in [x1, y1, x2, y2] format."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Check for intersection
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)

        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0

        inter_area = (xi2 - xi1) * (yi2 - yi1)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area

        if union_area <= 0:
            return 0.0

        return inter_area / union_area

    def run(self, context: PipelineContext) -> StageResult:
        start_time = time.perf_counter()
        context.logger.info("=== Starting Real Object Tracking (Official SAM 3 Predictor) ===")

        # Retrieve configurations
        sam3_config = context.config.get("sam3", {})
        checkpoint_rel = sam3_config.get("checkpoint_multiplex", "models/sam3.1_multiplex.pt")
        checkpoint_path = context.workspace_dir / checkpoint_rel
        device_config = sam3_config.get("device", "cuda")
        conf_threshold = float(sam3_config.get("confidence_threshold", 0.25))

        # Retrieve detection outputs
        detection_res = context.stage_outputs.get("object_detection", {})
        detection_metadata = detection_res.get("metadata", {})
        label = detection_metadata.get("target_class", "cup")

        # Paths
        frames_dir = context.cache_dir / "frames"
        masks_dir = context.cache_dir / "masks"
        tracks_dir = context.cache_dir / "tracks"
        debug_dir = context.cache_dir / "debug"

        tracks_dir.mkdir(parents=True, exist_ok=True)
        debug_dir.mkdir(parents=True, exist_ok=True)

        frame_files = sorted(list(frames_dir.glob("frame_*.png")))
        if not frame_files:
            return StageResult(
                stage_name=self.name,
                success=False,
                runtime_seconds=time.perf_counter() - start_time,
                error_message="Frames directory is empty. Run frame extraction first."
            )

        # Retrieve video metadata
        extraction_res = context.stage_outputs.get("frame_extraction", {})
        extraction_metadata = extraction_res.get("metadata", {})
        fps = extraction_metadata.get("fps", 30.0)
        width = extraction_metadata.get("width", 1280)
        height = extraction_metadata.get("height", 720)

        # Load SAM 3 Video Predictor using official API
        context.logger.info("Initializing Meta SAM 3 Video Predictor...")
        try:
            import torch
            from sam3.model_builder import build_sam3_predictor

            # Determine device
            device = "cpu"
            if device_config == "cuda" and torch.cuda.is_available():
                device = "cuda"

            predictor = build_sam3_predictor(
                checkpoint_path=str(checkpoint_path),
                version="sam3.1",
                compile=False
            )
            context.logger.info("SAM 3 Video Predictor loaded successfully.")
        except Exception as e:
            return StageResult(
                stage_name=self.name,
                success=False,
                runtime_seconds=time.perf_counter() - start_time,
                error_message=f"Failed to load SAM 3 Video Predictor: {e}"
            )

        # Run Video Session Tracking
        tracks_data = {
            "track_id": 1,
            "label": label,
            "frames": []
        }

        session_id = None
        try:
            # 1. Start Session
            session_resp = predictor.handle_request({
                "type": "start_session",
                "resource_path": str(frames_dir)
            })
            session_id = session_resp["session_id"]
            context.logger.info(f"Initialized SAM 3 tracking session: {session_id}")

            # 2. Add Prompt (register text query at frame 0 with object ID 1)
            predictor.handle_request({
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": 0,
                "text": label,
                "obj_id": 1,
                "output_prob_thresh": conf_threshold
            })

            # 3. Propagate in video forward
            stream = predictor.handle_stream_request({
                "type": "propagate_in_video",
                "session_id": session_id,
                "propagation_direction": "forward"
            })

            # 4. Process streaming frames output
            for frame_data in stream:
                frame_idx = frame_data["frame_index"]
                outputs = frame_data["outputs"]

                obj_ids = outputs["out_obj_ids"]
                probs = outputs["out_probs"]
                boxes_xywh = outputs["out_boxes_xywh"]
                binary_masks = outputs["out_binary_masks"]

                # Check if target object_id=1 is present in the frame tracking output
                target_idx = -1
                if obj_ids is not None:
                    obj_ids_list = obj_ids.tolist() if hasattr(obj_ids, "tolist") else list(obj_ids)
                    if 1 in obj_ids_list:
                        target_idx = obj_ids_list.index(1)

                if target_idx != -1:
                    # Retrieve the binary mask array
                    mask_bool = binary_masks[target_idx]
                    mask = (mask_bool * 255).astype(np.uint8)

                    # Bounding Box (SAM 3 returns normalized center-x, center_y, width, height)
                    cx_norm, cy_norm, w_norm, h_norm = boxes_xywh[target_idx]
                    w = float(w_norm * width)
                    h = float(h_norm * height)
                    x_min = float((cx_norm - w_norm / 2.0) * width)
                    y_min = float((cy_norm - h_norm / 2.0) * height)
                    bbox_xywh = [x_min, y_min, w, h]

                    # Extract polygon contours
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    polygon = []
                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)
                        polygon = largest_contour.reshape(-1, 2).tolist()

                    confidence = float(probs[target_idx])
                    cx = int(x_min + w / 2)
                    cy = int(y_min + h / 2)
                    detected = True
                else:
                    # Lost track / Occluded: write a blank placeholder mask
                    mask = np.zeros((height, width), dtype=np.uint8)
                    bbox_xywh = [0.0, 0.0, 0.0, 0.0]
                    polygon = []
                    confidence = 0.0
                    cx, cy = int(width / 2), int(height / 2)
                    detected = False

                # Overwrite binary mask file in cache/masks
                mask_filename = f"mask_{frame_idx + 1:04d}.png"
                mask_path = masks_dir / mask_filename
                cv2.imwrite(str(mask_path), mask)

                # Save frame tracks metadata
                tracks_data["frames"].append({
                    "object_id": 1,
                    "frame_idx": frame_idx,
                    "center": [cx, cy],
                    "bbox": bbox_xywh,
                    "polygon": polygon,
                    "confidence": confidence,
                    "timestamp": frame_idx / fps if fps > 0 else 0.0
                })

                # Create visual debug overlay representing tracking (Blue-ish color)
                frame_file = frames_dir / f"frame_{frame_idx + 1:04d}.png"
                frame = cv2.imread(str(frame_file))
                if frame is not None:
                    if detected:
                        color_mask = np.zeros_like(frame)
                        color_mask[mask > 0] = [235, 140, 50] # Blue-ish mask overlay (BGR: [235, 140, 50])
                        cv2.addWeighted(frame, 1.0, color_mask, 0.4, 0, frame)
                        
                        # Draw bounding box
                        cv2.rectangle(
                            frame,
                            (int(x_min), int(y_min)),
                            (int(x_min + w), int(y_min + h)),
                            (235, 140, 50), 2
                        )
                        
                        # Draw label
                        label_str = f"TRACK ID: 1 - {label} ({confidence:.2f})"
                        cv2.putText(
                            frame, label_str, (int(x_min), int(y_min) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 140, 50), 2
                        )
                    
                    # Write debug image
                    cv2.imwrite(str(debug_dir / f"track_{frame_idx + 1:04d}.png"), frame)

                # Log progress
                if (frame_idx + 1) % 30 == 0 or (frame_idx + 1) == len(frame_files):
                    context.logger.info(f"Propagated frame {frame_idx + 1}/{len(frame_files)}")

        finally:
            # 5. Clean up and close the tracking session
            if session_id is not None:
                try:
                    predictor.handle_request({
                        "type": "close_session",
                        "session_id": session_id
                    })
                    context.logger.info(f"Closed SAM 3 session {session_id} successfully.")
                except Exception as ce:
                    context.logger.warning(f"Error while closing session: {ce}")

        # Save compiled results to tracks.json
        tracks_file = tracks_dir / "tracks.json"
        with open(tracks_file, "w", encoding="utf-8") as f:
            json.dump(tracks_data, f, indent=4)

        runtime = time.perf_counter() - start_time
        fps_rate = len(frame_files) / runtime if runtime > 0 else 0.0
        context.logger.info(f"SAM 3 Tracking stage completed in {runtime:.2f}s ({fps_rate:.2f} FPS).")

        return StageResult(
            stage_name=self.name,
            success=True,
            runtime_seconds=runtime,
            output_files=[tracks_file],
            metadata={
                "tracks": tracks_data,
                "num_tracked_objects": 1,
                "average_fps": fps_rate
            }
        )

    def is_cached(self, context: PipelineContext) -> bool:
        tracks_dir = context.cache_dir / "tracks"
        tracks_file = tracks_dir / "tracks.json"
        
        if not tracks_file.exists():
            return False
            
        try:
            with open(tracks_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return "track_id" in data and len(data.get("frames", [])) > 0
        except Exception:
            return False
