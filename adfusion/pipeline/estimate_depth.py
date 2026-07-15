import json
import time
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from adfusion.pipeline.base import BaseStage, PipelineContext, StageResult

class SceneDepthStage(BaseStage):
    """Stage 4: Scene Depth.
    
    Estimates dense scene depth and computes camera poses.
    Produces 16-bit depth map PNGs and a camera_poses.json file.
    """

    def __init__(self) -> None:
        super().__init__("estimate_depth")

    def run(self, context: PipelineContext) -> StageResult:
        start_time = time.perf_counter()
        context.logger.info("Starting depth estimation (placeholder)...")

        # Get metadata from tracking and frames
        tracking_res = context.stage_outputs.get("track_objects", {})
        tracks = tracking_res.get("metadata", {}).get("tracks", {})
        frames_meta = context.stage_outputs.get("frame_extraction", {}).get("metadata", {})
        
        width = frames_meta.get("width", 1280)
        height = frames_meta.get("height", 720)
        frame_count = frames_meta.get("frame_count", 0)

        depth_dir = context.cache_dir / "depth"
        depth_dir.mkdir(parents=True, exist_ok=True)

        frames_track = {f["frame_idx"]: f for f in tracks.get("frames", [])}

        # Find frames in cache
        frames_dir = context.cache_dir / "frames"
        frame_files = sorted(list(frames_dir.glob("frame_*.png")))
        if not frame_files:
            return StageResult(
                stage_name=self.name,
                success=False,
                runtime_seconds=time.perf_counter() - start_time,
                error_message="No frame files found in the cache frames directory."
            )

        if frame_count == 0:
            frame_count = len(frame_files)

        output_files: List[Path] = []
        camera_poses: Dict[str, Any] = {
            "intrinsics": {
                "fx": 1000.0, "fy": 1000.0,
                "cx": width / 2.0, "cy": height / 2.0
            },
            "poses": []
        }

        # Generate a grid for floor depth simulation (vertical gradient)
        y_indices = np.linspace(0, 1, height)
        x_indices = np.linspace(0, 1, width)
        xx, yy = np.meshgrid(x_indices, y_indices)

        for idx, frame_file in enumerate(frame_files):
            # Base floor depth: closer at the bottom (higher intensity), further at the top
            # Scale to 16-bit integer (0 to 65535)
            # Let's say top (sky/background) is 10000, bottom (ground) is 50000
            depth_base = 10000 + 40000 * yy

            # Get target object position if tracked
            track_info = frames_track.get(idx)
            if track_info:
                cx, cy = track_info["center"]
                
                # Create a radial gradient around the object (closer object = higher depth value, e.g. 55000)
                # Distance squared map
                dist_sq = (np.arange(height)[:, None] - cy)**2 + (np.arange(width)[None, :] - cx)**2
                # Standard deviation for Gaussian-like falloff
                sigma = 200.0
                radial_influence = np.exp(-dist_sq / (2.0 * sigma**2))
                
                # Blend the object depth into base depth
                depth_base = depth_base + (60000 - depth_base) * radial_influence

            # Convert to uint16
            depth_map = np.clip(depth_base, 0, 65535).astype(np.uint16)

            # Save depth map as 16-bit PNG
            depth_filename = f"depth_{idx + 1:04d}.png"
            depth_path = depth_dir / depth_filename
            cv2.imwrite(str(depth_path), depth_map)
            output_files.append(depth_path)

            # Generate simulated camera extrinsic matrix (4x4)
            # Simulates a slow panning camera
            pan_angle = 0.05 * np.sin(2 * np.pi * (idx / frame_count))
            cos_a, sin_a = np.cos(pan_angle), np.sin(pan_angle)
            
            # Rotation around Y axis + slight translations
            extrinsic_matrix = [
                [float(cos_a), 0.0, float(sin_a), 0.1 * idx / frame_count],
                [0.0, 1.0, 0.0, 0.0],
                [float(-sin_a), 0.0, float(cos_a), 2.0], # camera is 2 meters back
                [0.0, 0.0, 0.0, 1.0]
            ]
            
            camera_poses["poses"].append({
                "frame_idx": idx,
                "extrinsic": extrinsic_matrix
            })

        # Save camera poses
        poses_file = depth_dir / "camera_poses.json"
        with open(poses_file, "w", encoding="utf-8") as f:
            json.dump(camera_poses, f, indent=4)
        
        output_files.append(poses_file)
        context.logger.info(f"Generated depth maps and camera poses in {depth_dir}")

        runtime = time.perf_counter() - start_time
        return StageResult(
            stage_name=self.name,
            success=True,
            runtime_seconds=runtime,
            output_files=output_files,
            metadata={"camera_poses": camera_poses}
        )

    def is_cached(self, context: PipelineContext) -> bool:
        depth_dir = context.cache_dir / "depth"
        if not depth_dir.exists():
            return False
            
        depth_maps = list(depth_dir.glob("depth_*.png"))
        poses_file = depth_dir / "camera_poses.json"
        
        if not depth_maps or not poses_file.exists():
            return False
            
        return True
