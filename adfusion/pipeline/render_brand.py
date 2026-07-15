import time
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from adfusion.pipeline.base import BaseStage, PipelineContext, StageResult

class BrandRenderingStage(BaseStage):
    """Stage 7: Brand Rendering.
    
    Overlays the brand asset (e.g. logo or textured product) onto the clean plates.
    Utilizes tracking coordinates to scale and position the asset on the video.
    """

    def __init__(self) -> None:
        super().__init__("render_brand")

    def run(self, context: PipelineContext) -> StageResult:
        start_time = time.perf_counter()
        context.logger.info("Starting brand rendering stage...")

        cleanplates_dir = context.cache_dir / "cleanplates"
        renders_dir = context.cache_dir / "renders"
        renders_dir.mkdir(parents=True, exist_ok=True)

        # Get tracking data
        tracking_res = context.stage_outputs.get("track_objects", {})
        tracks = tracking_res.get("metadata", {}).get("tracks", {})
        frames_track = {f["frame_idx"]: f for f in tracks.get("frames", [])}

        if not cleanplates_dir.exists():
            return StageResult(
                stage_name=self.name,
                success=False,
                runtime_seconds=time.perf_counter() - start_time,
                error_message="Clean plates directory not found. Run object removal first."
            )

        cleanplate_files = sorted(list(cleanplates_dir.glob("cleanplate_*.png")))
        if not cleanplate_files:
            return StageResult(
                stage_name=self.name,
                success=False,
                runtime_seconds=time.perf_counter() - start_time,
                error_message="No clean plate frames found."
            )

        # Retrieve brand logo path from config
        logo_path_str = context.config.get("stages", {}).get("render_brand", {}).get("brand_logo_path", "input/logo.png")
        logo_path = context.workspace_dir / logo_path_str

        # Load or generate brand asset
        logo = None
        if logo_path.exists():
            context.logger.info(f"Loading user brand asset: {logo_path}")
            logo = cv2.imread(str(logo_path), cv2.IMREAD_UNCHANGED) # Load with alpha if present
        
        if logo is None:
            context.logger.info("No brand asset found or failed to load. Generating fallback brand logo...")
            # Generate a 200x200 placeholder badge (blue background, white text)
            logo = np.zeros((200, 200, 4), dtype=np.uint8)
            cv2.circle(logo, (100, 100), 90, (235, 140, 50, 255), -1) # Blue badge with alpha
            cv2.putText(
                logo, "AD", (50, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255, 255), 5, cv2.LINE_AA
            )
            cv2.putText(
                logo, "FUSION", (30, 155),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255, 255), 2, cv2.LINE_AA
            )

        output_files: List[Path] = []

        for idx, plate_path in enumerate(cleanplate_files):
            frame = cv2.imread(str(plate_path))
            if frame is None:
                context.logger.error(f"Failed to read cleanplate: {plate_path}")
                continue

            track_info = frames_track.get(idx)
            if track_info:
                cx, cy = track_info["center"]
                bbox = track_info["bbox"] # [x, y, w, h]
                w, h = bbox[2], bbox[3]

                # Convert to native Python integers
                w = int(round(float(w)))
                h = int(round(float(h)))

                # Ensure dimensions are positive and non-zero
                w = max(20, w)
                h = max(20, h)

                context.logger.info(f"Rendering logo at frame {idx}: w={w} (type={type(w)}), h={h} (type={type(h)})")

                # Verify logo is a valid numpy ndarray
                if not isinstance(logo, np.ndarray):
                    raise TypeError(f"Brand asset 'logo' must be a numpy ndarray, got {type(logo)}")

                # Resize the logo to fit the bounding box
                resized_logo = cv2.resize(logo, (w, h), interpolation=cv2.INTER_AREA)

                # Overlay the logo centered at (cx, cy)
                x_start = cx - w // 2
                y_start = cy - h // 2

                # Clip coordinates to frame boundary
                frame_h, frame_w = frame.shape[:2]
                
                # Compute overlap coordinates
                x1_l = max(0, -x_start)
                y1_l = max(0, -y_start)
                x2_l = min(w, frame_w - x_start)
                y2_l = min(h, frame_h - y_start)

                x1_f = max(0, x_start)
                y1_f = max(0, y_start)
                x2_f = min(frame_w, x_start + w)
                y2_f = min(frame_h, y_start + h)

                if (x2_f > x1_f) and (y2_f > y1_f):
                    sub_logo = resized_logo[y1_l:y2_l, x1_l:x2_l]
                    sub_frame = frame[y1_f:y2_f, x1_f:x2_f]

                    # Alpha blending if logo has 4 channels
                    if sub_logo.shape[2] == 4:
                        alpha = sub_logo[:, :, 3:4] / 255.0
                        rgb_logo = sub_logo[:, :, :3]
                        blended = (1.0 - alpha) * sub_frame + alpha * rgb_logo
                        frame[y1_f:y2_f, x1_f:x2_f] = blended.astype(np.uint8)
                    else:
                        frame[y1_f:y2_f, x1_f:x2_f] = sub_logo

            # Save the frame
            render_filename = f"render_{idx + 1:04d}.png"
            render_path = renders_dir / render_filename
            cv2.imwrite(str(render_path), frame)
            output_files.append(render_path)

        context.logger.info(f"Generated {len(output_files)} rendered frames in {renders_dir}")

        runtime = time.perf_counter() - start_time
        return StageResult(
            stage_name=self.name,
            success=True,
            runtime_seconds=runtime,
            output_files=output_files,
            metadata={"brand_logo_used": str(logo_path) if logo_path.exists() else "dynamic_fallback"}
        )

    def is_cached(self, context: PipelineContext) -> bool:
        renders_dir = context.cache_dir / "renders"
        if not renders_dir.exists():
            return False
            
        renders = list(renders_dir.glob("render_*.png"))
        if not renders:
            return False
            
        if any(r.stat().st_size == 0 for r in renders):
            return False
            
        return True
