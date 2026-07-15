import time
import cv2
from pathlib import Path
from typing import Dict, Any, List
from adfusion.pipeline.base import BaseStage, PipelineContext, StageResult

class ObjectRemovalStage(BaseStage):
    """Stage 6: Object Removal.
    
    Removes the detected object from the frame to create clean plates.
    Uses OpenCV's Telea inpainting algorithm to fill in the masked region
    with surrounding pixels, producing a real clean plate simulation.
    """

    def __init__(self) -> None:
        super().__init__("remove_object")

    def run(self, context: PipelineContext) -> StageResult:
        start_time = time.perf_counter()
        context.logger.info("Starting object removal (inpainting clean plate)...")

        frames_dir = context.cache_dir / "frames"
        masks_dir = context.cache_dir / "masks"
        cleanplates_dir = context.cache_dir / "cleanplates"
        cleanplates_dir.mkdir(parents=True, exist_ok=True)

        if not frames_dir.exists() or not masks_dir.exists():
            return StageResult(
                stage_name=self.name,
                success=False,
                runtime_seconds=time.perf_counter() - start_time,
                error_message="Required input directories (frames, masks) not found in cache."
            )

        frame_files = sorted(list(frames_dir.glob("frame_*.png")))
        mask_files = sorted(list(masks_dir.glob("mask_*.png")))

        if len(frame_files) != len(mask_files):
            context.logger.warning(
                f"Mismatch in frame count ({len(frame_files)}) and mask count ({len(mask_files)}). "
                "Aligning to minimum."
            )

        num_files = min(len(frame_files), len(mask_files))
        output_files: List[Path] = []

        for idx in range(num_files):
            frame_path = frame_files[idx]
            mask_path = mask_files[idx]

            # Read frame (color) and mask (grayscale)
            frame = cv2.imread(str(frame_path))
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

            if frame is None or mask is None:
                context.logger.error(f"Failed to read file index {idx}: {frame_path} or {mask_path}")
                continue

            # Apply OpenCV inpainting to create the clean plate
            # Telea's method is fast and effective for this size
            clean_plate = cv2.inpaint(frame, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

            # Construct clean plate filename
            clean_plate_filename = f"cleanplate_{idx + 1:04d}.png"
            clean_plate_path = cleanplates_dir / clean_plate_filename

            cv2.imwrite(str(clean_plate_path), clean_plate)
            output_files.append(clean_plate_path)

        context.logger.info(f"Generated {len(output_files)} clean plate frames in {cleanplates_dir}")

        runtime = time.perf_counter() - start_time
        return StageResult(
            stage_name=self.name,
            success=True,
            runtime_seconds=runtime,
            output_files=output_files,
            metadata={"inpainting_algorithm": "INPAINT_TELEA"}
        )

    def is_cached(self, context: PipelineContext) -> bool:
        cleanplates_dir = context.cache_dir / "cleanplates"
        if not cleanplates_dir.exists():
            return False
            
        cleanplates = list(cleanplates_dir.glob("cleanplate_*.png"))
        if not cleanplates:
            return False
            
        if any(c.stat().st_size == 0 for c in cleanplates):
            return False
            
        return True
