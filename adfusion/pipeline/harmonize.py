import time
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from adfusion.pipeline.base import BaseStage, PipelineContext, StageResult

class AIHarmonizationStage(BaseStage):
    """Stage 8: AI Harmonization.
    
    Harmonizes the rendered brand asset with the background plate's lighting.
    Sprint 0 uses visual blending techniques, including mask edge feathering
    and local luminance matching, to simulate photoreal integration.
    """

    def __init__(self) -> None:
        super().__init__("harmonize")

    def run(self, context: PipelineContext) -> StageResult:
        start_time = time.perf_counter()
        context.logger.info("Starting AI harmonization stage...")

        renders_dir = context.cache_dir / "renders"
        cleanplates_dir = context.cache_dir / "cleanplates"
        masks_dir = context.cache_dir / "masks"
        harmonized_dir = context.cache_dir / "harmonized"
        harmonized_dir.mkdir(parents=True, exist_ok=True)

        if not renders_dir.exists() or not cleanplates_dir.exists() or not masks_dir.exists():
            return StageResult(
                stage_name=self.name,
                success=False,
                runtime_seconds=time.perf_counter() - start_time,
                error_message="Required cache directories (renders, cleanplates, masks) not found."
            )

        render_files = sorted(list(renders_dir.glob("render_*.png")))
        cleanplate_files = sorted(list(cleanplates_dir.glob("cleanplate_*.png")))
        mask_files = sorted(list(masks_dir.glob("mask_*.png")))

        num_files = min(len(render_files), len(cleanplate_files), len(mask_files))
        output_files: List[Path] = []

        for idx in range(num_files):
            render = cv2.imread(str(render_files[idx]))
            cleanplate = cv2.imread(str(cleanplate_files[idx]))
            mask = cv2.imread(str(mask_files[idx]), cv2.IMREAD_GRAYSCALE)

            if render is None or cleanplate is None or mask is None:
                context.logger.error(f"Failed to read assets for index {idx}")
                continue

            # Calculate luminance matching: match render badge brightness to cleanplate region
            # Get mean brightness of cleanplate in the masked region
            mean_bg = cv2.mean(cleanplate, mask=mask)[:3]
            
            # Simple brightness/tint harmonization
            # Target average luminance (say, 128 is middle-ground)
            target_lum = np.mean(mean_bg)
            logo_factor = target_lum / 128.0 if target_lum > 0 else 1.0
            
            # Clamp factor to keep colors realistic
            logo_factor = np.clip(logo_factor, 0.6, 1.4)
            
            # Apply color adjustments only within the masked region
            harmonized_frame = render.copy()
            adjusted_render = cv2.convertScaleAbs(render, alpha=logo_factor, beta=0)
            
            # Use mask to copy adjusted pixels
            np.copyto(harmonized_frame, adjusted_render, where=(mask[:, :, None] > 0))

            # Apply Feathering to the edges to avoid harsh boundaries
            # Blur the mask to create a soft alpha transition channel
            soft_mask = cv2.GaussianBlur(mask, (15, 15), 0)
            alpha = soft_mask[:, :, None] / 255.0

            # Soft blend between the harmonized frame and the original cleanplate
            final_blend = (alpha * harmonized_frame + (1.0 - alpha) * cleanplate).astype(np.uint8)

            # Save the final frame
            harmonized_filename = f"harmonized_{idx + 1:04d}.png"
            harmonized_path = harmonized_dir / harmonized_filename
            cv2.imwrite(str(harmonized_path), final_blend)
            output_files.append(harmonized_path)

        context.logger.info(f"Generated {len(output_files)} harmonized frames in {harmonized_dir}")

        runtime = time.perf_counter() - start_time
        return StageResult(
            stage_name=self.name,
            success=True,
            runtime_seconds=runtime,
            output_files=output_files,
            metadata={
                "feather_kernel": "15x15",
                "luminance_factor_applied": float(logo_factor)
            }
        )

    def is_cached(self, context: PipelineContext) -> bool:
        harmonized_dir = context.cache_dir / "harmonized"
        if not harmonized_dir.exists():
            return False
            
        harmonized = list(harmonized_dir.glob("harmonized_*.png"))
        if not harmonized:
            return False
            
        if any(h.stat().st_size == 0 for h in harmonized):
            return False
            
        return True
