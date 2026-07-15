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
    
    Loads the official Meta SAM 3 model (via Ultralytics) using a locally cached
    checkpoint. Automatically downloads the checkpoint from a public Hugging Face
    mirror if missing, runs inference on CUDA/CPU, saves binary masks, and
    generates visual debug overlays under cache/debug/.
    """

    def __init__(self) -> None:
        super().__init__("object_detection")

    def _ensure_checkpoint(self, checkpoint_path: Path, context: PipelineContext) -> None:
        """Downloads the SAM 3 checkpoint from a public mirror if it doesn't exist."""
        if checkpoint_path.exists():
            context.logger.info(f"SAM 3 checkpoint already exists at: {checkpoint_path}")
            return

        context.logger.info(f"SAM 3 checkpoint not found at {checkpoint_path}")
        context.logger.info("Downloading checkpoint (~3.45 GB) from public Hugging Face mirror (1038lab/sam3)...")
        
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Import huggingface_hub inside method to prevent global dependencies
        try:
            from huggingface_hub import hf_hub_download
            
            # Download the sam3.pt weight file
            downloaded = hf_hub_download(
                repo_id="1038lab/sam3",
                filename="sam3.pt",
                local_dir=str(checkpoint_path.parent),
                local_dir_use_symlinks=False
            )
            
            downloaded_path = Path(downloaded)
            if downloaded_path.exists() and downloaded_path.resolve() != checkpoint_path.resolve():
                # If downloaded to a subfolder or different name, rename it to target path
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
                os.rename(downloaded_path, checkpoint_path)
                
            context.logger.info(f"SAM 3 checkpoint successfully downloaded and saved to {checkpoint_path}")
            
        except Exception as e:
            context.logger.error(f"Failed to automatically download SAM 3 checkpoint: {e}")
            raise RuntimeError(
                f"SAM 3 checkpoint is missing and automatic download failed: {e}. "
                f"Please place the 'sam3.pt' file manually at: {checkpoint_path}"
            ) from e

    def run(self, context: PipelineContext) -> StageResult:
        start_time = time.perf_counter()
        context.logger.info("=== Starting Real SAM 3 Object Detection ===")

        # Retrieve configurations
        sam3_config = context.config.get("sam3", {})
        checkpoint_rel = sam3_config.get("checkpoint", "models/sam3.pt")
        checkpoint_path = context.workspace_dir / checkpoint_rel
        device_config = sam3_config.get("device", "cuda")
        conf_threshold = float(sam3_config.get("confidence_threshold", 0.25))

        target_class = context.config.get("stages", {}).get("object_detection", {}).get("target_class", "cup")

        # 1. Download checkpoint if missing
        self._ensure_checkpoint(checkpoint_path, context)

        # 2. Setup Device with CPU Fallback
        device = "cpu"
        if device_config == "cuda":
            if torch.cuda.is_available():
                device = "cuda"
            else:
                context.logger.warning("CUDA is requested in config but not available. Falling back to CPU.")
        
        # 3. Load SAM 3 Model using official Meta API
        context.logger.info("Initializing Meta SAM 3 model using official API...")
        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            from PIL import Image

            # Initialize model via official builder
            model = build_sam3_image_model(
                checkpoint_path=str(checkpoint_path),
                device=device,
                compile=False
            )
            processor = Sam3Processor(model)
            processor.set_confidence_threshold(conf_threshold)
            
            context.logger.info(f"SAM 3 model loaded successfully on device: {device}")
        except Exception as e:
            return StageResult(
                stage_name=self.name,
                success=False,
                runtime_seconds=time.perf_counter() - start_time,
                error_message=f"Failed to load SAM 3 model: {e}"
            )

        # Retrieve source frames metadata
        extraction_res = context.stage_outputs.get("frame_extraction", {})
        metadata = extraction_res.get("metadata", {})
        width = metadata.get("width", 1280)
        height = metadata.get("height", 720)

        frames_dir = context.cache_dir / "frames"
        masks_dir = context.cache_dir / "masks"
        debug_dir = context.cache_dir / "debug"

        masks_dir.mkdir(parents=True, exist_ok=True)
        debug_dir.mkdir(parents=True, exist_ok=True)

        frame_files = sorted(list(frames_dir.glob("frame_*.png")))
        if not frame_files:
            return StageResult(
                stage_name=self.name,
                success=False,
                runtime_seconds=time.perf_counter() - start_time,
                error_message="No frames found in the cache frames directory. Run frame extraction first."
            )

        output_files: List[Path] = []
        masks_meta: List[Dict[str, Any]] = []

        context.logger.info(f"Processing {len(frame_files)} frames for prompt: '{target_class}'")

        # Track VRAM utilization if on CUDA
        max_vram_bytes = 0
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        for idx, frame_file in enumerate(frame_files):
            # Load frame to check for corruptions/empty images
            frame = cv2.imread(str(frame_file))
            if frame is None or frame.size == 0:
                context.logger.error(f"Frame {frame_file.name} is empty or corrupted.")
                # Create a blank mask to maintain continuity
                mask = np.zeros((height, width), dtype=np.uint8)
                mask_filename = f"mask_{idx + 1:04d}.png"
                mask_path = masks_dir / mask_filename
                cv2.imwrite(str(mask_path), mask)
                output_files.append(mask_path)
                masks_meta.append({
                    "frame_idx": idx,
                    "center": [width // 2, height // 2],
                    "radius": 0,
                    "bbox": [0, 0, 0, 0],
                    "polygon": [],
                    "confidence": 0.0,
                    "detected": False
                })
                continue

            # Run SAM 3 Inference using official processor
            try:
                # Open PIL image
                pil_img = Image.open(str(frame_file))
                
                # Run prediction
                inference_state = processor.set_image(pil_img)
                inference_state = processor.set_text_prompt(state=inference_state, prompt=target_class)
                
                masks_tensor = inference_state.get("masks")
                boxes_tensor = inference_state.get("boxes")
                scores_tensor = inference_state.get("scores")
            except RuntimeError as re:
                # Handle CUDA OOM or other exceptions and fallback to CPU dynamically
                if "out of memory" in str(re).lower() and device == "cuda":
                    context.logger.warning("CUDA Out Of Memory encountered. Falling back to CPU execution.")
                    device = "cpu"
                    model = model.to("cpu")
                    processor = Sam3Processor(model)
                    processor.set_confidence_threshold(conf_threshold)
                    
                    # Retry on CPU
                    pil_img = Image.open(str(frame_file))
                    inference_state = processor.set_image(pil_img)
                    inference_state = processor.set_text_prompt(state=inference_state, prompt=target_class)
                    
                    masks_tensor = inference_state.get("masks")
                    boxes_tensor = inference_state.get("boxes")
                    scores_tensor = inference_state.get("scores")
                else:
                    raise re

            # Extract prediction properties
            if masks_tensor is not None and masks_tensor.shape[0] > 0:
                best_idx = int(torch.argmax(scores_tensor).item())
                
                # Get binary mask and convert to 0-255 uint8 numpy array
                mask = (masks_tensor[best_idx].cpu().numpy() * 255).astype(np.uint8)
                
                # Ensure mask matches video resolution
                if mask.shape[:2] != (height, width):
                    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

                # Get Bounding Box [x1, y1, x2, y2]
                bbox = boxes_tensor[best_idx].cpu().numpy().tolist()
                
                # Get Polygon Points (list of coordinates [[x, y], ...])
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                polygon = []
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    polygon = largest_contour.reshape(-1, 2).tolist()
                
                # Calculate center and diameter (simulated radius)
                x1, y1, x2, y2 = bbox
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                radius = int(max(x2 - x1, y2 - y1) / 2)
                
                confidence = float(scores_tensor[best_idx].item())
                detected = True
            else:
                # No object detected: output a blank mask
                mask = np.zeros((height, width), dtype=np.uint8)
                bbox = [0, 0, 0, 0]
                polygon = []
                cx, cy = width // 2, height // 2
                radius = 0
                confidence = 0.0
                detected = False

            # Write binary mask to disk
            mask_filename = f"mask_{idx + 1:04d}.png"
            mask_path = masks_dir / mask_filename
            cv2.imwrite(str(mask_path), mask)
            output_files.append(mask_path)

            # Store detection metadata
            masks_meta.append({
                "frame_idx": idx,
                "center": [cx, cy],
                "radius": radius,
                "bbox": bbox, # [x1, y1, x2, y2]
                "polygon": polygon,
                "confidence": confidence,
                "detected": detected
            })

            # Create visual debug overlays
            debug_frame = frame.copy()
            if detected:
                # Draw semi-transparent green mask
                color_mask = np.zeros_like(frame)
                color_mask[mask > 0] = [0, 255, 0] # Green mask overlay
                cv2.addWeighted(debug_frame, 1.0, color_mask, 0.4, 0, debug_frame)

                # Draw bounding box
                cv2.rectangle(
                    debug_frame,
                    (int(bbox[0]), int(bbox[1])),
                    (int(bbox[2]), int(bbox[3])),
                    (0, 255, 0), 2
                )
                
                # Draw text label
                label = f"DET: {target_class} ({confidence:.2f})"
                cv2.putText(
                    debug_frame, label, (int(bbox[0]), int(bbox[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )

            # Write visual debug frame to cache/debug/
            debug_filename = f"detect_{idx + 1:04d}.png"
            cv2.imwrite(str(debug_dir / debug_filename), debug_frame)

            # Log progress every 30 frames
            if (idx + 1) % 30 == 0 or (idx + 1) == len(frame_files):
                context.logger.info(f"Processed frame {idx + 1}/{len(frame_files)}")

        # Calculate peak VRAM usage
        peak_vram_mb = 0.0
        if device == "cuda":
            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            context.logger.info(f"Peak GPU VRAM allocated by PyTorch: {peak_vram_mb:.2f} MB")

        runtime = time.perf_counter() - start_time
        fps = len(frame_files) / runtime if runtime > 0 else 0.0
        
        context.logger.info(f"SAM 3 Detection stage finished. Avg FPS: {fps:.2f}. Total processed frames: {len(frame_files)}")

        return StageResult(
            stage_name=self.name,
            success=True,
            runtime_seconds=runtime,
            output_files=output_files,
            metadata={
                "masks_metadata": masks_meta,
                "target_class": target_class,
                "device": device,
                "peak_vram_mb": peak_vram_mb,
                "average_fps": fps,
                "checkpoint_version": "Meta SAM 3.0 Base"
            }
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
