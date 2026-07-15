import json
import time
import cv2
from pathlib import Path
from typing import Dict, Any, List
from adfusion.pipeline.base import BaseStage, PipelineContext, StageResult

class QAStage(BaseStage):
    """Stage 9: Quality Assurance.
    
    Performs integrity, resolution, frame-count, and corruption checks
    on the harmonized output frames. Saves a structured qa_report.json.
    """

    def __init__(self) -> None:
        super().__init__("qa")

    def run(self, context: PipelineContext) -> StageResult:
        start_time = time.perf_counter()
        context.logger.info("Starting quality assurance checks...")

        harmonized_dir = context.cache_dir / "harmonized"
        extraction_res = context.stage_outputs.get("frame_extraction", {})
        metadata = extraction_res.get("metadata", {})
        
        expected_frames = metadata.get("frame_count", 0)
        expected_width = metadata.get("width", 1280)
        expected_height = metadata.get("height", 720)

        if not harmonized_dir.exists():
            return StageResult(
                stage_name=self.name,
                success=False,
                runtime_seconds=time.perf_counter() - start_time,
                error_message="Harmonized frames directory not found. Run harmonization first."
            )

        harmonized_files = sorted(list(harmonized_dir.glob("harmonized_*.png")))
        actual_frames = len(harmonized_files)

        qa_passed = True
        failures: List[str] = []
        warnings: List[str] = []

        # 1. Frame Count Validation
        if expected_frames > 0 and actual_frames != expected_frames:
            qa_passed = False
            failures.append(
                f"Frame count mismatch: expected {expected_frames} frames, but found {actual_frames}."
            )
        elif actual_frames == 0:
            qa_passed = False
            failures.append("No output frames found in the harmonized directory.")

        # 2. Check individual frames for resolution, corruption, and empty sizes
        corrupt_frames = 0
        resolution_mismatch_frames = 0
        empty_frames = 0

        for idx, file_path in enumerate(harmonized_files):
            # Check file size on disk
            if file_path.stat().st_size == 0:
                empty_frames += 1
                qa_passed = False
                failures.append(f"Frame {file_path.name} is empty (0 bytes).")
                continue

            # Read frame headers to check resolution
            frame = cv2.imread(str(file_path))
            if frame is None:
                corrupt_frames += 1
                qa_passed = False
                failures.append(f"Frame {file_path.name} is corrupt and cannot be read.")
                continue

            h, w = frame.shape[:2]
            if expected_frames > 0 and (w != expected_width or h != expected_height):
                resolution_mismatch_frames += 1
                qa_passed = False
                failures.append(
                    f"Frame {file_path.name} resolution mismatch: got {w}x{h}, expected {expected_width}x{expected_height}."
                )

        # 3. Simulated Brand Visibility & Temporal Flicker Metrics (Sprint 0 placeholders)
        brand_visibility_score = 1.0  # mock metric (0.0 to 1.0)
        temporal_flicker_score = 0.02 # mock metric (lower is better)
        
        if corrupt_frames > 0 or empty_frames > 0:
            brand_visibility_score = 0.0

        qa_report: Dict[str, Any] = {
            "qa_passed": qa_passed,
            "metrics": {
                "expected_frames": expected_frames,
                "actual_frames": actual_frames,
                "corrupt_frames": corrupt_frames,
                "empty_frames": empty_frames,
                "resolution_mismatches": resolution_mismatch_frames,
                "brand_visibility_index": brand_visibility_score,
                "temporal_flicker_index": temporal_flicker_score
            },
            "failures": failures,
            "warnings": warnings,
            "timestamp": time.time()
        }

        # Write QA report to the experiment directory
        qa_report_path = context.experiment_dir / "qa_report.json"
        with open(qa_report_path, "w", encoding="utf-8") as f:
            json.dump(qa_report, f, indent=4)

        context.logger.info(f"QA execution completed. Result: {'PASSED' if qa_passed else 'FAILED'}")
        if failures:
            context.logger.warning(f"QA Failures found: {failures[:3]}")

        runtime = time.perf_counter() - start_time
        return StageResult(
            stage_name=self.name,
            success=qa_passed,
            runtime_seconds=runtime,
            output_files=[qa_report_path],
            metadata=qa_report
        )

    def is_cached(self, context: PipelineContext) -> bool:
        # QA is short, but we can cache if qa_report.json exists
        qa_report_path = context.experiment_dir / "qa_report.json"
        if not qa_report_path.exists():
            return False
            
        try:
            with open(qa_report_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("qa_passed", False)
        except Exception:
            return False
