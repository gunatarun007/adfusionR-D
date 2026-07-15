import subprocess
import time
from pathlib import Path
from typing import Dict, Any, List
from adfusion.pipeline.base import BaseStage, PipelineContext, StageResult

class VideoExportStage(BaseStage):
    """Stage 10: Video Export.
    
    Compiles the final harmonized frames into a single MP4 video file
    and multiplexes the original audio track back in using FFmpeg.
    """

    def __init__(self) -> None:
        super().__init__("export_video")

    def run(self, context: PipelineContext) -> StageResult:
        start_time = time.perf_counter()
        context.logger.info("Starting video export...")

        harmonized_dir = context.cache_dir / "harmonized"
        output_path_str = context.config["pipeline"]["output_path"]
        output_path = context.workspace_dir / output_path_str

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not harmonized_dir.exists():
            return StageResult(
                stage_name=self.name,
                success=False,
                runtime_seconds=time.perf_counter() - start_time,
                error_message="Harmonized frames directory not found. Run harmonization first."
            )

        # Retrieve FPS and audio parameters from Stage 1 Frame Extraction metadata
        extraction_res = context.stage_outputs.get("frame_extraction", {})
        metadata = extraction_res.get("metadata", {})
        fps = metadata.get("fps", 30.0)
        has_audio = metadata.get("has_audio", False)
        audio_path = metadata.get("audio_path")

        # Fallback verification of audio file existence
        if has_audio and audio_path:
            actual_audio_path = Path(audio_path)
            if not actual_audio_path.exists():
                has_audio = False
                context.logger.warning(
                    f"Metadata indicated audio exists, but audio file not found at {actual_audio_path}. "
                    "Exporting video without audio."
                )

        # Construct FFmpeg command
        # Read from sequence: -i harmonized_%04d.png
        # Encode video: -c:v libx264 -pix_fmt yuv420p
        # If audio exists, include it and encode: -c:a aac -shortest
        input_pattern = str(harmonized_dir / "harmonized_%04d.png")

        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", input_pattern
        ]

        if has_audio and audio_path:
            cmd.extend(["-i", str(audio_path)])
            cmd.extend([
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-shortest"
            ])
        else:
            cmd.extend([
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p"
            ])

        cmd.extend(["-loglevel", "error", str(output_path)])

        context.logger.info(f"Executing FFmpeg video compile: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
                context.logger.info(f"Successfully exported final video to: {output_path}")
            else:
                return StageResult(
                    stage_name=self.name,
                    success=False,
                    runtime_seconds=time.perf_counter() - start_time,
                    error_message=f"FFmpeg failed with exit code {result.returncode}. Stderr: {result.stderr.strip()}"
                )
        except Exception as e:
            return StageResult(
                stage_name=self.name,
                success=False,
                runtime_seconds=time.perf_counter() - start_time,
                error_message=f"Failed to execute FFmpeg compiler command: {e}"
            )

        runtime = time.perf_counter() - start_time
        return StageResult(
            stage_name=self.name,
            success=True,
            runtime_seconds=runtime,
            output_files=[output_path],
            metadata={
                "video_fps": fps,
                "muxed_audio": has_audio,
                "output_video_path": str(output_path)
            }
        )

    def is_cached(self, context: PipelineContext) -> bool:
        # Since this is the final export, if the output file already exists, we can resume,
        # but usually export is run to finalize. Let's return True if output file exists.
        output_path_str = context.config["pipeline"]["output_path"]
        output_path = context.workspace_dir / output_path_str
        return output_path.exists() and output_path.stat().st_size > 0
