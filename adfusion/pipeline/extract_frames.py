import os
import subprocess
import time
import cv2
from pathlib import Path
from typing import Dict, Any, List
from adfusion.pipeline.base import BaseStage, PipelineContext, StageResult

class FrameExtractionStage(BaseStage):
    """Stage 1: Frame Extraction.
    
    Extracts individual frames from the input video file and saves them to the
    cache directory, while also extracting the audio track as a WAV file.
    """

    def __init__(self) -> None:
        super().__init__("frame_extraction")

    def run(self, context: PipelineContext) -> StageResult:
        start_time = time.perf_counter()
        context.logger.info("Starting frame extraction...")

        video_path_str = context.config["pipeline"]["video_path"]
        video_path = context.workspace_dir / video_path_str

        if not video_path.exists():
            return StageResult(
                stage_name=self.name,
                success=False,
                runtime_seconds=time.perf_counter() - start_time,
                error_message=f"Input video file not found: {video_path}"
            )

        # Create frames cache directory
        frames_dir = context.cache_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        # Open video file
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return StageResult(
                stage_name=self.name,
                success=False,
                runtime_seconds=time.perf_counter() - start_time,
                error_message=f"Failed to open video file: {video_path}"
            )

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        context.logger.info(
            f"Video metadata - FPS: {fps:.2f}, Frames: {frame_count}, Resolution: {width}x{height}"
        )

        output_files: List[Path] = []
        frame_idx = 0
        timestamps: List[float] = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Calculate current frame timestamp in seconds
            timestamp = frame_idx / fps if fps > 0 else 0.0
            timestamps.append(timestamp)

            # Format frame filename (zero-padded, 1-indexed)
            frame_filename = f"frame_{frame_idx + 1:04d}.png"
            frame_path = frames_dir / frame_filename
            
            # Write frame to disk
            cv2.imwrite(str(frame_path), frame)
            output_files.append(frame_path)
            frame_idx += 1

        cap.release()
        context.logger.info(f"Extracted {frame_idx} frames to {frames_dir}")

        # Extract audio using FFmpeg
        audio_path = context.cache_dir / "audio.wav"
        has_audio = False
        
        try:
            # We run FFmpeg command: ffmpeg -y -i <video> -vn -acodec pcm_s16le -ar 44100 <audio_path>
            # -vn: disable video, -y: overwrite output, -loglevel error: suppress noise
            context.logger.info("Extracting audio from video using FFmpeg...")
            cmd = [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", "44100",
                "-loglevel", "error",
                str(audio_path)
            ]
            
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode == 0 and audio_path.exists() and audio_path.stat().st_size > 0:
                context.logger.info(f"Successfully extracted audio to {audio_path}")
                output_files.append(audio_path)
                has_audio = True
            else:
                context.logger.warning(
                    f"FFmpeg audio extraction failed or no audio stream found. Return code: {result.returncode}. "
                    f"Stderr: {result.stderr.strip()}"
                )
        except Exception as e:
            context.logger.warning(f"Failed to execute FFmpeg for audio extraction: {e}")

        # Save metadata to context
        metadata = {
            "fps": fps,
            "frame_count": frame_idx,
            "width": width,
            "height": height,
            "timestamps": timestamps,
            "has_audio": has_audio,
            "audio_path": str(audio_path) if has_audio else None
        }
        
        runtime = time.perf_counter() - start_time
        return StageResult(
            stage_name=self.name,
            success=True,
            runtime_seconds=runtime,
            output_files=output_files,
            metadata=metadata
        )

    def is_cached(self, context: PipelineContext) -> bool:
        frames_dir = context.cache_dir / "frames"
        if not frames_dir.exists():
            return False
            
        # Check if there are png files in the folder
        frames = list(frames_dir.glob("frame_*.png"))
        if not frames:
            return False
            
        # Check if they are valid files (non-empty)
        if any(f.stat().st_size == 0 for f in frames):
            return False
            
        return True
