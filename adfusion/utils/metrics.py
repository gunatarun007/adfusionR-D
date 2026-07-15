import time
import json
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

class PerformanceTracker:
    """Tracks execution time and hardware resource statistics for pipeline runs."""

    def __init__(self) -> None:
        self.start_times: Dict[str, float] = {}
        self.stage_timings: Dict[str, float] = {}
        self.metrics: Dict[str, Any] = {
            "stages": {},
            "hardware": {},
            "total_runtime_seconds": 0.0
        }
        self.total_start_time: float = 0.0

    def start_total(self) -> None:
        """Starts timing the overall pipeline run."""
        self.total_start_time = time.perf_counter()

    def end_total(self) -> None:
        """Ends timing the overall pipeline run and calculates total duration."""
        if self.total_start_time > 0:
            self.metrics["total_runtime_seconds"] = time.perf_counter() - self.total_start_time

    def start_stage(self, stage_name: str) -> None:
        """Starts tracking timing for a specific stage.

        Args:
            stage_name: Unique identifier of the stage.
        """
        self.start_times[stage_name] = time.perf_counter()

    def end_stage(self, stage_name: str) -> float:
        """Ends tracking timing for a specific stage.

        Args:
            stage_name: Unique identifier of the stage.

        Returns:
            The duration of the stage in seconds.
        """
        if stage_name not in self.start_times:
            return 0.0
        
        duration = time.perf_counter() - self.start_times[stage_name]
        self.stage_timings[stage_name] = duration
        
        # Get mock/actual hardware details
        gpu_info = self._get_gpu_memory_usage()
        
        self.metrics["stages"][stage_name] = {
            "runtime_seconds": duration,
            "gpu_memory_used_mb": gpu_info.get("used_mb", 0.0) if gpu_info else 0.0
        }
        
        return duration

    def _get_gpu_memory_usage(self) -> Optional[Dict[str, Any]]:
        """Queries GPU memory usage using nvidia-smi if available, returns mock if not.

        Returns:
            A dictionary containing GPU utilization metrics or None if not queryable.
        """
        # First check if nvidia-smi is in PATH
        if not shutil.which("nvidia-smi"):
            return {"used_mb": 0.0, "status": "no_nvidia_smi"}
            
        try:
            # Query nvidia-smi for memory utilization
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,nounits,noheader"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            lines = result.stdout.strip().split("\n")
            if lines:
                used, total = map(float, lines[0].split(","))
                return {
                    "used_mb": used,
                    "total_mb": total,
                    "status": "success"
                }
        except Exception:
            pass
            
        return {"used_mb": 0.0, "status": "query_failed"}

    def save_metrics(self, output_path: Path) -> None:
        """Saves gathered metrics to a JSON file.

        Args:
            output_path: Location of the metrics JSON file.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Query overall hardware profile at the end
        self.metrics["hardware"] = {
            "gpu": self._get_gpu_memory_usage()
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, indent=4)
