import argparse
import json
import logging
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional

from adfusion.utils.logging import setup_logger
from adfusion.utils.metrics import PerformanceTracker
from adfusion.pipeline.base import PipelineContext, StageResult, BaseStage
from adfusion.pipeline import (
    FrameExtractionStage,
    ObjectDetectionStage,
    VideoTrackingStage,
    SceneDepthStage,
    ObjectReconstructionStage,
    ObjectRemovalStage,
    BrandRenderingStage,
    AIHarmonizationStage,
    QAStage,
    VideoExportStage
)

# Map of stage identifier to Stage Class
STAGE_MAP: Dict[str, BaseStage] = {
    "frame_extraction": FrameExtractionStage(),
    "object_detection": ObjectDetectionStage(),
    "track_objects": VideoTrackingStage(),
    "estimate_depth": SceneDepthStage(),
    "reconstruct_object": ObjectReconstructionStage(),
    "remove_object": ObjectRemovalStage(),
    "render_brand": BrandRenderingStage(),
    "harmonize": AIHarmonizationStage(),
    "qa": QAStage(),
    "export_video": VideoExportStage()
}

# Short aliases for stages to support quick CLI execution
STAGE_ALIASES: Dict[str, str] = {
    "extract": "frame_extraction",
    "frames": "frame_extraction",
    "detect": "object_detection",
    "detection": "object_detection",
    "track": "track_objects",
    "tracking": "track_objects",
    "depth": "estimate_depth",
    "reconstruct": "reconstruct_object",
    "mesh": "reconstruct_object",
    "remove": "remove_object",
    "cleanplate": "remove_object",
    "render": "render_brand",
    "harmonization": "harmonize",
    "export": "export_video"
}

def resolve_stage_name(name: str) -> Optional[str]:
    """Resolves a short alias or full name to the standard stage name."""
    name_clean = name.lower().strip()
    if name_clean in STAGE_MAP:
        return name_clean
    if name_clean in STAGE_ALIASES:
        return STAGE_ALIASES[name_clean]
    return None

def load_config(config_path: Path) -> Dict[str, Any]:
    """Loads configuration from a YAML file."""
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}", file=sys.stderr)
        sys.exit(1)
        
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_next_experiment_dir(experiments_root: Path, custom_name: Optional[str] = None) -> Path:
    """Creates and returns the next auto-incremented experiment directory."""
    experiments_root.mkdir(parents=True, exist_ok=True)
    
    if custom_name:
        exp_dir = experiments_root / custom_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir

    # Find the next experiment_XXX folder
    existing_dirs = list(experiments_root.glob("experiment_*"))
    max_num = 0
    for d in existing_dirs:
        if d.is_dir():
            name = d.name
            try:
                num = int(name.split("_")[1])
                if num > max_num:
                    max_num = num
            except (IndexError, ValueError):
                pass
                
    next_num = max_num + 1
    exp_dir = experiments_root / f"experiment_{next_num:03d}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir

def save_stage_metadata(cache_dir: Path, stage_name: str, metadata: Dict[str, Any]) -> None:
    """Saves a stage's metadata to a JSON file in cache for resumability."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_file = cache_dir / f"metadata_{stage_name}.json"
    try:
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
    except Exception as e:
        print(f"Warning: Failed to save stage cache metadata for {stage_name}: {e}", file=sys.stderr)

def load_stage_metadata(cache_dir: Path, stage_name: str) -> Dict[str, Any]:
    """Loads cached metadata for a skipped stage to keep the pipeline context updated."""
    meta_file = cache_dir / f"metadata_{stage_name}.json"
    if meta_file.exists():
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AdFusion AI-powered Virtual Product Placement (VPP) modular research framework."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="adfusion/config/config.yaml",
        help="Path to the config YAML file."
    )
    parser.add_argument(
        "--stage",
        type=str,
        default=None,
        help="Execute only a single stage (e.g. 'depth', 'render', 'harmonize')."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume pipeline from the first stage without valid cached outputs."
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Custom name for the experiment folder."
    )
    args = parser.parse_args()

    workspace_dir = Path(__file__).resolve().parent
    config_path = workspace_dir / args.config
    config = load_config(config_path)

    # Resolve paths from config
    cache_dir = workspace_dir / config["pipeline"].get("cache_dir", "cache")
    experiments_root = workspace_dir / config["pipeline"].get("experiments_dir", "experiments")
    
    # Establish experiment directory and logger
    experiment_dir = get_next_experiment_dir(experiments_root, args.experiment)
    global_log_file = workspace_dir / "logs" / "pipeline.log"
    experiment_log_file = experiment_dir / "pipeline.log"

    # Setup logger (writes to console, global log, and experiment log)
    logger = setup_logger(
        name="adfusion",
        log_file=experiment_log_file
    )
    # Add handler for global log file
    try:
        global_log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(global_log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"Failed to register global log handler: {e}")

    logger.info(f"Initialized AdFusion Pipeline Context")
    logger.info(f"Workspace: {workspace_dir}")
    logger.info(f"Active Experiment Directory: {experiment_dir}")

    # Build Pipeline Context
    context = PipelineContext(
        config=config,
        workspace_dir=workspace_dir,
        cache_dir=cache_dir,
        experiment_dir=experiment_dir,
        logger=logger
    )

    # Sequential list of pipeline stages
    ordered_stages: List[BaseStage] = [
        STAGE_MAP["frame_extraction"],
        STAGE_MAP["object_detection"],
        STAGE_MAP["track_objects"],
        STAGE_MAP["estimate_depth"],
        STAGE_MAP["reconstruct_object"],
        STAGE_MAP["remove_object"],
        STAGE_MAP["render_brand"],
        STAGE_MAP["harmonize"],
        STAGE_MAP["qa"],
        STAGE_MAP["export_video"]
    ]

    # Resolve stage selection if specified
    selected_stage_name: Optional[str] = None
    if args.stage:
        resolved = resolve_stage_name(args.stage)
        if not resolved:
            logger.error(f"Unknown stage argument: '{args.stage}'. Available stages: {list(STAGE_MAP.keys())}")
            sys.exit(1)
        selected_stage_name = resolved
        logger.info(f"Executing single stage mode: '{selected_stage_name}'")

    # Initialize Performance Tracker
    tracker = PerformanceTracker()
    tracker.start_total()

    resume_triggered = False

    for stage in ordered_stages:
        # Check if this stage is enabled in config
        stage_enabled = config.get("stages", {}).get(stage.name, {}).get("enabled", True)
        if not stage_enabled:
            logger.info(f"Stage '{stage.name}' is disabled in configuration. Skipping.")
            continue

        # Single stage mode logic
        if selected_stage_name and stage.name != selected_stage_name:
            # If it's a previous stage, we need to load its metadata from cache to keep context valid
            if stage.is_cached(context):
                logger.info(f"Loading cached metadata for pre-requisite stage '{stage.name}'")
                cached_meta = load_stage_metadata(cache_dir, stage.name)
                context.stage_outputs[stage.name] = {
                    "success": True,
                    "metadata": cached_meta
                }
            else:
                logger.warning(
                    f"Pre-requisite stage '{stage.name}' is not cached, but we are running in single stage mode. "
                    "This might cause errors in downstream processing."
                )
            continue

        # Resume mode logic
        if args.resume and not resume_triggered:
            if stage.is_cached(context):
                logger.info(f"Stage '{stage.name}' is cached. Skipping execution (Resume mode).")
                cached_meta = load_stage_metadata(cache_dir, stage.name)
                context.stage_outputs[stage.name] = {
                    "success": True,
                    "metadata": cached_meta
                }
                continue
            else:
                logger.info(f"First non-cached stage encountered: '{stage.name}'. Resuming pipeline from here.")
                resume_triggered = True

        # Execute Stage
        logger.info(f"=== Executing Stage: {stage.name} ===")
        tracker.start_stage(stage.name)
        
        try:
            result: StageResult = stage.run(context)
        except Exception as e:
            logger.exception(f"Unhandled exception occurred in stage '{stage.name}': {e}")
            result = StageResult(
                stage_name=stage.name,
                success=False,
                runtime_seconds=0.0,
                error_message=f"Unhandled exception: {str(e)}"
            )

        duration = tracker.end_stage(stage.name)
        
        if not result.success:
            logger.error(f"Stage '{stage.name}' failed! Error: {result.error_message}")
            tracker.end_total()
            tracker.save_metrics(experiment_dir / "metrics.json")
            sys.exit(1)

        logger.info(f"Stage '{stage.name}' completed successfully in {duration:.4f} seconds.")
        
        # Save results in context for next stages
        context.stage_outputs[stage.name] = {
            "success": True,
            "metadata": result.metadata,
            "output_files": [str(f) for f in result.output_files]
        }

        # Save metadata to cache folder to allow resumability
        save_stage_metadata(cache_dir, stage.name, result.metadata)

    # Complete pipeline run
    tracker.end_total()
    
    # Populate SAM 3 metrics for Sprint 1
    det_outputs = context.stage_outputs.get("object_detection", {})
    track_outputs = context.stage_outputs.get("track_objects", {})

    det_runtime = det_outputs.get("metadata", {}).get("runtime_seconds", 0.0)
    if not det_runtime and "object_detection" in tracker.metrics.get("stages", {}):
        det_runtime = tracker.metrics["stages"]["object_detection"].get("runtime_seconds", 0.0)

    track_runtime = track_outputs.get("metadata", {}).get("runtime_seconds", 0.0)
    if not track_runtime and "track_objects" in tracker.metrics.get("stages", {}):
        track_runtime = tracker.metrics["stages"]["track_objects"].get("runtime_seconds", 0.0)

    frame_count = context.stage_outputs.get("frame_extraction", {}).get("metadata", {}).get("frame_count", 0)
    if not frame_count:
        cached_meta = load_stage_metadata(cache_dir, "frame_extraction")
        frame_count = cached_meta.get("frame_count", 0)

    total_sam_time = det_runtime + track_runtime
    avg_fps = frame_count / total_sam_time if total_sam_time > 0 else 0.0

    peak_vram = det_outputs.get("metadata", {}).get("peak_vram_mb", 0.0)
    checkpoint_ver = det_outputs.get("metadata", {}).get("checkpoint_version", "Meta SAM 3.0 Base")

    # Inject requested metrics keys flat at root level
    tracker.metrics["Detection Runtime"] = det_runtime
    tracker.metrics["Tracking Runtime"] = track_runtime
    tracker.metrics["Average FPS"] = avg_fps
    tracker.metrics["Peak GPU Memory"] = f"{peak_vram:.2f} MB" if peak_vram > 0 else "0.0 MB"
    tracker.metrics["Checkpoint Version"] = checkpoint_ver

    metrics_file = experiment_dir / "metrics.json"
    tracker.save_metrics(metrics_file)
    logger.info(f"=== Pipeline Completed Successfully ===")
    logger.info(f"Total Runtime: {tracker.metrics['total_runtime_seconds']:.4f} seconds")
    logger.info(f"Performance metrics saved to: {metrics_file}")

if __name__ == "__main__":
    main()
