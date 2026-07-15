#!/usr/bin/env python3
"""
AdFusion System Diagnostics & Verification
Checks all environment requirements for the GroundingDINO + SAM2 pipeline.
No Hugging Face token or Triton required.
"""
import os
import sys
import shutil
import subprocess


def run_command(cmd):
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return result.stdout.strip()
    except Exception:
        return None


def check_disk_space():
    root = "/" if sys.platform != "win32" else "C:\\"
    total, used, free = shutil.disk_usage(root)
    return total / (1024 ** 3), used / (1024 ** 3), free / (1024 ** 3)


def main():
    print("=" * 60)
    print("     ADFUSION SYSTEM DIAGNOSTICS & VERIFICATION")
    print("     Backend: GroundingDINO + SAM2 (open-source)")
    print("=" * 60)

    all_passed = True
    warnings = []

    # 1. Python
    python_version = sys.version.split()[0]
    print(f"[*] Python Version: {python_version} ... OK")

    # 2. PyTorch & CUDA
    try:
        import torch
        print(f"[*] PyTorch Version: {torch.__version__} ... OK")
        cuda_available = torch.cuda.is_available()
        print(f"[*] PyTorch CUDA Available: {cuda_available}")

        if not cuda_available:
            print("[WARNING] PyTorch running on CPU only. CUDA is recommended for RunPod.")
            warnings.append("CUDA not available — inference will be slow on CPU.")
        else:
            gpu_count = torch.cuda.device_count()
            gpu_name  = torch.cuda.get_device_name(0)
            props     = torch.cuda.get_device_properties(0)
            vram_gb   = props.total_memory / (1024 ** 3)
            print(f"[*] GPU: {gpu_name} ({gpu_count} GPU(s))")
            print(f"[*] VRAM: {vram_gb:.2f} GB")
            if vram_gb < 6.0:
                warnings.append(f"Low VRAM ({vram_gb:.1f} GB). 8 GB+ recommended for SAM2 + GroundingDINO.")
            cuda_ver = torch.version.cuda
            print(f"[*] CUDA (compiled): {cuda_ver}")
            driver = run_command(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
            if driver:
                print(f"[*] NVIDIA Driver: {driver}")
    except ImportError:
        print("[ERROR] PyTorch is NOT installed.")
        all_passed = False

    # 3. GroundingDINO package check
    try:
        import groundingdino  # noqa: F401
        print("[*] GroundingDINO package: importable ... OK")
    except ImportError:
        print("[ERROR] groundingdino package NOT importable. "
              "Run: pip install -e third_party/GroundingDINO")
        all_passed = False

    # 4. SAM2 package check
    try:
        import sam2  # noqa: F401
        print("[*] SAM2 package: importable ... OK")
    except ImportError:
        print("[ERROR] sam2 package NOT importable. "
              "Run: pip install -e third_party/sam2")
        all_passed = False

    # 5. Checkpoint files
    models_dir = "models"
    for fname, min_mb in [
        ("sam2.1_hiera_large.pt",          800),
        ("groundingdino_swinb_cogcoor.pth", 300),
    ]:
        path = os.path.join(models_dir, fname)
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            if size_mb >= min_mb:
                print(f"[*] Checkpoint {fname}: {size_mb:.0f} MB ... OK")
            else:
                print(f"[WARNING] Checkpoint {fname}: {size_mb:.0f} MB (expected ≥ {min_mb} MB)")
                warnings.append(f"{fname} may be corrupt or incomplete.")
        else:
            print(f"[WARNING] Checkpoint {fname}: NOT FOUND — run utils/download_models.py")
            warnings.append(f"Missing checkpoint: {fname}")

    # 6. OpenCV
    try:
        import cv2
        print(f"[*] OpenCV: {cv2.__version__} ... OK")
    except ImportError:
        print("[ERROR] OpenCV NOT installed.")
        all_passed = False

    # 7. FFmpeg
    ffmpeg_out = run_command(["ffmpeg", "-version"])
    if ffmpeg_out:
        print(f"[*] FFmpeg: {ffmpeg_out.splitlines()[0]} ... OK")
    else:
        print("[ERROR] FFmpeg binary not found in PATH.")
        all_passed = False

    # 8. Disk Space
    total_gb, _, free_gb = check_disk_space()
    print(f"[*] Disk: {free_gb:.1f} GB free / {total_gb:.1f} GB total")
    if free_gb < 15.0:
        warnings.append(f"Low disk space ({free_gb:.1f} GB). 15 GB+ recommended.")

    # Summary
    print("=" * 60)
    if warnings:
        print("Warnings:")
        for w in warnings:
            print(f"  [!] {w}")
    if all_passed:
        print("[SUCCESS] Environment verification PASSED.")
        sys.exit(0)
    else:
        print("[FAILURE] Environment verification FAILED. Fix errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
