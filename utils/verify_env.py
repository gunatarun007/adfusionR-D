#!/usr/bin/env python3
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
    total, used, free = shutil.disk_usage("/")
    total_gb = total / (1024 ** 3)
    used_gb = used / (1024 ** 3)
    free_gb = free / (1024 ** 3)
    return total_gb, used_gb, free_gb

def main():
    print("=" * 60)
    print("           ADFUSION SYSTEM DIAGNOSTICS & VERIFICATION")
    print("=" * 60)

    all_passed = True
    warnings = []

    # 1. Python Check
    python_version = sys.version.split()[0]
    print(f"[*] Python Version: {python_version} ... OK")

    # 2. PyTorch & CUDA Check
    try:
        import torch
        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        print(f"[*] PyTorch Version: {torch_version} ... OK")
        print(f"[*] PyTorch CUDA Available: {cuda_available}")

        if not cuda_available:
            print("[ERROR] PyTorch is running on CPU only. CUDA is REQUIRED for RunPod pipeline execution.")
            all_passed = False
        else:
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            device_properties = torch.cuda.get_device_properties(0)
            total_vram = device_properties.total_memory / (1024 ** 3) # GB
            
            print(f"[*] GPU Device Name: {gpu_name} ({gpu_count} GPU(s) detected)")
            print(f"[*] Total VRAM Capacity: {total_vram:.2f} GB")
            
            # Warn if VRAM is below recommended threshold
            if total_vram < 6.0:
                warnings.append(f"Low VRAM ({total_vram:.2f} GB). 8GB+ is recommended for SAM 3 and video processing.")
            
            cuda_ver = torch.version.cuda
            print(f"[*] PyTorch Compiled CUDA: {cuda_ver}")

            # Try to get driver version via nvidia-smi
            nvidia_smi_out = run_command(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
            if nvidia_smi_out:
                print(f"[*] NVIDIA GPU Driver Version: {nvidia_smi_out}")
            else:
                print("[!] NVIDIA Driver Version: (nvidia-smi not available/failed)")
    except ImportError:
        print("[ERROR] PyTorch is NOT installed in this environment.")
        all_passed = False

    # 3. Triton Check (Optional for CPU, Required for GPU R&D)
    try:
        import triton
        print(f"[*] Triton Compiler: Available ({triton.__version__}) ... OK")
    except ImportError:
        # Triton is Linux-only, so warn if on Windows, but fail on Linux
        if sys.platform == "win32":
            print("[*] Triton Compiler: Not Available (Expected on Windows)")
            warnings.append("Triton is missing (Expected on Windows). Triton is required on RunPod Linux.")
        else:
            print("[ERROR] Triton is NOT installed. Triton is REQUIRED for SAM 3 tracking kernels on Linux.")
            all_passed = False

    # 4. OpenCV Check
    try:
        import cv2
        print(f"[*] OpenCV Version: {cv2.__version__} ... OK")
    except ImportError:
        print("[ERROR] OpenCV is NOT installed.")
        all_passed = False

    # 5. FFmpeg Check
    ffmpeg_version_out = run_command(["ffmpeg", "-version"])
    if ffmpeg_version_out:
        ffmpeg_line = ffmpeg_version_out.split("\n")[0]
        print(f"[*] FFmpeg Binary: Available ({ffmpeg_line}) ... OK")
    else:
        print("[ERROR] FFmpeg binary not found in system PATH. Video pipeline requires FFmpeg installed.")
        all_passed = False

    # 6. Disk Space Check
    total_gb, used_gb, free_gb = check_disk_space()
    print(f"[*] Disk Space: Total={total_gb:.1f}GB, Free={free_gb:.1f}GB")
    if free_gb < 15.0:
        warnings.append(f"Low disk space ({free_gb:.1f} GB free). Downloading checkpoints and frames requires ~15GB free.")

    # Summary
    print("=" * 60)
    if all_passed:
        print("[SUCCESS] Environment verification PASSED.")
        if warnings:
            print("\nWarnings:")
            for w in warnings:
                print(f"  - [WARNING] {w}")
        sys.exit(0)
    else:
        print("[FAILURE] Environment verification FAILED. Please correct errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
