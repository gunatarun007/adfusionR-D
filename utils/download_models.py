#!/usr/bin/env python3
"""
AdFusion Model Downloader
Downloads all required model checkpoints from publicly available URLs.
No Hugging Face token or Meta gating approval required.
"""
import os
import sys
import urllib.request
from pathlib import Path


# ── Public checkpoint URLs ──────────────────────────────────────────────────
MODELS = [
    {
        "name":     "SAM2.1 Hiera-Large",
        "filename": "sam2.1_hiera_large.pt",
        "url":      "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
        "min_size_mb": 800,   # ~860 MB
    },
    {
        "name":     "GroundingDINO SwinB",
        "filename": "groundingdino_swinb_cogcoor.pth",
        "url":      "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth",
        "min_size_mb": 300,   # ~340 MB
    },
]


def _human_size(bytes_: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if bytes_ < 1024:
            return f"{bytes_:.1f} {unit}"
        bytes_ /= 1024
    return f"{bytes_:.1f} TB"


def _progress_hook(count, block_size, total_size):
    downloaded = count * block_size
    if total_size > 0:
        pct = min(100.0, downloaded / total_size * 100)
        bar = "#" * int(pct / 2)
        print(f"\r  [{bar:<50}] {pct:5.1f}%  {_human_size(downloaded)} / {_human_size(total_size)}",
              end="", flush=True)
    else:
        print(f"\r  Downloaded {_human_size(downloaded)}", end="", flush=True)


def download_checkpoint(entry: dict, target_dir: Path) -> bool:
    """Download a single checkpoint; skip if already cached and valid."""
    target_dir.mkdir(parents=True, exist_ok=True)
    dest = target_dir / entry["filename"]
    min_bytes = entry["min_size_mb"] * 1024 * 1024

    if dest.exists() and dest.stat().st_size >= min_bytes:
        print(f"  [CACHED] {entry['filename']} ({_human_size(dest.stat().st_size)})")
        return True

    print(f"\n[*] Downloading {entry['name']} …")
    print(f"    Source : {entry['url']}")
    print(f"    Target : {dest}")
    try:
        urllib.request.urlretrieve(entry["url"], str(dest), reporthook=_progress_hook)
        print()  # newline after progress bar
        actual = dest.stat().st_size
        if actual < min_bytes:
            print(f"  [ERROR] Downloaded file too small ({_human_size(actual)}). "
                  f"Expected ≥ {entry['min_size_mb']} MB.")
            dest.unlink(missing_ok=True)
            return False
        print(f"  [OK] Saved {_human_size(actual)} → {dest}")
        return True
    except Exception as exc:
        print(f"\n  [ERROR] Download failed: {exc}")
        dest.unlink(missing_ok=True)
        return False


def main():
    print("=" * 60)
    print("        ADFUSION MODEL DOWNLOADER")
    print("  Open-source checkpoints — no token required")
    print("=" * 60)

    target_dir = Path("models")
    results = []
    for entry in MODELS:
        ok = download_checkpoint(entry, target_dir)
        results.append((entry["name"], ok))

    print("\n" + "=" * 60)
    all_ok = True
    for name, ok in results:
        status = "[OK]  " if ok else "[FAIL]"
        print(f"  {status} {name}")
        if not ok:
            all_ok = False

    print("=" * 60)
    if all_ok:
        print("[SUCCESS] All checkpoints ready.")
        sys.exit(0)
    else:
        print("[FAILURE] One or more downloads failed. Check output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
