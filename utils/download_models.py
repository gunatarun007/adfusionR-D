#!/usr/bin/env python3
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("[ERROR] huggingface_hub is not installed. Please install it using 'pip install huggingface_hub'.")
    sys.exit(1)

def check_hf_auth():
    """Verify Hugging Face authentication token is set up."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        # Check standard cache file for Hugging Face CLI login
        hf_token_path = Path.home() / ".cache" / "huggingface" / "token"
        if hf_token_path.exists():
            token = hf_token_path.read_text().strip()
            os.environ["HF_TOKEN"] = token
            
    if not token:
        print("[WARNING] No HF_TOKEN environment variable or local login credentials found.")
        print("Official Segment Anything Model 3 checkpoints are gated on Hugging Face.")
        print("Please accept terms at https://huggingface.co/facebook/sam3 and set your HF_TOKEN:")
        print("  export HF_TOKEN=\"your_huggingface_write_token\"")
        print("-" * 60)

def download_official_checkpoint(repo_id, filename, target_dir):
    """Download checkpoint from Hugging Face into target directory."""
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    dest_path = target_dir / filename

    if dest_path.exists() and dest_path.stat().st_size > 1000 * 1024 * 1024: # Must be > 1GB
        print(f"[*] Checkpoint '{filename}' already cached at {dest_path}. Skipping.")
        return True

    print(f"[*] Initiating official download: {repo_id}/{filename} ...")
    try:
        # Download and write directly to target folder
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
            token=os.environ.get("HF_TOKEN")
        )
        print(f"[SUCCESS] Saved checkpoint to {downloaded_path}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to download {filename} from {repo_id}.")
        print(f"Technical Details: {e}")
        print("\nFix checklist:")
        print(f"1. Open https://huggingface.co/{repo_id} and click 'Access Repository'")
        print("2. Retrieve your Hugging Face read access token from settings")
        print("3. Export it in your environment: export HF_TOKEN=\"hf_xxxxxxx\"")
        print("4. Re-run this installer")
        return False

def main():
    print("=" * 60)
    print("                ADFUSION MODEL INSTALLER")
    print("=" * 60)
    
    check_hf_auth()
    
    target_dir = "models"
    success_3_0 = download_official_checkpoint(
        repo_id="facebook/sam3",
        filename="sam3.pt",
        target_dir=target_dir
    )
    
    success_3_1 = download_official_checkpoint(
        repo_id="facebook/sam3.1",
        filename="sam3.1_multiplex.pt",
        target_dir=target_dir
    )
    
    print("=" * 60)
    if success_3_0 and success_3_1:
        print("[SUCCESS] All official checkpoints downloaded and verified.")
        sys.exit(0)
    else:
        print("[FAILURE] One or more checkpoints failed to download. Check errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
