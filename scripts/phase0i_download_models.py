"""
Phase 0I — Download ComfyUI-compatible models.

Downloads:
  SD 1.5 pruned checkpoint  → .local/ComfyUI/models/checkpoints/v1-5-pruned-emaonly.safetensors
  ControlNet Depth SD1.5    → .local/ComfyUI/models/controlnet/control_v11f1p_sd15_depth.safetensors
"""

import os, sys
from pathlib import Path

ROOT     = Path(__file__).parent.parent
COMFYUI  = ROOT / ".local" / "ComfyUI"
CK_DIR   = COMFYUI / "models" / "checkpoints"
CN_DIR   = COMFYUI / "models" / "controlnet"
HF_CACHE = Path("D:/hf_cache")

CK_DIR.mkdir(parents=True, exist_ok=True)
CN_DIR.mkdir(parents=True, exist_ok=True)

# Prefer hf_hub_download which handles resume + caching
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("ERROR: huggingface_hub not installed in this environment")
    sys.exit(1)

def download_to(repo_id, filename, dest_path, cache_dir=None):
    """Download one file from HF Hub, place it at dest_path."""
    if dest_path.exists():
        size_mb = dest_path.stat().st_size / 1024**2
        print(f"SKIP (exists {size_mb:.0f} MB): {dest_path.name}")
        return True
    print(f"Downloading {repo_id}/{filename} → {dest_path.name} ...")
    try:
        tmp = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=str(cache_dir) if cache_dir else None,
            local_files_only=False,
        )
        # hf_hub_download returns the cached path; copy/link it
        import shutil
        shutil.copy2(tmp, dest_path)
        size_mb = dest_path.stat().st_size / 1024**2
        print(f"OK  {dest_path.name}  ({size_mb:.0f} MB)")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False

# ── SD 1.5 checkpoint ────────────────────────────────────────────────────────
sd15_dest = CK_DIR / "v1-5-pruned-emaonly.safetensors"
ok_sd15 = download_to(
    repo_id="runwayml/stable-diffusion-v1-5",
    filename="v1-5-pruned-emaonly.safetensors",
    dest_path=sd15_dest,
    cache_dir=HF_CACHE,
)

# ── ControlNet Depth SD1.5 ───────────────────────────────────────────────────
cn_dest = CN_DIR / "control_v11f1p_sd15_depth.safetensors"
ok_cn = download_to(
    repo_id="lllyasviel/control_v11f1p_sd15_depth",
    filename="diffusion_pytorch_model.safetensors",
    dest_path=cn_dest,
    cache_dir=HF_CACHE,
)

print()
if ok_sd15 and ok_cn:
    print("All models ready.")
else:
    print("INCOMPLETE — check errors above.")
    sys.exit(1)
