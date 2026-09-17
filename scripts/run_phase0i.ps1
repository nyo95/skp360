# Phase 0I — Run ComfyUI local generation tests
# Tests: A (RGB-only) and B (RGB + ControlNet Depth)

$VENV_PY   = "D:\Projects\skpto360\.local\comfyui_env\Scripts\python.exe"
$ROOT      = "D:\Projects\skpto360"
$COMFYUI   = "$ROOT\.local\ComfyUI"
$SCRIPT    = "$ROOT\scripts\phase0i_run_comfyui.py"
$OUT_DIR   = "$ROOT\output\phase0i"

Write-Host "=== Phase 0I: ComfyUI Local Feasibility Bake-off ===" -ForegroundColor Cyan
Write-Host "Model A: SD 1.5 img2img RGB-only"
Write-Host "Model B: SD 1.5 img2img + ControlNet Depth"
Write-Host "VRAM limit: --lowvram"
Write-Host ""

# Run the main generation script
$env:PYTHONPATH = $COMFYUI
& $VENV_PY $SCRIPT 2>&1
