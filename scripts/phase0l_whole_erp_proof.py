#!/usr/bin/env python3
"""
Phase 0L — Whole ERP RGB + Depth Proof

1. Stitch 6x RGB cubemap faces into one RGB ERP.
2. Run Blender to generate Depth ERP from OBJ.
3. Run ComfyUI (SD1.5 + ControlNet Depth) on the whole ERPs.
4. Validate outputs and write report.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output" / "phase0l"
COMFYUI = ROOT / ".local" / "ComfyUI"
VENV_PY = ROOT / ".local" / "comfyui_env" / "Scripts" / "python.exe"

FACES = ("front", "right", "back", "left", "top", "bottom")
ERP_WIDTH = 2048
ERP_HEIGHT = 1024

def stitch(cube: dict[str, np.ndarray], width: int) -> np.ndarray:
    height = width // 2
    size = next(iter(cube.values())).shape[0]

    u_px, v_px = np.meshgrid(np.arange(width, dtype=np.float32),
                             np.arange(height, dtype=np.float32))
    
    # U=0.5 -> lon=0 (Front)
    # U=0.75 -> lon=pi/2 (Right)
    # U=0.25 -> lon=-pi/2 (Left)
    lon = (u_px / width - 0.5) * 2 * np.pi
    lat = (0.5 - v_px / height) * np.pi

    # SketchUp world space: Front = +X, Right = -Y, Up = +Z
    X = np.cos(lat) * np.cos(lon)
    Y = -np.cos(lat) * np.sin(lon)
    Z = np.sin(lat)

    aX, aY, aZ = np.abs(X), np.abs(Y), np.abs(Z)
    output = np.zeros((height, width, 3), dtype=np.uint8)

    def remap(mask, u_raw, v_raw, face):
        u = np.clip((u_raw + 1) * 0.5 * (size - 1), 0, size - 1).astype(np.float32)
        v = np.clip((v_raw + 1) * 0.5 * (size - 1), 0, size - 1).astype(np.float32)
        s = cv2.remap(cube[face], u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        output[mask] = s[mask]

    with np.errstate(divide="ignore", invalid="ignore"):
        # Front: +X
        m = (X > 0) & (aX >= aY) & (aX >= aZ)
        remap(m, -Y / np.where(m, X, 1), -Z / np.where(m, X, 1), "front")
        
        # Right: -Y
        m = (Y < 0) & (aY >= aX) & (aY >= aZ)
        remap(m, X / np.where(m, Y, 1), Z / np.where(m, Y, 1), "right")
        
        # Back: -X
        m = (X < 0) & (aX >= aY) & (aX >= aZ)
        remap(m, -Y / np.where(m, X, 1), -Z / np.where(m, X, 1), "back")
        
        # Left: +Y
        m = (Y > 0) & (aY >= aX) & (aY >= aZ)
        remap(m, X / np.where(m, Y, 1), -Z / np.where(m, Y, 1), "left")
        
        # Top: +Z
        m = (Z > 0) & (aZ >= aX) & (aZ >= aY)
        remap(m, -Y / np.where(m, Z, 1), X / np.where(m, Z, 1), "top")
        
        # Bottom: -Z
        m = (Z < 0) & (aZ >= aX) & (aZ >= aY)
        remap(m, Y / np.where(m, Z, 1), X / np.where(m, Z, 1), "bottom")

    return output


def step_stitch_rgb():
    print("[Stage 1] Stitching SketchUp RGB Cubemap...")
    src_dir = ROOT / "output" / "phase0c" / "rgb"
    cube = {}
    for face in FACES:
        img_path = src_dir / f"{face}.png"
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Missing face: {img_path}")
        s = min(img.shape[:2])
        cube[face] = img[:s, :s]
    
    panorama = stitch(cube, ERP_WIDTH)
    out_path = OUT_DIR / "rgb_erp.png"
    cv2.imwrite(str(out_path), panorama)
    print(f"Stitched RGB ERP saved to {out_path}")
    return out_path


def step_generate_blender_depth():
    print("[Stage 2] Generating Blender Depth ERP...")
    obj_path = ROOT / "output" / "obj" / "scene.obj"
    json_path = ROOT / "output" / "obj" / "scene.json"
    blender_out = OUT_DIR / "blender_tmp"
    
    blender_exe = "C:\\Program Files\\Blender Foundation\\Blender 4.2\\blender.exe"
    if not os.path.exists(blender_exe):
        blender_exe = "C:\\Program Files\\Blender Foundation\\Blender 5.1\\blender.exe"
    
    script_path = ROOT / "scripts" / "phase0d_equirectangular_passes.py"
    
    cmd = [
        blender_exe, "--background", "--factory-startup",
        "--python", str(script_path), "--",
        "--obj", str(obj_path),
        "--json", str(json_path),
        "--out", str(blender_out),
        "--width", str(ERP_WIDTH),
        "--height", str(ERP_HEIGHT),
        "--engine", "BLENDER_EEVEE"
    ]
    
    subprocess.run(cmd, check=True)
    
    depth_preview = blender_out / "depth" / "equirectangular_preview.png"
    depth_png = OUT_DIR / "depth_erp.png"
    
    if not depth_preview.exists():
        raise FileNotFoundError(f"Blender failed to generate depth preview: {depth_preview}")
        
    shutil.copy2(depth_preview, depth_png)
    print(f"Depth ERP saved to {depth_png}")
    return depth_png


def wait_for_server(timeout=90):
    url = "http://127.0.0.1:8188/system_stats"
    for _ in range(timeout):
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:
            time.sleep(1)
    return False

def step_run_comfyui(rgb_path, depth_path):
    print("[Stage 3] Running Whole-ERP AI Inference (ComfyUI)...")
    
    comfy_in = COMFYUI / "input"
    comfy_out = COMFYUI / "output"
    comfy_in.mkdir(parents=True, exist_ok=True)
    
    rgb_name = "phase0l_rgb.png"
    depth_name = "phase0l_depth.png"
    shutil.copy2(rgb_path, comfy_in / rgb_name)
    shutil.copy2(depth_path, comfy_in / depth_name)
    
    print("Starting ComfyUI server...")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(COMFYUI)
    proc = subprocess.Popen(
        [str(VENV_PY), str(COMFYUI / "main.py"), "--lowvram", "--port", "8188"],
        cwd=str(COMFYUI), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    
    if not wait_for_server():
        proc.terminate()
        raise RuntimeError("ComfyUI server failed to start.")
        
    workflow = {
        "1":  {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "v1-5-pruned-emaonly.safetensors"}},
        "2":  {"class_type": "CLIPTextEncode", "inputs": {"text": "photorealistic architectural interior visualization, 8k resolution, sharp, highly detailed, realistic materials", "clip": ["1", 1]}},
        "3":  {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry, low quality, distortion, bad geometry", "clip": ["1", 1]}},
        "4":  {"class_type": "LoadImage",      "inputs": {"image": rgb_name}},
        "5":  {"class_type": "LoadImage",      "inputs": {"image": depth_name}},
        "6":  {"class_type": "ControlNetLoader", "inputs": {"control_net_name": "control_v11f1p_sd15_depth.safetensors"}},
        "7":  {
            "class_type": "ControlNetApplyAdvanced",
            "inputs": {
                "positive": ["2", 0], "negative": ["3", 0],
                "control_net": ["6", 0], "image": ["5", 0],
                "strength": 0.8, "start_percent": 0.0, "end_percent": 1.0
            }
        },
        "8":  {"class_type": "VAEEncode", "inputs": {"pixels": ["4", 0], "vae": ["1", 2]}},
        "9":  {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0], "positive": ["7", 0], "negative": ["7", 1],
                "latent_image": ["8", 0],
                "seed": 42, "steps": 20, "cfg": 7.0,
                "sampler_name": "euler_ancestral", "scheduler": "karras", "denoise": 0.45
            }
        },
        "10": {"class_type": "VAEDecode",  "inputs": {"samples": ["9", 0], "vae": ["1", 2]}},
        "11": {"class_type": "SaveImage",  "inputs": {"images": ["10", 0], "filename_prefix": "phase0l_final"}}
    }
    
    try:
        data = json.dumps({"prompt": workflow}).encode()
        req = urllib.request.Request("http://127.0.0.1:8188/prompt", data=data, headers={"Content-Type": "application/json"})
        resp = urllib.request.urlopen(req, timeout=10)
        prompt_id = json.loads(resp.read())["prompt_id"]
        
        final_img = None
        start_time = time.time()
        while time.time() - start_time < 600:
            resp = urllib.request.urlopen(f"http://127.0.0.1:8188/history/{prompt_id}", timeout=10)
            hist = json.loads(resp.read())
            if prompt_id in hist:
                for node_id, node_out in hist[prompt_id].get("outputs", {}).items():
                    if "images" in node_out:
                        img_info = node_out["images"][0]
                        src = comfy_out / img_info["subfolder"] / img_info["filename"] if img_info["subfolder"] else comfy_out / img_info["filename"]
                        dst = OUT_DIR / "final_erp.png"
                        shutil.copy2(src, dst)
                        final_img = dst
                break
            time.sleep(2)
            
    finally:
        proc.terminate()
        proc.wait(timeout=10)
        
    if not final_img:
        raise RuntimeError("AI inference timed out or failed.")
        
    print(f"Final ERP saved to {final_img}")
    return final_img


def validate_outputs(rgb_path, depth_path, final_path):
    print("[Stage 4] QC Validation...")
    report = {
        "status": "PASS",
        "files": {
            "rgb_erp": str(rgb_path),
            "depth_erp": str(depth_path),
            "final_erp": str(final_path)
        },
        "qc_checks": []
    }
    
    img = cv2.imread(str(final_path))
    h, w = img.shape[:2]
    
    # Check 2:1 ratio
    is_2_to_1 = bool(w == h * 2)
    report["qc_checks"].append({"name": "dimensions_2_to_1", "pass": is_2_to_1, "details": f"{w}x{h}"})
    
    # Check seam
    left = img[:, 0]
    right = img[:, -1]
    seam_diff = float(np.mean(np.abs(left.astype(float) - right.astype(float))))
    report["qc_checks"].append({"name": "x_wrap_seam_check", "pass": bool(seam_diff < 30.0), "details": f"Mean difference: {seam_diff:.2f}"})
    
    if not all(chk["pass"] for chk in report["qc_checks"]):
        report["status"] = "FAIL"
        
    report_file = OUT_DIR / "phase0l_report.json"
    report_file.write_text(json.dumps(report, indent=2))
    print(f"Report written to {report_file}")
    
    
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rgb_erp = step_stitch_rgb()
    depth_erp = step_generate_blender_depth()
    final_erp = step_run_comfyui(rgb_erp, depth_erp)
    validate_outputs(rgb_erp, depth_erp, final_erp)
    
if __name__ == "__main__":
    main()
