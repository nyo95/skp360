"""
Phase 0I — Run ComfyUI tests via API.

Starts ComfyUI server, copies inputs, runs Test A (RGB-only) and Test B (RGB+ControlNet Depth),
saves outputs and generation.json for each.
"""

import os, sys, json, time, shutil, subprocess, signal
import urllib.request, urllib.parse, urllib.error
from pathlib import Path

ROOT      = Path(__file__).parent.parent
COMFYUI   = ROOT / ".local" / "ComfyUI"
VENV_PY   = ROOT / ".local" / "comfyui_env" / "Scripts" / "python.exe"
COMFY_IN  = COMFYUI / "input"
COMFY_OUT = COMFYUI / "output"
PHASE_OUT = ROOT / "output" / "phase0i"
WORKFLOWS = ROOT / "workflows"

COMFYUI_URL = "http://127.0.0.1:8188"
SEED    = 42
STEPS   = 20
CFG     = 7.0
DENOISE = 0.65
SAMPLER = "euler_ancestral"
SCHED   = "karras"
CKPT    = "v1-5-pruned-emaonly.safetensors"
CNET    = "control_v11f1p_sd15_depth.safetensors"
CNET_STR  = 0.8
CNET_START = 0.0
CNET_END   = 1.0

PROMPT_POS = (
    "photorealistic architectural interior visualization, modern living room, "
    "preserve exact wall geometry, wall openings, ceiling structure, cabinetry, "
    "furniture placement, improve material realism, realistic lighting, "
    "sharp details, high quality render"
)
PROMPT_NEG = (
    "cartoon, illustration, painting, sketch, low quality, blurry, "
    "distorted geometry, missing walls, extra furniture, redesigned space, "
    "fish eye, panorama distortion"
)


def wait_for_server(timeout=60):
    for _ in range(timeout):
        try:
            urllib.request.urlopen(f"{COMFYUI_URL}/system_stats", timeout=2)
            return True
        except Exception:
            time.sleep(1)
    return False


def post_prompt(workflow: dict) -> str:
    data = json.dumps({"prompt": workflow}).encode()
    req  = urllib.request.Request(f"{COMFYUI_URL}/prompt", data=data,
                                   headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=10)
    return json.loads(resp.read())["prompt_id"]


def wait_for_job(prompt_id: str, timeout=600):
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = urllib.request.urlopen(f"{COMFYUI_URL}/history/{prompt_id}", timeout=10)
            hist = json.loads(resp.read())
            if prompt_id in hist:
                return hist[prompt_id]
        except Exception:
            pass
        time.sleep(2)
    return None


def collect_output(hist_entry: dict, test_label: str, prefix: str):
    """Find the generated image, copy to phase0i output dir."""
    out_dir = PHASE_OUT / test_label
    out_dir.mkdir(parents=True, exist_ok=True)
    images = []
    for node_id, node_out in hist_entry.get("outputs", {}).items():
        if "images" in node_out:
            for img_info in node_out["images"]:
                src = COMFY_OUT / img_info["subfolder"] / img_info["filename"] if img_info["subfolder"] \
                      else COMFY_OUT / img_info["filename"]
                dst = out_dir / "output.png"
                shutil.copy2(src, dst)
                images.append(str(dst))
    return images


def build_test_a():
    """img2img, no ControlNet."""
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": PROMPT_POS, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": PROMPT_NEG, "clip": ["1", 1]}},
        "4": {"class_type": "LoadImage",      "inputs": {"image": "phase0i_rgb.png"}},
        "5": {"class_type": "VAEEncode",      "inputs": {"pixels": ["4", 0], "vae": ["1", 2]}},
        "6": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
                "latent_image": ["5", 0],
                "seed": SEED, "steps": STEPS, "cfg": CFG,
                "sampler_name": SAMPLER, "scheduler": SCHED, "denoise": DENOISE
            }
        },
        "7": {"class_type": "VAEDecode",  "inputs": {"samples": ["6", 0], "vae": ["1", 2]}},
        "8": {"class_type": "SaveImage",  "inputs": {"images": ["7", 0], "filename_prefix": "phase0i_A"}},
    }


def build_test_b():
    """img2img + ControlNet Depth."""
    return {
        "1":  {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "2":  {"class_type": "CLIPTextEncode", "inputs": {"text": PROMPT_POS, "clip": ["1", 1]}},
        "3":  {"class_type": "CLIPTextEncode", "inputs": {"text": PROMPT_NEG, "clip": ["1", 1]}},
        "4":  {"class_type": "LoadImage",      "inputs": {"image": "phase0i_rgb.png"}},
        "5":  {"class_type": "LoadImage",      "inputs": {"image": "phase0i_depth.png"}},
        "6":  {"class_type": "ControlNetLoader", "inputs": {"control_net_name": CNET}},
        "7":  {
            "class_type": "ControlNetApplyAdvanced",
            "inputs": {
                "positive": ["2", 0], "negative": ["3", 0],
                "control_net": ["6", 0], "image": ["5", 0],
                "strength": CNET_STR, "start_percent": CNET_START, "end_percent": CNET_END
            }
        },
        "8":  {"class_type": "VAEEncode", "inputs": {"pixels": ["4", 0], "vae": ["1", 2]}},
        "9":  {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0], "positive": ["7", 0], "negative": ["7", 1],
                "latent_image": ["8", 0],
                "seed": SEED, "steps": STEPS, "cfg": CFG,
                "sampler_name": SAMPLER, "scheduler": SCHED, "denoise": DENOISE
            }
        },
        "10": {"class_type": "VAEDecode",  "inputs": {"samples": ["9", 0], "vae": ["1", 2]}},
        "11": {"class_type": "SaveImage",  "inputs": {"images": ["10", 0], "filename_prefix": "phase0i_B"}},
    }


def gen_meta(test_label, extra=None):
    d = {
        "test": test_label,
        "model": CKPT,
        "prompt_positive": PROMPT_POS,
        "prompt_negative": PROMPT_NEG,
        "seed": SEED,
        "steps": STEPS,
        "cfg": CFG,
        "sampler": SAMPLER,
        "scheduler": SCHED,
        "denoise": DENOISE,
        "resolution": "512x512",
        "rgb_source": "output/phase0c/rgb/front.png (1024x1024 → resized 512x512)",
    }
    if extra:
        d.update(extra)
    return d


def main():
    import time as _t

    # ── copy inputs into ComfyUI input dir ───────────────────────────────────
    COMFY_IN.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PHASE_OUT / "input" / "rgb.png",   COMFY_IN / "phase0i_rgb.png")
    shutil.copy2(PHASE_OUT / "input" / "depth.png", COMFY_IN / "phase0i_depth.png")
    print("Inputs copied to ComfyUI input dir.")

    # ── start ComfyUI server ─────────────────────────────────────────────────
    print(f"Starting ComfyUI server ({COMFYUI}) ...")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(COMFYUI)
    proc = subprocess.Popen(
        [str(VENV_PY), str(COMFYUI / "main.py"), "--lowvram", "--port", "8188",
         "--output-directory", str(COMFY_OUT)],
        cwd=str(COMFYUI),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print("Waiting for server ...")
    if not wait_for_server(timeout=90):
        # print last server output
        proc.kill()
        print("ERROR: ComfyUI server did not start. Check VRAM / dependencies.")
        sys.exit(1)
    print("Server ready.")

    results = {}

    try:
        # ── Test A: RGB only ─────────────────────────────────────────────────
        print("\n=== TEST A: RGB ONLY ===")
        t0 = _t.time()
        pid_a = post_prompt(build_test_a())
        print(f"Job submitted: {pid_a}")
        hist_a = wait_for_job(pid_a, timeout=600)
        t_a = _t.time() - t0
        if not hist_a:
            print("ERROR: Test A timed out.")
            results["A_rgb_only"] = {"status": "TIMEOUT", "runtime_s": t_a}
        else:
            imgs_a = collect_output(hist_a, "A_rgb_only", "phase0i_A")
            meta_a = gen_meta("A_RGB_ONLY")
            meta_a["runtime_s"] = round(t_a, 1)
            meta_a["outputs"] = imgs_a
            json.dump(meta_a, open(PHASE_OUT / "A_rgb_only" / "generation.json", "w"), indent=2)
            results["A_rgb_only"] = {"status": "OK", "runtime_s": t_a}
            print(f"Test A done in {t_a:.1f}s  → {imgs_a}")

        # ── Test B: RGB + ControlNet Depth ────────────────────────────────────
        print("\n=== TEST B: RGB + CONTROLNET DEPTH ===")
        t0 = _t.time()
        pid_b = post_prompt(build_test_b())
        print(f"Job submitted: {pid_b}")
        hist_b = wait_for_job(pid_b, timeout=600)
        t_b = _t.time() - t0
        if not hist_b:
            print("ERROR: Test B timed out.")
            results["B_rgb_depth"] = {"status": "TIMEOUT", "runtime_s": t_b}
        else:
            imgs_b = collect_output(hist_b, "B_rgb_depth", "phase0i_B")
            meta_b = gen_meta("B_RGB_DEPTH", {
                "controlnet_model": CNET,
                "controlnet_strength": CNET_STR,
                "controlnet_start": CNET_START,
                "controlnet_end": CNET_END,
                "depth_source": "output/phase0c/depth/front.exr (E_PERCENTILE_CLAMPED → 512x512 PNG)",
                "depth_mapping": "E_PERCENTILE_CLAMPED (p5-p95, near=white, matches Phase 0H D3 decision)",
            })
            meta_b["runtime_s"] = round(t_b, 1)
            meta_b["outputs"] = imgs_b
            json.dump(meta_b, open(PHASE_OUT / "B_rgb_depth" / "generation.json", "w"), indent=2)
            results["B_rgb_depth"] = {"status": "OK", "runtime_s": t_b}
            print(f"Test B done in {t_b:.1f}s  → {imgs_b}")

    finally:
        proc.terminate()
        proc.wait(timeout=10)
        print("\nComfyUI server stopped.")

    print("\nResults:", results)
    return results


if __name__ == "__main__":
    main()
