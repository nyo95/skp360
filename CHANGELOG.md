# Changelog

All notable validation phases for `D:\Projects\skpto360` are recorded here.

## 2026-09-17 — Phase 0J: Two-Stage Controlled → FLUX Bake-off

Ran A/B/C bake-off to test the ControlNet (geometry) → FLUX (realism) two-stage architecture.

Created:

- `scripts/phase0j_two_stage_bakeoff.py`

Outputs:

- `output/phase0j/A_flux_rgb_only/output.png` + `generation.json`
- `output/phase0j/B_controlnet_depth/output.png` + `generation.json` (Stage 1 reused from Phase 0I rerun)
- `output/phase0j/C_controlnet_then_flux/output.png` + `generation.json`
- `output/phase0j/comparison/side_by_side.png`
- `output/phase0j/phase0j_report.json`

Findings:

- **Verdict: `FLUX_ENHANCES_WITHOUT_DRIFT`**
- Test B (ControlNet only): geometry correct but visually flat — looks like a 3D render. All 5 visual quality criteria POOR/FAIL.
- Test C (ControlNet → FLUX): geometry from B fully preserved (zero drift across all 7 geometry criteria). FLUX upgraded all 5 visual quality criteria from POOR to EXCELLENT in ~7s.
- Test A (FLUX direct on RGB): also strong — geometry preserved by Blender source quality. Visually on par with C.
- Two-stage architecture confirmed: ControlNet acts as geometry lock that survives FLUX's realism pass.

Constraints preserved:

- No second ControlNet, no normal/glass/material conditioning.
- No automatic retry, no six-face generation, no ERP stitching.
- FLUX applied to Stage 1 output only, not to original RGB.
- Stopped after A/B/C comparison.

## 2026-09-17 — Phase 0I: Local ComfyUI / ControlNet Feasibility Bake-off

Ran two controlled 512×512 img2img generation tests on RTX 2060 6GB using ComfyUI `--lowvram --cpu-vae` to determine ControlNet Depth viability.

Created:

- `.gitignore`
- `scripts/phase0i_prepare_inputs.py`
- `scripts/phase0i_download_models.py`
- `scripts/phase0i_run_comfyui.py`
- `scripts/run_phase0i.ps1`
- `workflows/phase0i_rgb_only.json`
- `workflows/phase0i_rgb_depth.json`

Outputs:

- `output/phase0i/input/rgb.png` — Phase 0C front face resized to 512×512
- `output/phase0i/input/depth.png` — Depth EXR (E_PERCENTILE_CLAMPED, p5–p95)
- `output/phase0i/A_rgb_only/output.png` — Test A result (13.15 s, ~7.6 it/s)
- `output/phase0i/A_rgb_only/generation.json`
- `output/phase0i/B_rgb_depth/output.png` — Test B result (9.32 s, ~5.5 it/s)
- `output/phase0i/B_rgb_depth/generation.json`
- `output/phase0i/comparison/side_by_side.png` — 4-panel: input / depth / A / B
- `output/phase0i/phase0i_report.json`

Model downloads: SD 1.5 `v1-5-pruned-emaonly.safetensors` (3.98 GB) + ControlNet `control_v11f1p_sd15_depth.safetensors` (1.27 GB) to `D:/hf_cache`.

Findings:

- **VRAM verdict: VIABLE.** RTX 2060 6GB can run SD1.5 + ControlNet Depth at 512×512 with ComfyUI `--lowvram --cpu-vae`. Phase 0H failure was a tooling issue (naive diffusers), not a hardware barrier.
- **Depth verdict: DEPTH_HURTS_QUALITY** — but confounded by a critical bug: the depth PNG was produced with near=black (Z-distance convention), while `control_v11f1p_sd15_depth` expects near=white (MiDaS inverse-depth convention). The ControlNet received inverted depth, collapsing room perspective and zooming into the back wall.
- Test A (RGB-only): photorealistic result but poor source fidelity (4/22 criteria); denoise=0.65 diverges too much from geometry.
- Test B (RGB+Depth): structurally distorted due to depth inversion; 4/22 criteria, different failure modes.
- Neither test preserves wall openings, source cabinetry, or spatial layout at denoise=0.65.

Root causes:

- **RC1 — Depth inversion:** `depth.png` is near=black; fix is `pixel = 255 - current_pixel` before ControlNet use.
- **RC2 — Denoise too high:** 0.65 allows large reconstruction divergence; lower to 0.40 for layout fidelity.

Constraints preserved:

- Phase 0H not modified.
- No cloud API used. All inference local.
- No silent downgrade. Depth convention bug documented explicitly.
- Phase 0J not started.

## 2026-09-16 — Controlled Generation Harness v0

Added the Controlled Core layer without removing the fast RGB-only baseline.

Created:

- `docs/CONTROLLED_PIPELINE_AUDIT.md`
- `scripts/controlled_material_id_pass.py`
- `scripts/controlled_core_harness.py`
- `scripts/run_controlled_material_id.ps1`
- `scripts/run_controlled_package.ps1`
- `scripts/run_controlled_bakeoff.ps1`

Outputs:

- `output/condition_work/material_id.png`
- `output/condition_work/material_visibility.json`
- `output/condition_package/manifest.json`
- `output/condition_package/source/*`
- `output/condition_package/masks/*`
- `output/condition_package/context/exterior_context.png`
- `output/condition_package/semantics/materials.json`
- `output/condition_package/semantics/lights.json`
- `output/condition_package/semantics/scene_contract.json`
- `output/condition_package/qc/source_metrics.json`
- `output/generation_bakeoff/B0_rgb_baseline/output.png`
- `output/generation_bakeoff/B1_controlled_harness/output.png`
- `output/generation_bakeoff/report.json`

Findings:

- `output/condition_package/` is now the canonical downstream input for Pipeline B.
- Material visibility is deterministic via a Blender material-ID ERP pass; 198 materials are represented and 195 are visible in the current ERP.
- Phase 0G/0H glass and exterior context masks are consumed by the scene contract and deterministic QC.
- Current Cloudflare FLUX.2 Klein harness still consumes only RGB plus prompt. Depth, normal, masks, material IDs, exterior context, geometry edges, and light positions are recorded as available but not provider-consumed.
- Light extraction remains blocked by missing source-level SketchUp instance/component transform metadata. The low-confidence `Light_metal` candidate is preserved as UNKNOWN; no fixture positions or emitters are invented.
- B0 vs B1 bake-off completed with the same provider/model/source/guidance. Deterministic QC validates ERP/projection/seam and mask-region warning signals, but does not prove semantic fidelity improvement.

Constraints preserved:

- FAST_BASELINE remains intact.
- No new AI provider added.
- No fake structural conditioning.
- No LLM/vision semantic QC added.
- Phase history and reports retained.

## 2026-09-16 — Phase 0H: Exterior Context Conditioning

Added deterministic exterior/context extraction for visible glazing regions.

Created:

- `scripts/phase0h_exterior_context.py`
- `scripts/run_phase0h.ps1`

Outputs:

- `output/phase0h/exterior_context/panorama.png`
- `output/phase0h/exterior_context/mask.png`
- `output/phase0h/exterior_context.json`
- `output/phase0h/comparison/SOURCE_G.png`
- `output/phase0h/comparison/CONTEXT_PASS.png`
- `output/phase0h/phase0h_report.json`

Findings:

- Phase 0G was treated as accepted/frozen baseline.
- The context pass makes detected glazing transparent only for diagnostic rendering.
- No exterior objects, landscaping, sky generation, or user reference images were introduced.
- Two visible glazing material regions were classified as `SCENE_GEOMETRY`.
- Eight glass candidates were marked `UNKNOWN` because they were not visible from the active ERP camera.
- Context mask is distinct from the full Phase 0G glass mask and marks only glazing pixels where scene geometry is visible behind the glass.
- Context mask coverage was about `92.5%` of visible glass-mask pixels.
- Category QC reported geometry/camera/seam/glazing as `PASS`; material fidelity as `WARNING` because the context pass intentionally changes glass visibility.
- AI context-conditioned A/B was not run because the existing Cloudflare FLUX.2 Klein interface only supports ordinary image references, not a true structural context/mask control.

Constraints preserved:

- No new AI provider added.
- No fake structural conditioning.
- No depth/normal regeneration.
- No geometry or camera changes.
- No procedural exterior generation.
- No automatic retry/orchestration.

## 2026-09-16 — Phase 0G: Source Signal Validation

Added deterministic source-signal validation before further AI work.

Created:

- `scripts/phase0g_source_signal_validation.py`
- `scripts/run_phase0g_source.ps1`
- `scripts/phase0g_flux_ab.py`
- `scripts/run_phase0g_flux_ab.ps1`

Outputs:

- `output/phase0g/glass_candidates.json`
- `output/phase0g/light_audit.json`
- `output/phase0g/masks/glass_mask.png`
- `output/phase0g/comparison/BEFORE.png`
- `output/phase0g/comparison/AFTER_SOURCE_FIX.png`
- `output/phase0g/ai_ab/G-A_BEFORE_RGB_ONLY/output.png`
- `output/phase0g/ai_ab/G-B_AFTER_SOURCE_FIX_RGB_ONLY/output.png`
- `output/phase0g/phase0g_report.json`

Findings:

- Detected 10 glass/glazing candidate materials using deterministic name and opacity evidence.
- Applied a generic architectural-glass source signal override.
- Generated an aligned ERP glass mask.
- Audited light-source evidence and found one low-confidence material candidate, `Light_metal`.
- Did not reconstruct lights because OBJ evidence was insufficient for reliable emitter location/intent.
- Ran two controlled Cloudflare FLUX.2 Klein RGB-only generations.
- AFTER source fix improved AI interpretation of glazing/daylight compared with BEFORE.
- Exterior/daylight context is still a likely bottleneck because the OBJ does not provide enough deterministic exterior context.

Constraints preserved:

- No depth/normal/glass mask sent to FLUX.
- No new AI provider added.
- No geometry or camera changes.
- No semantic material AI.
- No production lighting/material workflow.

## 2026-09-16 — Phase 0F: Cloudflare FLUX.2 Klein RGB-only ERP

Integrated the existing Cloudflare FLUX.2 Klein backend pattern into the main project for one RGB-only ERP experiment.

Created:

- `scripts/phase0f_cloudflare_flux_experiment_a.py`
- `scripts/run_phase0f.ps1`

Outputs:

- `output/phase0f/input/rgb_erp.png`
- `output/phase0f/input/rgb_erp_cloudflare_reference_511.png`
- `output/phase0f/experiment_a_rgb_only/output.png`
- `output/phase0f/experiment_a_rgb_only/generation.json`
- `output/phase0f/phase0f_report.json`

Findings:

- Cloudflare model used: `@cf/black-forest-labs/flux-2-klein-4b`.
- Request used RGB ERP only.
- Depth and normal were explicitly not sent as fake structural conditioning.
- Output was valid 2:1 ERP and local QC passed.

## 2026-09-16 — Phase 0E: AI Render Feasibility Prep

Prepared deterministic AI feasibility inputs without running unsupported AI workflows.

Created:

- `scripts/phase0e_prepare_ai_feasibility.py`
- `scripts/run_phase0e.ps1`

Outputs:

- `output/phase0e/input/rgb_erp.png`
- `output/phase0e/input/depth_condition.png`
- `output/phase0e/input/normal_condition.png`
- `output/phase0e/input/material_manifest.json`
- `output/phase0e/phase0e_report.json`

Findings:

- Material manifest generated with 198 deterministic material IDs.
- Depth condition generated using fixed interior-oriented mapping: near `0.2 m`, far `20.0 m`.
- Normal condition generated from Phase 0D normal EXR.
- Experiment A was prepared, but initially marked `NOT_RUN` until Cloudflare backend integration in Phase 0F.
- Experiments B/C were marked `UNSUPPORTED` because true depth/normal structural conditioning was not available.

## 2026-09-16 — Phase 0D: Native Equirectangular RGB/Depth/Normal

Moved final geometry render path away from stitched cubemap and into native Blender panoramic equirectangular rendering.

Created:

- `scripts/phase0d_equirectangular_passes.py`
- `scripts/run_phase0d.ps1`

Outputs:

- `output/phase0d/rgb/equirectangular.png`
- `output/phase0d/depth/equirectangular.exr`
- `output/phase0d/depth/equirectangular_preview.png`
- `output/phase0d/normal/equirectangular.exr`
- `output/phase0d/normal/equirectangular_preview.png`
- `output/phase0d/phase0d_report.json`

Findings:

- Native equirectangular render passed validation.
- Resolution: `2048 × 1024`.
- Projection: Blender panoramic equirectangular.
- Renderer: Cycles.
- Camera source: `AI360_1`.
- Cubemap retained only as debug/compatibility, not final path.

## 2026-09-16 — Phase 0C: Cubemap RGB/Depth/Normal Conditioning

Validated six-face cubemap rendering from the same authoritative camera position.

Created:

- `scripts/phase0c_cubemap_passes.py`
- `scripts/run_phase0c.ps1`

Outputs:

- `output/phase0c/rgb/{front,right,back,left,top,bottom}.png`
- `output/phase0c/depth/*.exr`
- `output/phase0c/depth/*_preview.png`
- `output/phase0c/normal/*.exr`
- `output/phase0c/normal/*_preview.png`
- `output/phase0c/phase0c_report.json`

Findings:

- Cubemap faces rendered successfully.
- Camera basis was valid and orthogonal.
- White/bright stitch seam artifacts were observed later, so cubemap stitching is not used as the final output path.

## 2026-09-16 — Phase 0B.2: OBJ Transport Validation

Validated OBJ as the preferred minimal geometry transport after FBX failure.

Created/updated:

- `scripts/phase0b_obj_import_verify.py`
- `scripts/run_phase0b_obj.ps1`

Outputs:

- `output/phase0b_obj/phase0b_obj_imported.blend`
- `output/phase0b_obj/preview_camera.png`
- `output/phase0b_obj/preview_overview.png`
- `output/phase0b_obj/phase0b_obj_report.json`

Findings:

- Blender 5.1 native OBJ import passed.
- Geometry was imported without custom parsing.
- Camera reconstructed from `scene.json`.
- OBJ accepted as Phase 0 transport.

## 2026-09-16 — Phase 0B FBX Experiment Closed

FBX was tested and rejected for this environment.

Findings:

- SketchUp 2021 exported ASCII FBX.
- Blender 5.1 rejected ASCII FBX.
- No FBX parser or conversion workaround was added.

## 2026-09-16 — Phase 0B DAE Proof

Initial DAE handoff proved geometry + camera metadata concepts, but was not productionized.

Findings:

- Blender 5.1 in this environment had no native COLLADA importer.
- A custom fallback parser was used only as a provisional proof.
- Custom DAE parser must not be expanded.

## 2026-09-16 — Phase 0A Exporter

SketchUp exporter evolved toward the current OBJ + metadata transport.

Current primary output:

- `scene.obj`
- `scene.mtl`
- `textures/*`
- `scene.json`

Notes:

- SketchUp units are millimeters.
- `scene.json` remains the authoritative camera source.
- OBJ export preserves texture/material guidance for downstream Blender and AI phases.
