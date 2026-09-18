# SKP to 360 — SketchUp → Blender → AI Panorama Pipeline

Pipeline ini memvalidasi alur dari model SketchUp menuju panorama 360° berbasis geometri, dengan Blender headless sebagai backend resmi untuk conditioning render dan AI image edit sebagai tahap eksperimen.

Status saat ini: Controlled Generation Harness v0 selesai di atas Phase 0H. OBJ sudah dikunci sebagai transport geometri minimal, Phase 0D sudah menghasilkan equirectangular native, Phase 0F sudah membuktikan Cloudflare FLUX.2 Klein RGB-only edit bisa berjalan, Phase 0G sudah memvalidasi peningkatan sinyal source untuk kaca/jendela/daylight, Phase 0H sudah mengekstrak exterior-context truth dari geometri scene tanpa membuat objek luar baru, dan Controlled Core sekarang mengumpulkan semua source truth ke satu `output/condition_package/`.

## Prinsip utama

- SketchUp/OBJ geometry adalah sumber kebenaran.
- `scene.json` adalah sumber kebenaran kamera.
- Blender headless menghasilkan RGB/depth/normal deterministik.
- AI boleh meningkatkan realisme, tetapi tidak boleh menjadi design generator.
- Depth, normal, dan glass mask tidak dikirim sebagai conditioning palsu ke provider yang tidak mendukung structural control.
- Exporter SketchUp yang dipelihara di `sketchup/` menghasilkan OBJ, `scene.json`,
  dan `scene_entities.json`. Visualizer lama tetap menjadi referensi untuk worker
  AI dan cubemap, bukan jalur utama proyek ini.

## Struktur penting

```text
D:\Projects\skpto360
├── docs\
│   └── PRD — SketchUp to AI 360 Rendering Pipeline v0.1.md
├── output\
│   ├── obj\
│   │   ├── scene.obj
│   │   ├── scene.mtl
│   │   ├── scene.json
│   │   └── textures\
│   ├── phase0c\
│   ├── phase0d\
│   ├── phase0e\
│   ├── phase0f\
│   ├── phase0g\
│   └── phase0h\
├── scripts\
│   ├── run_phase0b_obj.ps1
│   ├── run_phase0c.ps1
│   ├── run_phase0d.ps1
│   ├── run_phase0e.ps1
│   ├── run_phase0f.ps1
│   ├── run_phase0g_source.ps1
│   ├── run_phase0g_flux_ab.ps1
│   └── run_phase0h.ps1
└── sketchup\
```

## Current validated flow

Phase 0C and Phase 0D remain available as `LEGACY / R&D` historical paths. They are not the production-critical Blender path: Phase 0C creates preview guide lights and Phase 0D uses native Blender ERP plus view-Z/far-clip depth.

The lightweight replacement is `scripts/run_blender_fast_passes.ps1`. It imports the OBJ once, uses Eevee rasterization with no lights, generates six canonical cubemap faces for unlit albedo, true radial depth, world normals, material IDs, and object IDs, then assembles every pass through `backend/projection.py`. It writes raw radial depth, percentile-conditioned depth, alignment diagnostics, hashes, and a pass report under `output/fast_passes/`.

Ada dua pipeline yang sengaja dipertahankan:

```text
PIPELINE A — FAST_BASELINE
SketchUp Ruby cubemap/direct capture
→ panorama/RGB reference
→ Cloudflare FLUX RGB-only
```

```text
PIPELINE B — CONTROLLED_CORE
SketchUp
→ OBJ + scene metadata
→ Blender headless native ERP source truth
→ RGB/depth/normal/masks/material/context contracts
→ constrained FLUX request where actually supported
→ deterministic QC
```

Pipeline A tetap berguna untuk preview cepat dan benchmark. Pipeline B adalah arah controlled product.

```text
SketchUp 2021
→ scene.obj + scene.mtl + textures/* + scene.json
→ Blender 5.1 native OBJ import
→ native equirectangular RGB/depth/normal
→ Phase 0E AI-ready inputs + QC metadata
→ Cloudflare FLUX.2 Klein RGB-only edit
→ Phase 0G source-signal A/B validation
→ Phase 0H exterior-context extraction
→ Controlled Core condition package + B0/B1 bake-off
```

## Quick start

Run these from PowerShell:

```powershell
cd D:\Projects\skpto360
.\scripts\run_phase0d.ps1
.\scripts\run_phase0e.ps1
.\scripts\run_phase0f.ps1
```

For Phase 0G:

```powershell
.\scripts\run_phase0g_source.ps1
.\scripts\run_phase0g_flux_ab.ps1
```

`run_phase0g_source.ps1` creates deterministic Blender source validation outputs. `run_phase0g_flux_ab.ps1` runs exactly two Cloudflare FLUX.2 Klein RGB-only generations for BEFORE vs AFTER comparison.

For Phase 0H:

```powershell
.\scripts\run_phase0h.ps1
```

`run_phase0h.ps1` creates an exterior/context pass by making detected glazing transparent only for diagnostic rendering, then classifies visible glazing regions by whether scene geometry exists behind them.

For Controlled Core:

```powershell
.\scripts\run_controlled_material_id.ps1
.\scripts\run_controlled_package.ps1
.\scripts\run_controlled_bakeoff.ps1
```

`run_controlled_material_id.ps1` renders deterministic material IDs and visible coverage. `run_controlled_package.ps1` builds `output/condition_package/`. `run_controlled_bakeoff.ps1` runs B0 FAST_BASELINE vs B1 CONTROLLED_CORE with the same Cloudflare model/source/settings and writes `output/generation_bakeoff/report.json`.

### NEW — Single-click panorama pipeline (1x FLUX)

One menu item inside SketchUp runs the full chain and produces a panorama with a single Cloudflare FLUX call:

```text
Extensions > RAD AI360 > Export AI360 & Render Panorama (1-Click Pipeline)
```

Flow: SketchUp Ruby exporter → `output/obj/` → Blender headless fast passes (`scripts/run_blender_fast_passes.ps1` equivalent) → self-contained harness condition package (fresh from ERP, no legacy phase reports) → one FLUX.2 Klein call → `output/pipeline/panorama.png` + `output/pipeline/report.json`.

The same chain runs from PowerShell:

```powershell
.\scripts\run_pipeline.ps1
# --skip-blender (reuse existing fast passes), --skip-flux (stop before the paid API call), --face-resolution, --erp-width, --guidance, --timeout
```

`output/pipeline/condition_package/` is rebuilt deterministically from the freshly rendered material-ID ERP (labels joined to OBJ material names via `output/fast_passes/id_index.json`, glazing via name/opacity heuristic). The harness monkeypatches `controlled_core_harness.PACKAGE_DIR` so prompt compilation, provenance, and QC read only pipeline-fresh assets.

## Required local inputs

The current pipeline expects:

```text
output/obj/scene.obj
output/obj/scene.mtl
output/obj/textures/*
output/obj/scene.json
```

`scene.json` must contain the authoritative SketchUp camera. The current validated scene is `AI360_1`.

The maintained SketchUp exporter also writes `scene_entities.json`. Run
`Extensions > RAD AI360 Exporter > Export AI360 Scene` and select `output/obj`
as the destination when refreshing the source transport. This metadata is
consumed by Phase 0K to project confirmed fixture candidates into ERP/cubemap
coordinates without inventing light positions.

## Cloudflare FLUX.2 Klein

Phase 0F/0G use the existing Cloudflare FLUX.2 Klein backend pattern from the old plugin reference:

```text
@cf/black-forest-labs/flux-2-klein-4b
```

Credentials are not stored in this project report. The scripts read either:

```text
CF_ACCOUNT_ID
CF_API_TOKEN
```

or the existing local plugin config:

```text
C:\Users\berka\AppData\Roaming\SketchUp\SketchUp 2021\SketchUp\Plugins\rad_ai360_visualizer\config.json
```

The plugin is used only as a reference/config source. The project backend lives in `D:\Projects\skpto360\scripts`.

## Phase outputs

### Controlled Generation Harness v0 — Controlled Core

```text
output/condition_package/
├── manifest.json
├── source/
│   ├── rgb_erp.png
│   ├── depth.exr
│   ├── depth_condition.png
│   ├── normal.exr
│   └── normal_condition.png
├── masks/
│   ├── glass.png
│   ├── exterior_context.png
│   ├── material_id.png
│   ├── geometry_edges.png
│   └── light_mask.png
├── context/exterior_context.png
├── semantics/
│   ├── materials.json
│   ├── lights.json
│   └── scene_contract.json
└── qc/source_metrics.json
```

Result: Phase 0D-0H conditioning assets now flow through one canonical downstream contract. The current Cloudflare provider still consumes only RGB plus prompt; `generation.json` records which controls are available and which are actually consumed.

Bake-off:

```text
output/generation_bakeoff/
├── B0_rgb_baseline/
├── B1_controlled_harness/
└── report.json
```

The v0 result is intentionally skeptical: the harness improves reproducibility, traceability, prompt determinism, and mask-aware QC, but does not prove semantic fidelity improvement because FLUX remains RGB+prompt-only in this project.

### Phase 0B.2 — OBJ transport

```text
output/phase0b_obj/
├── phase0b_obj_imported.blend
├── preview_camera.png
├── preview_overview.png
└── phase0b_obj_report.json
```

Result: OBJ accepted as preferred minimal transport. FBX was rejected because SketchUp 2021 exported ASCII FBX and Blender 5.1 rejected it.

### Phase 0C — Cubemap conditioning

```text
output/phase0c/
├── rgb/{front,right,back,left,top,bottom}.png
├── depth/*.exr
├── depth/*_preview.png
├── normal/*.exr
├── normal/*_preview.png
└── phase0c_report.json
```

Result: six-face RGB/depth/normal worked, but cubemap stitch seams are not the final path.

### Phase 0D — Native equirectangular source

```text
output/phase0d/
├── rgb/equirectangular.png
├── depth/equirectangular.exr
├── depth/equirectangular_preview.png
├── normal/equirectangular.exr
├── normal/equirectangular_preview.png
└── phase0d_report.json
```

Result: native Blender panoramic equirectangular render accepted as the main final-output geometry render path.

### Phase 0E — AI feasibility prep

```text
output/phase0e/
├── input/rgb_erp.png
├── input/depth_condition.png
├── input/normal_condition.png
├── input/material_manifest.json
└── phase0e_report.json
```

Result: AI-ready inputs and deterministic QC/reporting prepared. No fake depth/normal structural conditioning.

### Phase 0F — Cloudflare RGB-only ERP edit

```text
output/phase0f/
├── input/rgb_erp.png
├── input/rgb_erp_cloudflare_reference_511.png
├── experiment_a_rgb_only/output.png
├── experiment_a_rgb_only/generation.json
└── phase0f_report.json
```

Result: Cloudflare FLUX.2 Klein generated a valid RGB-only ERP image.

### Phase 0G — Source signal validation

```text
output/phase0g/
├── glass_candidates.json
├── light_audit.json
├── phase0g_report.json
├── masks/glass_mask.png
├── comparison/BEFORE.png
├── comparison/AFTER_SOURCE_FIX.png
└── ai_ab/
    ├── G-A_BEFORE_RGB_ONLY/output.png
    └── G-B_AFTER_SOURCE_FIX_RGB_ONLY/output.png
```

Result: source-signal improvement helped AI read glazing/daylight better, but exterior context remains the next likely bottleneck.

### Phase 0H — Exterior context conditioning

```text
output/phase0h/
├── exterior_context/
│   ├── panorama.png
│   ├── mask.png
│   ├── _geometry_presence.png
│   └── _glass_id_mask.png
├── comparison/
│   ├── SOURCE_G.png
│   └── CONTEXT_PASS.png
├── exterior_context.json
└── phase0h_report.json
```

Result: visible glazing was classified by whether deterministic scene geometry is visible behind it when glazing is made transparent for the context pass. No exterior objects, landscaping, sky generation, or AI reference images were created.

## Current Phase 0G findings

- 10 glazing candidates detected deterministically.
- Glass mask generated and aligned with the ERP camera.
- 1 low-confidence light fixture material candidate found: `Light_metal`.
- No lights reconstructed because OBJ evidence was insufficient to recover reliable emitter transforms.
- Geometry and camera validation passed.
- Two controlled FLUX RGB-only runs completed.
- AFTER source fix produced better window/daylight interpretation than BEFORE.

## Current Phase 0H findings

- Phase 0G is frozen as the accepted baseline.
- Two visible glazing material regions were classified as `SCENE_GEOMETRY`.
- Eight glass candidates were `UNKNOWN` because those materials were not visible from the active ERP camera.
- Context mask was generated separately from the full glass mask.
- Constrained context pixels covered about `92.5%` of visible glass-mask pixels.
- AI context-conditioned A/B was not run because the existing Cloudflare FLUX.2 Klein interface in this project only accepts ordinary image references, not a true structural exterior-context/mask control.
- The correct conclusion is: deterministic exterior/context extraction is available, but using it meaningfully for AI requires a provider/interface that supports real control input.

## What not to do yet

Do not add these until explicitly planned:

- Astra/GPT/Claude orchestration
- semantic material AI
- new AI providers
- fake depth/normal conditioning
- ControlNet workarounds
- automatic retry agents
- cubemap seam repair as final output path
- production lighting/material workflow
- landscape generation
- procedural exterior/context generation

## Useful commands

```powershell
# Native ERP RGB/depth/normal
.\scripts\run_phase0d.ps1

# Prepare AI-oriented inputs and manifest
.\scripts\run_phase0e.ps1

# Run one Cloudflare RGB-only ERP edit
.\scripts\run_phase0f.ps1

# Phase 0G deterministic source validation
.\scripts\run_phase0g_source.ps1

# Phase 0G two-image FLUX A/B test
.\scripts\run_phase0g_flux_ab.ps1

# Phase 0H exterior context extraction
.\scripts\run_phase0h.ps1
```

## Review guidance

When comparing AI output, do not judge only by beauty. Check:

- windows remain openings;
- glazing remains transparent/believable;
- outdoor/daylight signal is preserved;
- doors/openings/ceiling are not redesigned;
- furniture count and placement remain stable;
- light fixture count and position remain stable;
- ERP remains 2:1 and seam remains usable.
