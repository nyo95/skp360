# Controlled Pipeline Audit

Status: Controlled Generation Harness v0 added on 2026-09-16.

This audit records the useful conditioning signals produced by Phase 0A-0H and how the new Controlled Core package uses them. It distinguishes source truth, deterministic semantic contracts, provider-consumed generation inputs, and QC inputs.

## Pipeline Boundary

Two pipelines are intentional:

- FAST_BASELINE: SketchUp Ruby cubemap/direct RGB source -> panorama/reference image -> Cloudflare FLUX RGB-only. This remains useful for fast preview, benchmark, and quick iteration.
- CONTROLLED_CORE: SketchUp/OBJ/scene metadata -> Blender headless ERP source truth -> RGB/depth/normal/masks/material/context contracts -> constrained FLUX request where supported -> deterministic QC.

The current Cloudflare FLUX path does not expose verified depth, normal, mask, or ControlNet-style structural conditioning. The harness therefore records those assets as available controls and uses them for prompt compilation, deterministic contracts, and QC, while generation metadata states that the provider actually consumed only RGB and prompt.

## Asset Inventory

| Asset | File/source | Producer | Data type | Coordinate/projection | Resolution | Current consumer | FLUX receives it? | QC consumes it? | Classification | Strategic value |
|---|---|---|---|---|---:|---|---|---|---|---|
| RGB ERP | `output/phase0g/comparison/AFTER_SOURCE_FIX.png` copied to `output/condition_package/source/rgb_erp.png` | Phase 0G Blender source-signal pass | PNG RGB | equirectangular ERP | 2048x1024 | Controlled package, B0/B1 FLUX reference | Yes, as resized ordinary image reference | Yes | GENERATION_INPUT | Primary visual source and provider reference. |
| Depth EXR | `output/phase0d/depth/equirectangular.exr` copied to `source/depth.exr` | Phase 0D Blender | OpenEXR metric depth | Blender panoramic ERP, metres | 2048x1024 | Controlled package | No | Available for future/derived QC | QC_INPUT | First-class geometry truth; not fake-sent as image conditioning. |
| Depth condition PNG | `output/phase0e/input/depth_condition.png` copied to `source/depth_condition.png` | Phase 0E | PNG depth guide | ERP, near/far mapped 0.2m-20m | 2048x1024 | Controlled package | No | Yes/metadata | QC_INPUT | Human/provider-ready condition preview if a future provider supports depth. |
| Normal EXR | `output/phase0d/normal/equirectangular.exr` copied to `source/normal.exr` | Phase 0D Blender | OpenEXR encoded normal | Blender world/shading normal, ERP | 2048x1024 | Controlled package | No | Available for future/derived QC | QC_INPUT | Geometry orientation truth. |
| Normal condition PNG | `output/phase0e/input/normal_condition.png` copied to `source/normal_condition.png` | Phase 0E | PNG RGB normal guide | ERP, normal * 0.5 + 0.5 | 2048x1024 | Controlled package | No | Yes/metadata | QC_INPUT | Provider-ready condition preview if future interface supports normals. |
| Material manifest | `output/phase0e/input/material_manifest.json` compiled into `semantics/materials.json` | Phase 0E Blender material extraction | JSON material IDs/colors/textures | Source material namespace | 198 materials | Controlled material contract | Prompt receives top visible material evidence only | Yes | SEMANTIC_INPUT | Stable IDs and non-AI material evidence for future semantic interpretation. |
| Material ID mask | `output/condition_work/material_id.png` copied to `masks/material_id.png` | Controlled material-ID pass | PNG deterministic color ID | ERP | 2048x1024 | Controlled package | No | Yes/coverage | QC_INPUT | Makes visible material coverage deterministic. |
| Glass mask | `output/phase0g/masks/glass_mask.png` copied to `masks/glass.png` | Phase 0G | PNG binary mask | ERP | 2048x1024 | Scene contract, QC | No | Yes | QC_INPUT | Preserves glazing regions and separates glass from ordinary material drift. |
| Exterior context panorama | `output/phase0h/exterior_context/panorama.png` copied to `context/exterior_context.png` | Phase 0H | PNG RGB context pass | ERP | 2048x1024 | Scene contract/reference metadata | No | Yes/metadata | REFERENCE_ONLY | Deterministic view of geometry behind transparent glazing. |
| Exterior context mask | `output/phase0h/exterior_context/mask.png` copied to `masks/exterior_context.png` | Phase 0H | PNG binary mask | ERP | 2048x1024 | Scene contract, QC | No | Yes | QC_INPUT | Distinguishes KNOWN_EXTERIOR from unknown/unconstrained glazing. |
| Geometry edges | `output/condition_package/masks/geometry_edges.png` | Controlled package builder | PNG Canny edge mask | ERP | 2048x1024 | QC/reference | No | Yes | QC_INPUT | Lightweight structural drift warning. |
| Light mask | `output/condition_package/masks/light_mask.png` | Controlled package builder | PNG binary mask | ERP | 2048x1024 | Scene contract/QC | No | Yes, as UNKNOWN | QC_INPUT | Empty by design until reliable light positions exist. |
| Light audit | `output/phase0g/light_audit.json` compiled into `semantics/lights.json` | Phase 0G | JSON candidate audit | Material namespace | n/a | Controlled light contract | Prompt receives "do not invent luminaires" guidance | Yes | SEMANTIC_INPUT | Preserves weak evidence without inventing fixture positions. |
| Camera metadata | `output/obj/scene.json`, Phase 0D report | SketchUp exporter + Blender validation | JSON camera/basis | SketchUp mm and Blender m | n/a | Scene contract, all renders | Prompt text only | Yes | QC_INPUT | Authoritative camera/projection contract. |
| Geometry hashes | `scene.obj`, `scene.json`, package manifest hashes | Phase 0D/0G/H + package builder | SHA-256 metadata | n/a | n/a | Manifest/generation metadata | No | Yes | QC_INPUT | Reproducibility and source-truth identity. |
| Seam metrics | Phase 0D/0H reports and bake-off QC | Phase 0D/0H/controlled QC | numeric metrics | ERP left/right boundary | n/a | QC reports | No | Yes | QC_INPUT | Validates panorama usability. |

## Canonical Condition Package

The canonical package is:

```text
output/condition_package/
  manifest.json
  source/
    rgb_erp.png
    depth.exr
    depth_condition.png
    normal.exr
    normal_condition.png
  masks/
    glass.png
    exterior_context.png
    material_id.png
    geometry_edges.png
    light_mask.png
  context/
    exterior_context.png
  semantics/
    materials.json
    lights.json
    scene_contract.json
  qc/
    source_metrics.json
```

Downstream controlled code reads this package instead of reading `phase0d/`, `phase0e/`, `phase0g/`, or `phase0h/` directly.

## Scene Contract

`semantics/scene_contract.json` contains deterministic camera, projection, region, constraint, and source hash data. It explicitly separates:

- INTERIOR: implicit source RGB non-glazing regions;
- GLAZING: Phase 0G deterministic glass candidates and mask;
- KNOWN_EXTERIOR: Phase 0H context mask/regions with `SCENE_GEOMETRY`;
- UNKNOWN: glazing regions with no visible deterministic context or unconstrained regions.

No speculative semantic descriptions are included.

## Material Contract

`semantics/materials.json` contains 198 source materials, including:

- deterministic material ID;
- SketchUp/OBJ material name;
- base color;
- opacity;
- visible pixel coverage from `masks/material_id.png`;
- glazing flag from Phase 0G evidence;
- source texture reference when available;
- `semantic_class: null`.

No LLM material interpretation is used.

## Light Contract

`semantics/lights.json` preserves the Phase 0G `Light_metal` material-name candidate as unresolved low-confidence evidence. No ERP position or light mask is projected because the current `scene.json`/OBJ export does not contain source-level component/group/instance transforms.

This is an upstream exporter gap, not a generation feature. Required future source-truth fields are component names, group names, instance names, tags/layers, world-space transforms, and bounding boxes/insertion points.

## Controlled FLUX Harness

The harness keeps Cloudflare FLUX as a provider implementation, not architecture. Each `generation.json` records:

- `available_controls`: depth, normal, glass mask, exterior context, material IDs, geometry edges, light positions;
- `actually_consumed`: RGB and prompt;
- `not_consumed_by_provider`: controls unavailable to the current Cloudflare interface.

No depth/normal/mask image is sent as pretend structural conditioning.

## Bake-Off Result

The controlled comparison is in:

```text
output/generation_bakeoff/
  B0_rgb_baseline/
  B1_controlled_harness/
  report.json
```

Both B0 and B1 used the same Cloudflare model, source RGB, guidance, and output size. The current provider does not expose seed control. B1 used the deterministic prompt compiler and records the package/QC contract.

Result: deterministic v0 QC confirms valid ERP projection and seam status for both outputs and records mask-region warning signals for B1. It does not prove semantic fidelity improvement. The current evidence is that the harness improves reproducibility, traceability, and constraint-aware QC, while FLUX fidelity remains limited by RGB+prompt-only consumption.

## Tech Debt Classification

- KEEP: OBJ transport, Phase 0D native ERP, Phase 0E condition images/material manifest, Phase 0G glass mask/source signal, Phase 0H exterior context, Cloudflare FLUX RGB-only backend.
- REFERENCE: old SketchUp `RAD AI360 Visualizer` cubemap worker, Phase 0C cubemap outputs, DAE proof, Phase 0B FBX failure notes.
- DEPRECATE AS RUNTIME PATH: DAE fallback parser, FBX experiment, cubemap stitch experiments as primary controlled architecture.
- REMOVE: nothing removed in v0; phase history is intentionally preserved.
