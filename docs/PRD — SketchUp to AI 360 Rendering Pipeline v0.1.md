# Product Requirements Document
## SketchUp → AI 360 Rendering Pipeline

**Version:** 0.1  
**Status:** Proof of Concept  
**Primary User:** Interior Designer / 3D Visualizer  
**Source Application:** SketchUp  
**Final Output:** Photorealistic AI-generated 360° panorama  
**Project Type:** R&D / Production Pipeline

---

# 1. Background

Traditional architectural visualization typically follows a workflow such as:

SketchUp → 3ds Max → Corona / V-Ray → Photoshop → Final Render

This workflow can produce highly controlled photorealistic results but requires significant manual setup, material conversion, lighting adjustment, rendering time, and post-production.

For the current project, the client specifically requires an **AI-generated rendering workflow**.

Therefore, the objective is not simply to automate a conventional renderer.

The system must use generative AI as a meaningful part of final image generation while preserving the architecture and design contained in the SketchUp model.

The SketchUp model is treated as the **design source of truth**.

AI is allowed to improve:

- material realism;
- lighting realism;
- surface imperfections;
- photographic characteristics;
- environmental appearance;
- micro-detail.

AI must not freely redesign:

- walls;
- openings;
- doors;
- ceilings;
- fixtures;
- furniture placement;
- spatial proportions;
- major material assignments.

---

# 2. Problem Statement

Current generative image models can produce highly photorealistic architectural images but have several problems when used directly with SketchUp screenshots:

1. geometry can change;
2. objects can move or disappear;
3. repeated fixtures may become inconsistent;
4. material identity may change;
5. 360° panorama boundaries may not match;
6. equirectangular projection may become distorted;
7. ceiling and floor regions may deform;
8. AI may invent architectural elements;
9. multiple generated views may not represent the same scene.

The product must find a practical way to combine:

**deterministic 3D geometry**

with

**generative AI appearance generation.**

---

# 3. Product Principle

The fundamental system principle is:

> **SketchUp defines what exists.  
> Blender describes where it exists.  
> AI determines how realistic it looks.**

Blender must not become a replacement for the traditional rendering workflow.

Blender acts primarily as a **geometry conditioning engine**.

The final appearance should be materially generated or transformed by AI.

---

# 4. Mandatory Requirements

The following are non-negotiable requirements.

## 4.1 AI Rendering

The final visual must meaningfully use generative AI.

A conventional Blender/Cycles render with only minor AI sharpening does not satisfy this requirement.

AI must materially contribute to:

- lighting appearance;
- material appearance;
- realism;
- photographic quality.

---

## 4.2 SketchUp as Source of Truth

The SketchUp model controls:

- geometry;
- room dimensions;
- fixture position;
- furniture position;
- openings;
- ceiling layout;
- camera location;
- broad material intent.

No manual rebuilding of the project inside Blender should be required.

---

## 4.3 360° Output

Final deliverable:

**360° × 180° equirectangular panorama**

Target aspect ratio:

**2:1**

Example production sizes:

- 4096 × 2048;
- 6144 × 3072;
- 8192 × 4096.

The output must work correctly inside a standard 360 panorama viewer.

---

## 4.4 Structural Preservation

The generated image must preserve the design sufficiently for architectural presentation.

Critical elements must not:

- disappear;
- duplicate;
- change shape significantly;
- move substantially;
- change count.

---

## 4.5 Seam Continuity

Longitude:

**0° and 360°**

must connect without a visually obvious break.

---

# 5. Ideal Requirements

These are desirable but must not delay the initial proof of concept.

- One-click export from SketchUp.
- Automatic material understanding.
- Automatic scene analysis.
- Automatic AI prompting.
- Automatic failure detection.
- Automatic correction loop.
- Batch rendering of multiple panorama cameras.
- Cloud rendering.
- Project presets.
- Material reference library.
- Human-readable rendering report.
- Cost tracking per generated panorama.

---

# 6. Explicit Non-Goals for V0 / V1

The project will NOT initially attempt to build:

- a new render engine;
- a Corona replacement;
- a complete Blender material pipeline;
- physically perfect shaders;
- a BIM conversion system;
- a universal SketchUp cleanup engine;
- automatic PBR material libraries;
- an unrestricted Blender AI agent;
- a full desktop application;
- a distributed render farm.

These features may be considered only after the AI rendering method is proven.

---

# 7. Proposed Architecture

```text
                 SKETCHUP
                    │
                    │
              Export Scene
                    │
                    ▼
             GLB / Scene Data
                    │
                    ▼
       ┌─────────────────────────┐
       │    BLENDER HEADLESS     │
       │                         │
       │ Geometry Truth Engine   │
       │                         │
       │ • cameras               │
       │ • RGB guide             │
       │ • depth                 │
       │ • normals               │
       │ • object masks          │
       │ • material masks        │
       │ • cubemap               │
       └────────────┬────────────┘
                    │
                    ▼
       ┌─────────────────────────┐
       │      AI ORCHESTRATOR    │
       │                         │
       │ Astra-class multimodal  │
       │ reasoning model         │
       │                         │
       │ • inspect scene         │
       │ • understand materials  │
       │ • prepare prompts       │
       │ • choose strategy       │
       │ • inspect results       │
       │ • detect errors         │
       └────────────┬────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │   AI IMAGE ENGINE     │
        │                       │
        │ GPT Image / FLUX /    │
        │ panorama diffusion /  │
        │ alternative model     │
        └────────────┬──────────┘
                     │
                     ▼
               AI Render Faces
                     │
                     ▼
        Cubemap → Equirectangular
                     │
                     ▼
                AI / Geometry QC
                     │
           ┌─────────┴─────────┐
           │                   │
         PASS                 FAIL
           │                   │
           ▼                   ▼
       Final 360          Regenerate /
                          repair region
```

---

# 8. Why Blender Headless

Blender is not used because the user needs Blender.

It is used because it provides a deterministic 3D environment capable of producing geometry-based information that image AI cannot reliably infer from a screenshot.

Blender responsibilities:

- import scene;
- reconstruct camera;
- generate perspective views;
- generate cubemap;
- generate depth;
- generate normal maps;
- generate object masks;
- generate material masks;
- convert cubemap to panorama;
- perform deterministic image projection operations.

Blender should run without requiring the designer to interact with its UI.

Example concept:

```text
blender --background scene.blend --python pipeline.py
```

The user should ideally never need to open Blender manually.

---

# 9. Why Blender Must NOT Become the Renderer

The following should be avoided:

```text
SketchUp
→ Blender
→ rebuild materials
→ setup lighting
→ Cycles production render
→ AI polish
```

This recreates the traditional visualization workflow using different software.

It provides little benefit compared with:

SketchUp → 3ds Max → Corona.

Instead, Blender output should intentionally remain inexpensive.

Example:

```text
Flat materials
+
simple lighting
+
correct geometry
+
correct camera
```

The AI model handles the photorealistic transformation.

---

# 10. AI Orchestrator

A multimodal reasoning model may serve as the **Render Director**.

Its responsibilities are semantic, not geometric.

Example responsibilities:

### Scene interpretation

Input:

```text
Material23
Wood_01_copy
pink01
metal123
glass2
```

Possible semantic output:

```json
{
  "Material23": {
    "class": "paint",
    "appearance": "warm white matte paint"
  },
  "Wood_01_copy": {
    "class": "wood",
    "appearance": "light oak"
  },
  "pink01": {
    "class": "solid_surface",
    "appearance": "matte muted pink"
  },
  "metal123": {
    "class": "metal",
    "appearance": "brushed stainless steel"
  },
  "glass2": {
    "class": "glass",
    "appearance": "clear architectural glass"
  }
}
```

No strict material naming convention is required for the initial prototype.

---

# 11. Restricted Agent Controls

The AI orchestrator must NOT receive unrestricted Blender control.

Instead, the Blender worker should expose deterministic functions such as:

```text
import_scene()

inspect_scene()

get_materials()

get_objects()

set_camera()

render_preview()

render_cubemap()

render_depth()

render_normals()

render_object_masks()

render_material_masks()

convert_cubemap_to_erp()

create_region_mask()
```

The reasoning model determines **which operation is required**.

The Blender worker determines **how the operation is performed**.

This prevents the AI from randomly modifying the scene.

---

# 12. Primary Generation Strategy

The initial hypothesis to test is:

## Cubemap-Based AI Generation

Instead of giving the AI one highly distorted equirectangular image, the scene will first be represented as six conventional perspective views.

```text
          TOP

LEFT  FRONT  RIGHT  BACK

         BOTTOM
```

Each face represents a 90° field of view.

For each direction the system may generate:

```text
RGB
Depth
Normal
Material Mask
Object Mask
```

Example dataset:

```text
/front/rgb.png
/front/depth.exr
/front/normal.png

/right/rgb.png
/right/depth.exr
/right/normal.png

/back/rgb.png
...
```

---

# 13. Why Cubemap First

Most modern generative image models are primarily trained on conventional perspective images.

An equirectangular image contains substantial spherical distortion.

For example:

- ceilings stretch toward the top;
- floors stretch toward the bottom;
- horizontal geometry changes scale around the sphere.

A cubemap keeps each AI generation close to a conventional architectural photograph.

Therefore the initial project hypothesis is:

> **Generating perspective cube faces may produce better realism and geometry preservation than directly generating an equirectangular panorama.**

This remains a hypothesis and must be validated experimentally.

---

# 14. Multi-Face Consistency Problem

Cubemap generation introduces another major problem:

six independently generated images may not represent exactly the same scene.

Example:

```text
FRONT:
counter = pink terrazzo

RIGHT:
same counter = pink marble
```

or:

```text
FRONT edge:
3 shelves

RIGHT edge:
4 shelves
```

Therefore cube faces must not be treated as unrelated generations.

---

# 15. Proposed Consistency Strategy

Each generation should receive:

- original SketchUp/Blender RGB view;
- depth;
- normal;
- adjacent generated face;
- material description;
- scene description;
- camera metadata.

Overlapping context may also be generated.

Instead of exactly 90°:

```text
Face output region = 90°

Generation context = 110–120°
```

The extra region exists only to provide consistency context.

Final projection uses the central valid region.

Concept:

```text
           Generated context

        |------------------|
        |    valid face    |
        |     90 deg       |
        |------------------|

extra context           extra context
```

---

# 16. Alternative Strategy: Direct ERP

The project must also evaluate direct panorama generation.

```text
Blender ERP RGB
+
ERP depth
+
ERP normal
+
AI conditioning
↓
AI ERP
```

This must be benchmarked rather than assumed superior.

Potential benefits:

- no multi-face consistency problem;
- naturally single image;
- easy panorama output.

Potential risks:

- distortion;
- pole artifacts;
- AI unfamiliarity with ERP projection;
- weaker geometry preservation.

---

# 17. AI Engine Abstraction

The architecture must avoid hard-coding the system to one image model.

Interface:

```text
generate(
    rgb,
    depth,
    normal,
    masks,
    references,
    prompt,
    settings
)
```

Possible implementations:

```text
GPT Image
FLUX
Stable Diffusion
ControlNet
Panorama-specific diffusion
Future image model
```

The best engine should be selected empirically.

---

# 18. Prompt Generation

Prompts should be scene-specific.

Example input:

```json
{
  "project": "beauty retail store",
  "design_style": "modern minimal",
  "lighting": "bright neutral retail lighting",
  "camera": "eye level",
  "materials": [
    "muted pink solid surface",
    "brushed stainless steel",
    "light oak",
    "mosaic tile"
  ]
}
```

Generated prompt may resemble:

```text
Photorealistic contemporary beauty retail interior.

Preserve the exact architectural geometry, furniture placement,
fixture count, shelving dimensions and camera composition
from the input image.

Bright neutral commercial lighting.

Muted matte pink solid surface,
subtle brushed stainless steel,
light natural oak,
fine mosaic wall finish.

Realistic material microtexture,
soft indirect illumination,
architectural photography,
natural exposure.

Do not redesign the interior.
Do not move furniture.
Do not add objects.
```

---

# 19. AI Quality Control

Generated images must not automatically become final outputs.

The AI orchestrator should perform visual comparison between:

```text
SOURCE GEOMETRY

vs.

GENERATED IMAGE
```

Checks include:

### Geometry

- wall positions;
- openings;
- ceiling shape;
- countertop shape;
- furniture position.

### Object Count

Example:

```text
source stools = 6
generated stools = 7
```

Result:

```text
FAIL
```

### Material Identity

Example:

```text
source:
brushed stainless

generated:
black painted metal
```

Result:

```text
FAIL / WARNING
```

### Panorama Boundary

Verify that features crossing:

```text
359°
→
0°
```

remain continuous.

---

# 20. QC Output

Example machine-readable QC:

```json
{
  "status": "FAIL",
  "score": 0.82,
  "issues": [
    {
      "type": "geometry_change",
      "object": "cashier_counter",
      "severity": "high"
    },
    {
      "type": "missing_object",
      "object": "track_light_07",
      "severity": "medium"
    },
    {
      "type": "material_change",
      "object": "feature_wall",
      "severity": "medium"
    }
  ]
}
```

This data may trigger targeted regeneration.

---

# 21. Human Approval

AI QC does not replace designer approval.

Workflow:

```text
Generate
↓
Automated QC
↓
Designer Preview
↓
Approve / Retry
```

The designer remains responsible for judging whether the image still represents the approved design.

---

# 22. SketchUp Integration — V0

Do not build a large SketchUp extension initially.

V0 may use:

```text
SketchUp
↓
manual GLB export
↓
pipeline folder
```

Example:

```text
project/
    scene.glb
    references/
    output/
```

This intentionally minimizes engineering work.

---

# 23. SketchUp Integration — V1

After the rendering approach works, create a small Ruby extension.

UI example:

```text
RAD AI Render

Camera:
[ Current Camera ]

Output:
[ 4096 × 2048 ]

Style:
[ Photoreal Interior ]

References:
[ Add Images ]

[ GENERATE AI 360 ]
```

Plugin responsibilities:

- export GLB;
- store camera location;
- store camera direction;
- collect material metadata;
- send job;
- display status;
- open result.

Nothing more initially.

---

# 24. Scene Manifest

Optional metadata:

```json
{
  "project": "Sociolla Store",
  "camera": {
    "name": "CAM_01",
    "position": [1.2, 3.4, 1.6]
  },
  "materials": [
    {
      "name": "Pink Material"
    },
    {
      "name": "Metal01"
    }
  ]
}
```

The manifest provides semantic context without altering geometry.

---

# 25. Proof-of-Concept Experiment

The first POC should intentionally be small.

Use:

- one SketchUp interior;
- one 360 camera;
- relatively clean geometry;
- several recognizable materials;
- repeated furniture;
- doors/openings;
- ceiling lights.

Prefer a scene where geometry errors are obvious.

---

# 26. Benchmark Candidates

At minimum test:

## Experiment A — Direct AI Edit

```text
rough RGB
→ image editing model
→ final
```

Purpose:

determine whether current high-end image models already preserve enough structure without advanced conditioning.

---

## Experiment B — Geometry Conditioned AI

```text
RGB
+ depth
+ normal
→ conditioned generation
```

Purpose:

measure whether geometry conditioning substantially reduces hallucination.

---

## Experiment C — Cubemap AI

```text
6 RGB
+ depth
+ normal
→ multi-face generation
→ ERP
```

Purpose:

test whether perspective generation improves visual quality enough to justify the consistency problem.

---

## Experiment D — Direct ERP AI

```text
ERP RGB
+ ERP depth
→ panorama-aware generation
```

Purpose:

determine whether a dedicated panorama method provides superior spherical consistency.

---

# 27. Evaluation Matrix

Every output should be scored against the same criteria.

| Criterion | Weight |
|---|---:|
| Geometry preservation | Critical |
| Object preservation | Critical |
| 360 continuity | Critical |
| Projection correctness | Critical |
| Photorealism | High |
| Material accuracy | High |
| Lighting quality | Medium |
| Repeatability | High |
| Generation speed | Medium |
| Cost | Medium |

A visually beautiful image fails if the architecture changes substantially.

---

# 28. Automatic Failure Conditions

The result should immediately fail QC if:

- major wall moves;
- opening disappears;
- large fixture changes form;
- furniture count significantly changes;
- camera perspective changes;
- severe panorama seam appears;
- ERP becomes invalid;
- large region is hallucinated.

---

# 29. Success Criteria for POC

The POC is considered successful when one pipeline can generate a panorama that:

1. clearly looks AI-generated / AI-rendered;
2. remains recognizably the exact SketchUp design;
3. has no major object hallucinations;
4. produces a valid 360 panorama;
5. has no obvious longitude seam;
6. requires substantially less manual work than traditional 3ds Max + Corona;
7. can be reproduced on a second camera from the same project.

The POC does NOT need to be fully automatic.

---

# 30. Performance Goal

Initial goal:

```text
Designer setup:
< 5 minutes

Manual AI intervention:
minimal

Final output:
one usable AI 360 panorama
```

Raw compute duration is less important than reducing operator work.

---

# 31. Cost Model

Every generated panorama should eventually log:

```text
AI image calls
reasoning calls
GPU runtime
number of retries
final generation cost
```

Example:

```json
{
  "job": "CAM_01",
  "generation_attempts": 4,
  "ai_cost": 3.20,
  "gpu_cost": 0.60,
  "total_cost": 3.80
}
```

The cheapest model is not automatically preferred.

Cost must be considered relative to human rendering hours.

---

# 32. Risks

## Risk 1 — AI Geometry Drift

Most serious technical risk.

Mitigation:

- depth;
- normals;
- masks;
- low transformation strength;
- automated QC;
- region regeneration.

---

## Risk 2 — Cubemap Face Inconsistency

Mitigation:

- overlap;
- adjacent image references;
- shared seed where supported;
- shared scene description;
- boundary-aware regeneration.

---

## Risk 3 — Equirectangular Distortion

Mitigation:

compare direct ERP against cubemap workflows rather than assuming either is correct.

---

## Risk 4 — Dirty SketchUp Files

Do not solve this by requiring perfect naming.

The initial pipeline must tolerate:

```text
Material23
Group#47
Component#99
```

Semantic interpretation may be performed by vision/reasoning models.

---

## Risk 5 — Overengineering

This is a major project risk.

No infrastructure should be built until a visual generation method proves successful.

Rule:

> Every engineering feature must solve an observed production problem.

---

# 33. Development Phases

## Phase 0 — Feasibility

Goal:

**Can AI create one acceptable 360 interior from one SketchUp scene?**

Tasks:

- export one model;
- generate geometry passes;
- test image models;
- inspect panorama;
- compare approaches.

No plugin.

No UI.

No material system.

---

## Phase 1 — Reproducible Pipeline

Goal:

make the successful experiment repeatable.

Add:

- scripts;
- configuration;
- standardized folders;
- automatic Blender execution;
- cubemap conversion;
- QC.

---

## Phase 2 — SketchUp Integration

Goal:

designer can launch generation from SketchUp.

Add:

- Ruby plugin;
- camera selection;
- GLB export;
- job submission;
- result viewer.

---

## Phase 3 — AI Render Director

Add:

- automatic material interpretation;
- prompt generation;
- QC;
- retry logic;
- targeted corrections.

---

## Phase 4 — Production System

Potential additions:

- multiple cameras;
- job queue;
- GPU cloud;
- presets;
- project memory;
- material references;
- cost tracking;
- team workflows.

---

# 34. Recommended First Implementation

The first implementation should be deliberately ugly internally.

Example:

```text
POC/
│
├── input/
│   └── scene.glb
│
├── blender/
│   └── extract.py
│
├── passes/
│   ├── front/
│   ├── right/
│   ├── back/
│   ├── left/
│   ├── top/
│   └── bottom/
│
├── references/
│
├── generated/
│
└── final/
```

Command:

```text
run_poc scene.glb
```

Result:

```text
final_360.png
report.json
```

No production-grade application architecture is required yet.

---

# 35. Core Decision Gates

## Gate 1

Can a current AI image model preserve the SketchUp scene sufficiently?

If **NO**:

investigate stronger structural conditioning.

If **YES**:

avoid unnecessary ControlNet complexity.

---

## Gate 2

Does direct ERP generation work reliably?

If **YES**:

prefer direct ERP because it removes multi-face complexity.

If **NO**:

continue cubemap development.

---

## Gate 3

Does cubemap generation maintain cross-face consistency?

If **NO**:

test panorama-native diffusion or stronger multi-view conditioning.

---

## Gate 4

Does Blender provide meaningful value?

If the same quality can be achieved using V-Ray/another existing renderer as the conditioning generator with significantly less complexity:

remove Blender.

Blender is not sacred architecture.

---

## Gate 5

Does an AI orchestrator materially improve output?

If automated reasoning does not improve:

- material interpretation;
- consistency;
- QC;
- retry success;

do not place an expensive LLM in every render job.

---

# 36. Engineering Philosophy

This project should not assume that more AI equals a better system.

Likewise it should not assume that more deterministic engineering equals a better product.

The final architecture should be selected based on actual rendered results.

Key principle:

> **Use deterministic software for geometry.  
> Use generative AI where generation actually adds value.  
> Do not ask AI to solve problems that mathematics already solves perfectly.**

---

# 37. Final Product Vision

The long-term user experience may eventually become:

```text
SketchUp

Place Camera

↓

AI 360 Render

↓

Select:
[ Photoreal Commercial Interior ]

↓

Generate

↓

AI analyzing scene...
Geometry extracted...
AI rendering...
Checking design consistency...

↓

360 PREVIEW

[ Approve ]
[ Regenerate ]
[ Edit Material ]
```

The designer remains inside SketchUp.

Internally the system may use:

- Blender;
- image generation models;
- reasoning models;
- GPU workers;
- projection utilities.

The designer does not need to know or operate them.

---

# 38. Product Vision Statement

> Build a geometry-aware AI rendering pipeline capable of transforming relatively clean SketchUp interior models into photorealistic 360° AI-generated panoramas while preserving the approved architectural design and eliminating the traditional SketchUp → 3ds Max → Corona → Photoshop production workflow.