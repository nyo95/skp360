# SketchUp Exporter

The maintained exporter source for this repository lives here. Install the
loader and `rad_ai360_exporter/` directory into the SketchUp Plugins folder.

`Export AI360 Scene` writes the canonical OBJ transport, `scene.json`,
normalized textures, and `scene_entities.json` in one export directory.
`scene_entities.json` preserves group/component instance transforms, bounds,
tags, materials, and light-keyword evidence for the Phase 0K light contract.

The separate `Export Entity Metadata` command is useful when the OBJ export
already exists and only source entity metadata needs to be refreshed.