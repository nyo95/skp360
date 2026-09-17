$ErrorActionPreference = "Stop"

$ProjectRoot = "D:\Projects\skpto360"
$BlenderExe = "C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"
$ObjPath = Join-Path $ProjectRoot "output\obj\scene.obj"
$JsonPath = Join-Path $ProjectRoot "output\obj\scene.json"
$MaterialManifestPath = Join-Path $ProjectRoot "output\phase0e\input\material_manifest.json"
$OutDir = Join-Path $ProjectRoot "output\condition_work"
$ScriptPath = Join-Path $ProjectRoot "scripts\controlled_material_id_pass.py"

if (!(Test-Path -LiteralPath $BlenderExe)) { throw "Blender executable not found: $BlenderExe" }
if (!(Test-Path -LiteralPath $ObjPath)) { throw "OBJ file not found: $ObjPath" }
if (!(Test-Path -LiteralPath $JsonPath)) { throw "scene.json not found: $JsonPath" }
if (!(Test-Path -LiteralPath $MaterialManifestPath)) { throw "material_manifest.json not found: $MaterialManifestPath" }

New-Item -ItemType Directory -Path $OutDir -Force | Out-Null

& $BlenderExe --background --python $ScriptPath -- `
  --obj $ObjPath `
  --json $JsonPath `
  --material-manifest $MaterialManifestPath `
  --out $OutDir `
  --width 2048 `
  --height 1024 `
  --samples 16
