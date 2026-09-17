$ErrorActionPreference = "Stop"

$ProjectRoot = "D:\Projects\skpto360"
$BlenderExe = "C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"
$ObjPath = Join-Path $ProjectRoot "output\obj\scene.obj"
$JsonPath = Join-Path $ProjectRoot "output\obj\scene.json"
$Phase0DReportPath = Join-Path $ProjectRoot "output\phase0d\phase0d_report.json"
$MaterialManifestPath = Join-Path $ProjectRoot "output\phase0e\input\material_manifest.json"
$OutDir = Join-Path $ProjectRoot "output\phase0g"
$ScriptPath = Join-Path $ProjectRoot "scripts\phase0g_source_signal_validation.py"

if (!(Test-Path -LiteralPath $BlenderExe)) { throw "Blender executable not found: $BlenderExe" }
if (!(Test-Path -LiteralPath $ObjPath)) { throw "OBJ file not found: $ObjPath" }
if (!(Test-Path -LiteralPath $JsonPath)) { throw "scene.json not found: $JsonPath" }
if (!(Test-Path -LiteralPath $Phase0DReportPath)) { throw "Phase 0D report not found: $Phase0DReportPath" }
if (!(Test-Path -LiteralPath $MaterialManifestPath)) { throw "Phase 0E material manifest not found: $MaterialManifestPath" }

New-Item -ItemType Directory -Path $OutDir -Force | Out-Null

& $BlenderExe --background --python $ScriptPath -- `
  --obj $ObjPath `
  --json $JsonPath `
  --phase0d-report $Phase0DReportPath `
  --material-manifest $MaterialManifestPath `
  --out $OutDir `
  --width 2048 `
  --height 1024 `
  --samples 32
