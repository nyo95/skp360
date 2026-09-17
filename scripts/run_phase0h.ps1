$ErrorActionPreference = "Stop"

$ProjectRoot = "D:\Projects\skpto360"
$BlenderExe = "C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"
$ObjPath = Join-Path $ProjectRoot "output\obj\scene.obj"
$JsonPath = Join-Path $ProjectRoot "output\obj\scene.json"
$Phase0GReportPath = Join-Path $ProjectRoot "output\phase0g\phase0g_report.json"
$OutDir = Join-Path $ProjectRoot "output\phase0h"
$ScriptPath = Join-Path $ProjectRoot "scripts\phase0h_exterior_context.py"

if (!(Test-Path -LiteralPath $BlenderExe)) { throw "Blender executable not found: $BlenderExe" }
if (!(Test-Path -LiteralPath $ObjPath)) { throw "OBJ file not found: $ObjPath" }
if (!(Test-Path -LiteralPath $JsonPath)) { throw "scene.json not found: $JsonPath" }
if (!(Test-Path -LiteralPath $Phase0GReportPath)) { throw "Phase 0G report not found: $Phase0GReportPath" }

New-Item -ItemType Directory -Path $OutDir -Force | Out-Null

& $BlenderExe --background --python $ScriptPath -- `
  --obj $ObjPath `
  --json $JsonPath `
  --phase0g-report $Phase0GReportPath `
  --out $OutDir `
  --width 2048 `
  --height 1024 `
  --samples 32
