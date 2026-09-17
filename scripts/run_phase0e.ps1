$ErrorActionPreference = "Stop"

$ProjectRoot = "D:\Projects\skpto360"
$BlenderExe = "C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"
$ObjPath = Join-Path $ProjectRoot "output\obj\scene.obj"
$Phase0DReportPath = Join-Path $ProjectRoot "output\phase0d\phase0d_report.json"
$OutDir = Join-Path $ProjectRoot "output\phase0e"
$ScriptPath = Join-Path $ProjectRoot "scripts\phase0e_prepare_ai_feasibility.py"

if (!(Test-Path -LiteralPath $BlenderExe)) {
  throw "Blender executable not found: $BlenderExe"
}

if (!(Test-Path -LiteralPath $ObjPath)) {
  throw "OBJ file not found: $ObjPath"
}

if (!(Test-Path -LiteralPath $Phase0DReportPath)) {
  throw "Phase 0D report not found: $Phase0DReportPath"
}

New-Item -ItemType Directory -Path $OutDir -Force | Out-Null

& $BlenderExe --background --python $ScriptPath -- `
  --project-root $ProjectRoot `
  --obj $ObjPath `
  --phase0d-report $Phase0DReportPath `
  --out $OutDir `
  --depth-near 0.2 `
  --depth-far 20.0
