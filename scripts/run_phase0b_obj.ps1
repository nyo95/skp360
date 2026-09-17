$ErrorActionPreference = "Stop"

$ProjectRoot = "D:\Projects\skpto360"
$BlenderExe = "C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"
$ObjPath = Join-Path $ProjectRoot "output\obj\scene.obj"
$JsonPath = Join-Path $ProjectRoot "output\obj\scene.json"
$OutDir = Join-Path $ProjectRoot "output\phase0b_obj"
$ScriptPath = Join-Path $ProjectRoot "scripts\phase0b_obj_import_verify.py"
$DaeReportPath = Join-Path $ProjectRoot "output\phase0b\phase0b_report.json"
$FbxReportPath = Join-Path $ProjectRoot "output\phase0b_fbx\phase0b_fbx_report.json"

if (!(Test-Path -LiteralPath $BlenderExe)) {
  throw "Blender executable not found: $BlenderExe"
}

if (!(Test-Path -LiteralPath $ObjPath)) {
  throw "OBJ file not found: $ObjPath"
}

if (!(Test-Path -LiteralPath $JsonPath)) {
  throw "JSON file not found: $JsonPath"
}

New-Item -ItemType Directory -Path $OutDir -Force | Out-Null

$argsList = @(
  "--background",
  "--python", $ScriptPath,
  "--",
  "--obj", $ObjPath,
  "--json", $JsonPath,
  "--out", $OutDir,
  "--render-preview"
)

if (Test-Path -LiteralPath $DaeReportPath) {
  $argsList += @("--dae-report", $DaeReportPath)
}

if (Test-Path -LiteralPath $FbxReportPath) {
  $argsList += @("--fbx-report", $FbxReportPath)
}

& $BlenderExe @argsList
