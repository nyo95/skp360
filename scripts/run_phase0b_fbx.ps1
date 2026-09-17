$ErrorActionPreference = "Stop"

$ProjectRoot = "D:\Projects\skpto360"
$BlenderExe = "C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"
$FbxPath = Join-Path $ProjectRoot "output\fbx\scene.fbx"
$JsonPath = Join-Path $ProjectRoot "output\fbx\scene.json"
$OutDir = Join-Path $ProjectRoot "output\phase0b_fbx"
$ScriptPath = Join-Path $ProjectRoot "scripts\phase0b_fbx_import_verify.py"
$DaeReportPath = Join-Path $ProjectRoot "output\phase0b\phase0b_report.json"

if (!(Test-Path -LiteralPath $BlenderExe)) {
  throw "Blender executable not found: $BlenderExe"
}

if (!(Test-Path -LiteralPath $FbxPath)) {
  throw "FBX file not found: $FbxPath"
}

if (!(Test-Path -LiteralPath $JsonPath)) {
  throw "JSON file not found: $JsonPath"
}

New-Item -ItemType Directory -Path $OutDir -Force | Out-Null

$argsList = @(
  "--background",
  "--python", $ScriptPath,
  "--",
  "--fbx", $FbxPath,
  "--json", $JsonPath,
  "--out", $OutDir,
  "--render-preview"
)

if (Test-Path -LiteralPath $DaeReportPath) {
  $argsList += @("--dae-report", $DaeReportPath)
}

& $BlenderExe @argsList
