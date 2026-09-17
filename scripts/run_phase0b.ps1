$ErrorActionPreference = "Stop"

$ProjectRoot = "D:\Projects\skpto360"
$BlenderExe = "C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"
$DaePath = Join-Path $ProjectRoot "output\dae\scene.dae"
$JsonPath = Join-Path $ProjectRoot "output\dae\scene.json"
$OutDir = Join-Path $ProjectRoot "output\phase0b"
$ScriptPath = Join-Path $ProjectRoot "scripts\phase0b_import_verify.py"

if (!(Test-Path -LiteralPath $BlenderExe)) {
  throw "Blender executable not found: $BlenderExe"
}

if (!(Test-Path -LiteralPath $DaePath)) {
  throw "DAE file not found: $DaePath"
}

if (!(Test-Path -LiteralPath $JsonPath)) {
  throw "JSON file not found: $JsonPath"
}

New-Item -ItemType Directory -Path $OutDir -Force | Out-Null

& $BlenderExe --background --python $ScriptPath -- `
  --dae $DaePath `
  --json $JsonPath `
  --out $OutDir `
  --render-preview
