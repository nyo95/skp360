$ErrorActionPreference = "Stop"

$ProjectRoot = "D:\Projects\skpto360"
$BlenderExe = "C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"
$ObjPath = Join-Path $ProjectRoot "output\obj\scene.obj"
$JsonPath = Join-Path $ProjectRoot "output\obj\scene.json"
$OutDir = Join-Path $ProjectRoot "output\phase0c"
$ScriptPath = Join-Path $ProjectRoot "scripts\phase0c_cubemap_passes.py"

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

& $BlenderExe --background --python $ScriptPath -- `
  --obj $ObjPath `
  --json $JsonPath `
  --out $OutDir `
  --resolution 1024
