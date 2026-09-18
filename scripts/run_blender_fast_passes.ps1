$ErrorActionPreference = "Stop"

$ProjectRoot = "D:\Projects\skpto360"
$BlenderExe = "C:\Program Files\Blender Foundation\Blender 5.1\blender.exe"
$PythonExe = "C:\Users\berka\AppData\Local\Programs\Python\Python314\python.exe"
$ObjPath = Join-Path $ProjectRoot "output\obj\scene.obj"
$JsonPath = Join-Path $ProjectRoot "output\obj\scene.json"
$OutDir = Join-Path $ProjectRoot "output\fast_passes"
$Worker = Join-Path $ProjectRoot "scripts\blender_passes.py"
$Assembler = Join-Path $ProjectRoot "scripts\assemble_blender_passes.py"

foreach ($Path in @($BlenderExe, $PythonExe, $ObjPath, $JsonPath)) {
  if (!(Test-Path -LiteralPath $Path)) {
    throw "Required path not found: $Path"
  }
}

New-Item -ItemType Directory -Path $OutDir -Force | Out-Null

& $BlenderExe --background --python $Worker -- `
  --obj $ObjPath `
  --json $JsonPath `
  --out $OutDir `
  --resolution 512

& $PythonExe $Assembler `
  --in $OutDir `
  --out $OutDir `
  --width 2048