$ErrorActionPreference = "Stop"

$ProjectRoot = "D:\Projects\skpto360"
$ScriptPath = Join-Path $ProjectRoot "scripts\controlled_core_harness.py"

Set-Location $ProjectRoot
python $ScriptPath build-package
