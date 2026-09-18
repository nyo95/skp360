$ErrorActionPreference = "Stop"

$ProjectRoot = "D:\Projects\skpto360"
$PythonExe = "C:\Users\berka\AppData\Local\Programs\Python\Python314\python.exe"
$ScriptPath = Join-Path $ProjectRoot "scripts\run_pipeline.py"

if (!(Test-Path -LiteralPath $PythonExe)) { throw "Python executable not found: $PythonExe" }
if (!(Test-Path -LiteralPath $ScriptPath)) { throw "Pipeline script not found: $ScriptPath" }

& $PythonExe $ScriptPath @args
exit $LASTEXITCODE