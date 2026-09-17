$ErrorActionPreference = "Stop"

$ProjectRoot = "D:\Projects\skpto360"
$Phase0GReportPath = Join-Path $ProjectRoot "output\phase0g\phase0g_report.json"
$OutDir = Join-Path $ProjectRoot "output\phase0g\ai_ab"
$ScriptPath = Join-Path $ProjectRoot "scripts\phase0g_flux_ab.py"

if (!(Test-Path -LiteralPath $Phase0GReportPath)) { throw "Phase 0G report not found: $Phase0GReportPath" }

New-Item -ItemType Directory -Path $OutDir -Force | Out-Null

python $ScriptPath `
  --phase0g-report $Phase0GReportPath `
  --out $OutDir
