$ErrorActionPreference = "Stop"

$ProjectRoot = "D:\Projects\skpto360"
$Phase0EReportPath = Join-Path $ProjectRoot "output\phase0e\phase0e_report.json"
$OutDir = Join-Path $ProjectRoot "output\phase0f"
$ScriptPath = Join-Path $ProjectRoot "scripts\phase0f_cloudflare_flux_experiment_a.py"

if (!(Test-Path -LiteralPath $Phase0EReportPath)) {
  throw "Phase 0E report not found: $Phase0EReportPath"
}

New-Item -ItemType Directory -Path $OutDir -Force | Out-Null

python $ScriptPath `
  --project-root $ProjectRoot `
  --phase0e-report $Phase0EReportPath `
  --out $OutDir
