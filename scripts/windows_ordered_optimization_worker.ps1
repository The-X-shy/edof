param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot
$env:TORCH_HOME = Join-Path $ProjectRoot "torch-cache"
$env:PYTHONUTF8 = "1"
$env:PYTHONPATH = $ProjectRoot
$Python = Join-Path $ProjectRoot ".venv-edof\Scripts\python.exe"
$Runner = Join-Path $ProjectRoot "scripts\run_ordered_optimization_sequence.py"
$Output = Join-Path $ProjectRoot "workspace\edof_reproduction\windows_ordered_optimization"
$Stdout = Join-Path $Output "ordered_optimization.stdout.log"
$Stderr = Join-Path $Output "ordered_optimization.stderr.log"
$ExitStatus = Join-Path $Output "ordered_optimization.exit.json"
New-Item -ItemType Directory -Force -Path $Output | Out-Null

if (-not (Test-Path $Python)) {
    throw "EDOF environment is missing: $Python"
}
if (-not (Test-Path $Runner)) {
    throw "Ordered optimization runner is missing: $Runner"
}

$StartedAt = Get-Date
$Arguments = @("-u", $Runner, "--project-root", $ProjectRoot)
$Process = Start-Process `
    -FilePath $Python `
    -ArgumentList $Arguments `
    -WorkingDirectory $ProjectRoot `
    -RedirectStandardOutput $Stdout `
    -RedirectStandardError $Stderr `
    -Wait `
    -PassThru
@{
    exit_code = $Process.ExitCode
    started_at = $StartedAt.ToString("o")
    finished_at = (Get-Date).ToString("o")
} | ConvertTo-Json -Compress | Set-Content -Path $ExitStatus -Encoding UTF8
exit $Process.ExitCode
