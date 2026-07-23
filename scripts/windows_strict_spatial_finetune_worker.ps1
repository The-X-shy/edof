param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot
$env:TORCH_HOME = Join-Path $ProjectRoot "torch-cache"
$env:PYTHONUTF8 = "1"
$env:PYTHONPATH = $ProjectRoot
$Python = Join-Path $ProjectRoot ".venv-edof\Scripts\python.exe"
$Runner = Join-Path $ProjectRoot "scripts\run_strict_spatial_finetune_from_convergence.py"
$ConvergenceOutput = Join-Path $ProjectRoot "workspace\edof_reproduction\windows_strict_optics_convergence"
$ConvergenceState = Join-Path $ConvergenceOutput "state.json"
$ConvergenceSummary = Join-Path $ConvergenceOutput "summary.json"
$Output = Join-Path $ProjectRoot "workspace\edof_reproduction\windows_strict_spatial_finetune"
$Stdout = Join-Path $Output "strict_spatial_finetune.stdout.log"
$Stderr = Join-Path $Output "strict_spatial_finetune.stderr.log"
$ExitStatus = Join-Path $Output "strict_spatial_finetune.exit.json"
New-Item -ItemType Directory -Force -Path $Output | Out-Null

if (-not (Test-Path $Python)) {
    throw "EDOF environment is missing: $Python"
}
if (-not (Test-Path $Runner)) {
    throw "Strict spatial fine-tune runner is missing: $Runner"
}

while (-not (Test-Path $ConvergenceSummary)) {
    if (Test-Path $ConvergenceState) {
        $State = Get-Content -Raw -Path $ConvergenceState | ConvertFrom-Json
        if ($State.status -eq "failed") {
            throw "Strict optical convergence failed; fine-tuning was not started."
        }
    }
    Start-Sleep -Seconds 60
}

$Convergence = Get-Content -Raw -Path $ConvergenceSummary | ConvertFrom-Json
if ($Convergence.status -ne "completed") {
    throw "Strict optical convergence did not complete; fine-tuning was not started."
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
