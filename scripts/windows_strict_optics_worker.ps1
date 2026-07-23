param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot
$env:TORCH_HOME = Join-Path $ProjectRoot "torch-cache"
$env:PYTHONUTF8 = "1"
$Python = Join-Path $ProjectRoot ".venv-edof\Scripts\python.exe"
$Runner = Join-Path $ProjectRoot "scripts\run_strict_optics_convergence.py"
$EvaluationOutput = Join-Path $ProjectRoot "workspace\edof_reproduction\windows_strict_full_fov_evaluation"
$EvaluationState = Join-Path $EvaluationOutput "state.json"
$EvaluationSummary = Join-Path $EvaluationOutput "summary.json"
$Output = Join-Path $ProjectRoot "workspace\edof_reproduction\windows_strict_optics_convergence"
$Stdout = Join-Path $Output "convergence.stdout.log"
$Stderr = Join-Path $Output "convergence.stderr.log"
$ExitStatus = Join-Path $Output "convergence.exit.json"
New-Item -ItemType Directory -Force -Path $Output | Out-Null

if (-not (Test-Path $Python)) {
    throw "EDOF environment is missing: $Python"
}
if (-not (Test-Path $Runner)) {
    throw "Strict optics convergence runner is missing: $Runner"
}

while (-not (Test-Path $EvaluationSummary)) {
    if (Test-Path $EvaluationState) {
        $State = Get-Content -Raw -Path $EvaluationState | ConvertFrom-Json
        $FailedSteps = @(
            $State.steps.PSObject.Properties |
                ForEach-Object { $_.Value } |
                Where-Object { $_.status -eq "failed" }
        )
        if ($State.status -like "failed*" -or $FailedSteps.Count -gt 0) {
            throw "Strict full-field evaluation failed; optical convergence was not started."
        }
    }
    Start-Sleep -Seconds 60
}

$Evaluation = Get-Content -Raw -Path $EvaluationSummary | ConvertFrom-Json
if ($Evaluation.status -ne "completed") {
    throw "Strict full-field repeatability did not pass; optical convergence was not started."
}

$StartedAt = Get-Date
$Arguments = @("-u", $Runner, "--project-root", $ProjectRoot, "--max-images", "10")
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
