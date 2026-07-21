param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [string]$Config = "configs\edof_reproduction\windows_optimized.yaml",
    [string]$OutputName = "windows_optimized"
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot
$env:TORCH_HOME = Join-Path $ProjectRoot "torch-cache"
$Python = Join-Path $ProjectRoot ".venv-edof\Scripts\python.exe"
$Output = Join-Path $ProjectRoot "workspace\edof_reproduction\$OutputName"
$Checkpoint = Join-Path $Output "checkpoints\latest.pt"
$Stdout = Join-Path $Output "windows_train.stdout.log"
$Stderr = Join-Path $Output "windows_train.stderr.log"
$ExitStatus = Join-Path $Output "windows_train.exit.json"
New-Item -ItemType Directory -Force -Path $Output | Out-Null
$Summary = Join-Path $Output "summary.json"
if (Test-Path $Summary) {
    $Completed = Get-Content $Summary -Raw | ConvertFrom-Json
    if ($Completed.status -eq "completed") {
        Write-Output "WINDOWS_EDOF_ALREADY_COMPLETED"
        exit 0
    }
}

$Arguments = @(
    "-u", "-m", "edof_reproduction",
    "--config", $Config,
    "--output", $Output
)
if (Test-Path $Checkpoint) {
    $Arguments += @("--resume", $Checkpoint)
}

$StartedAt = Get-Date
$Process = Start-Process -FilePath $Python -ArgumentList $Arguments -WorkingDirectory $ProjectRoot -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -Wait -PassThru
@{
    exit_code = $Process.ExitCode
    started_at = $StartedAt.ToString("o")
    finished_at = (Get-Date).ToString("o")
    resumed = (Test-Path $Checkpoint)
} | ConvertTo-Json -Compress | Set-Content -Path $ExitStatus -Encoding UTF8
exit $Process.ExitCode
