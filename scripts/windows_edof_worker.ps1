param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot
$Python = Join-Path $ProjectRoot ".venv-edof\Scripts\python.exe"
$Output = Join-Path $ProjectRoot "workspace\edof_reproduction\windows_full_actual"
$Checkpoint = Join-Path $Output "checkpoints\latest.pt"
$Stdout = Join-Path $Output "windows_train.stdout.log"
$Stderr = Join-Path $Output "windows_train.stderr.log"
$ExitStatus = Join-Path $Output "windows_train.exit.json"
New-Item -ItemType Directory -Force -Path $Output | Out-Null

$Arguments = @(
    "-u", "-m", "edof_reproduction",
    "--config", "configs\edof_reproduction\windows_full.yaml",
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
