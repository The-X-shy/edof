param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot
$Python = Join-Path $ProjectRoot ".venv-edof\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "EDOF environment is missing. Run scripts\windows_edof_bootstrap.ps1 first."
}

$Output = Join-Path $ProjectRoot "workspace\edof_reproduction\windows_full_actual"
$Checkpoint = Join-Path $Output "checkpoints\latest.pt"
$Stdout = Join-Path $Output "windows_train.stdout.log"
$Stderr = Join-Path $Output "windows_train.stderr.log"
New-Item -ItemType Directory -Force -Path $Output | Out-Null

$Arguments = @(
    "-u", "-m", "edof_reproduction",
    "--config", "configs\edof_reproduction\windows_full.yaml",
    "--output", $Output
)
if (Test-Path $Checkpoint) {
    $Arguments += @("--resume", $Checkpoint)
}

$Process = Start-Process -FilePath $Python -ArgumentList $Arguments -WorkingDirectory $ProjectRoot -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -PassThru
@{
    pid = $Process.Id
    output = $Output
    stdout = $Stdout
    stderr = $Stderr
    resumed = (Test-Path $Checkpoint)
} | ConvertTo-Json -Compress | Write-Output
