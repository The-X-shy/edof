param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [string]$TaskName = "EDOFFullTraining"
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot
$Python = Join-Path $ProjectRoot ".venv-edof\Scripts\python.exe"
$Worker = Join-Path $ProjectRoot "scripts\windows_edof_worker.ps1"
if (-not (Test-Path $Python)) {
    throw "EDOF environment is missing. Run scripts\windows_edof_bootstrap.ps1 first."
}
if (-not (Test-Path $Worker)) {
    throw "Training worker is missing: $Worker"
}

$Output = Join-Path $ProjectRoot "workspace\edof_reproduction\windows_full_actual"
$Checkpoint = Join-Path $Output "checkpoints\latest.pt"
New-Item -ItemType Directory -Force -Path $Output | Out-Null

$PowerShell = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"
$ActionArguments = "-NoProfile -NonInteractive -ExecutionPolicy Bypass -File `"$Worker`" -ProjectRoot `"$ProjectRoot`""
$Action = New-ScheduledTaskAction -Execute $PowerShell -Argument $ActionArguments -WorkingDirectory $ProjectRoot
$Trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(5)
$Principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Days 30) `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1)

Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Principal $Principal -Settings $Settings -Force | Out-Null
Start-ScheduledTask -TaskName $TaskName
Start-Sleep -Seconds 2
$Task = Get-ScheduledTask -TaskName $TaskName
$TaskInfo = Get-ScheduledTaskInfo -TaskName $TaskName
@{
    task_name = $TaskName
    state = $Task.State.ToString()
    last_result = $TaskInfo.LastTaskResult
    output = $Output
    stdout = (Join-Path $Output "windows_train.stdout.log")
    stderr = (Join-Path $Output "windows_train.stderr.log")
    resumed = (Test-Path $Checkpoint)
} | ConvertTo-Json -Compress | Write-Output
