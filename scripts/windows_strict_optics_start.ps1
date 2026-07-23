param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [string]$TaskName = "EDOFStrictOpticsConvergence"
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot
$Worker = Join-Path $ProjectRoot "scripts\windows_strict_optics_worker.ps1"
if (-not (Test-Path $Worker)) {
    throw "Strict optics convergence worker is missing: $Worker"
}

$PowerShell = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"
$ActionArguments = "-NoProfile -NonInteractive -ExecutionPolicy Bypass -File `"$Worker`" -ProjectRoot `"$ProjectRoot`""
$Action = New-ScheduledTaskAction -Execute $PowerShell -Argument $ActionArguments -WorkingDirectory $ProjectRoot
$Trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(5)
$Principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Days 14) `
    -RestartCount 1 `
    -RestartInterval (New-TimeSpan -Minutes 1)

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Principal $Principal `
    -Settings $Settings `
    -Force | Out-Null
Start-ScheduledTask -TaskName $TaskName
Start-Sleep -Seconds 2
$Task = Get-ScheduledTask -TaskName $TaskName
$TaskInfo = Get-ScheduledTaskInfo -TaskName $TaskName
@{
    task_name = $TaskName
    state = $Task.State.ToString()
    last_result = $TaskInfo.LastTaskResult
    state_file = (Join-Path $ProjectRoot "workspace\edof_reproduction\windows_strict_optics_convergence\state.json")
    stdout = (Join-Path $ProjectRoot "workspace\edof_reproduction\windows_strict_optics_convergence\convergence.stdout.log")
    stderr = (Join-Path $ProjectRoot "workspace\edof_reproduction\windows_strict_optics_convergence\convergence.stderr.log")
} | ConvertTo-Json -Compress | Write-Output
