param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [string]$TaskName = "EDOFRecommendedSequence"
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot
$Worker = Join-Path $ProjectRoot "scripts\windows_edof_sequence_worker.ps1"
if (-not (Test-Path $Worker)) {
    throw "Recommended sequence worker is missing: $Worker"
}

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
    sequence_state = (Join-Path $ProjectRoot "workspace\edof_reproduction\windows_recommended_sequence\sequence_state.json")
    stdout = (Join-Path $ProjectRoot "workspace\edof_reproduction\windows_recommended_sequence\sequence.stdout.log")
    stderr = (Join-Path $ProjectRoot "workspace\edof_reproduction\windows_recommended_sequence\sequence.stderr.log")
} | ConvertTo-Json -Compress | Write-Output
