$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $RepoRoot ".venv-live\Scripts\python.exe"
$Jarvis = Join-Path $RepoRoot "jarvis.py"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Missing $Python"
}

$Action = New-ScheduledTaskAction -Execute $Python -Argument "`"$Jarvis`" --operation monitor" -WorkingDirectory $RepoRoot
$Trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$Settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Days 3650) -RestartCount 10 -RestartInterval (New-TimeSpan -Minutes 1)
Register-ScheduledTask -TaskName "JarvisTradingMonitor" -Action $Action -Trigger $Trigger -Settings $Settings -Description "Read-only MT5 catalogue monitor; live execution is not armed." -Force
Write-Host "Installed JarvisTradingMonitor in MONITOR mode."
