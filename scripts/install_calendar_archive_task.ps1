$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $RepoRoot ".venv-live\Scripts\python.exe"
$Verifier = Join-Path $RepoRoot "scripts\verify_calendar.py"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Missing $Python"
}

$Action = New-ScheduledTaskAction -Execute $Python -Argument "`"$Verifier`" --archive" -WorkingDirectory $RepoRoot
$Trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At 18:00
$Settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Minutes 10) -StartWhenAvailable
Register-ScheduledTask -TaskName "JarvisCalendarArchive" -Action $Action -Trigger $Trigger -Settings $Settings -Description "Archive economic-calendar events weekly for historical replay." -Force
Write-Host "Installed JarvisCalendarArchive (Sunday 18:00 local time)."
