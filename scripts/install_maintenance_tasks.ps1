# Two scheduled tasks: watch the system, and update it safely when it is idle.
#
# Neither of these trades. The self-check reads and reports; the updater only
# ever runs while the account is flat, outside session hours, and rolls back if
# the new code does not pass on this machine. Both are safe to remove at any
# time with Unregister-ScheduledTask.

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $RepoRoot ".venv-live\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Missing $Python. Run update.cmd once first."
}

# -- 1. self-check, every fifteen minutes ------------------------------------
#
# --quiet so a healthy run prints nothing and the task history stays readable;
# --alert so a problem reaches the operator rather than a log file nobody
# opens. The exit code is what Task Scheduler records, and a task sitting on
# "Last Run Result: 0x1" for two days is visible at a glance.
$CheckScript = Join-Path $RepoRoot "scripts\selfcheck.py"
$CheckAction = New-ScheduledTaskAction -Execute $Python -Argument "`"$CheckScript`" --quiet --alert" -WorkingDirectory $RepoRoot
$CheckTrigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 15)
$CheckSettings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Minutes 5) -StartWhenAvailable -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName "JarvisSelfCheck" -Action $CheckAction -Trigger $CheckTrigger -Settings $CheckSettings -Description "Report whether every layer of the trading system is alive. Reads only." -Force
Write-Host "Installed JarvisSelfCheck (every 15 minutes)."

# -- 2. auto-update, Sunday morning ------------------------------------------
#
# Sunday 08:00 local: FX has been closed since Friday evening, the account is
# flat because the evening flatten closed everything, and there is a full day
# before Monday's open to notice a rollback. The script re-checks all of that
# itself -- the schedule is convenience, the guards are the safety.
$UpdateScript = Join-Path $RepoRoot "scripts\auto_update.py"
$UpdateAction = New-ScheduledTaskAction -Execute $Python -Argument "`"$UpdateScript`"" -WorkingDirectory $RepoRoot
$UpdateTrigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At 08:00
$UpdateSettings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Minutes 30) -StartWhenAvailable
Register-ScheduledTask -TaskName "JarvisAutoUpdate" -Action $UpdateAction -Trigger $UpdateTrigger -Settings $UpdateSettings -Description "Pull, verify on this machine, and roll back if the new code fails. Refuses while a position is open." -Force
Write-Host "Installed JarvisAutoUpdate (Sunday 08:00 local time)."

Write-Host ""
Write-Host "  Both installed. To see them:      Get-ScheduledTask Jarvis*"
Write-Host "  To run the check right now:       Start-ScheduledTask JarvisSelfCheck"
Write-Host "  To remove either:                 Unregister-ScheduledTask -TaskName JarvisSelfCheck"
Write-Host ""
Write-Host "  The updater never runs while a position is open, never runs with a"
Write-Host "  STOP file present, and puts the old commit back if the suite fails."
Write-Host ""
