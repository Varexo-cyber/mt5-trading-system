param(
    [string]$Mt5Path = "C:\Program Files\MetaTrader 5\terminal64.exe",
    [ValidateRange(15, 600)]
    [int]$PaperDelaySeconds = 60
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Pythonw = Join-Path $RepoRoot ".venv-live\Scripts\pythonw.exe"
$Jarvis = Join-Path $RepoRoot "jarvis.py"
$Mt5Launcher = Join-Path $PSScriptRoot "launch_mt5_minimized.ps1"
$WindowsPowerShell = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"

foreach ($RequiredFile in @($Pythonw, $Jarvis, $Mt5Launcher, $Mt5Path, $WindowsPowerShell)) {
    if (-not (Test-Path -LiteralPath $RequiredFile)) {
        throw "Missing required file: $RequiredFile"
    }
}

$LoginTrigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$Mt5Action = New-ScheduledTaskAction `
    -Execute $WindowsPowerShell `
    -Argument "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$Mt5Launcher`" -Mt5Path `"$Mt5Path`"" `
    -WorkingDirectory (Split-Path -Parent $Mt5Path)
$Mt5Settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 5) `
    -MultipleInstances IgnoreNew `
    -StartWhenAvailable `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries

Register-ScheduledTask `
    -TaskName "JarvisMetaTrader5" `
    -Action $Mt5Action `
    -Trigger $LoginTrigger `
    -Settings $Mt5Settings `
    -Description "Start the saved Eightcap MetaTrader 5 terminal minimized at user login." `
    -Force | Out-Null

$PaperTrigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$PaperTrigger.Delay = "PT${PaperDelaySeconds}S"
$PaperAction = New-ScheduledTaskAction `
    -Execute $Pythonw `
    -Argument "`"$Jarvis`" --operation paper" `
    -WorkingDirectory $RepoRoot
$PaperSettings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Days 3650) `
    -RestartCount 30 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -MultipleInstances IgnoreNew `
    -StartWhenAvailable `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries

Register-ScheduledTask `
    -TaskName "JarvisTradingPaper" `
    -Action $PaperAction `
    -Trigger $PaperTrigger `
    -Settings $PaperSettings `
    -Description "Start Jarvis PAPER trading after MT5; this task cannot submit live orders." `
    -Force | Out-Null

$OldMonitor = Get-ScheduledTask -TaskName "JarvisTradingMonitor" -ErrorAction SilentlyContinue
if ($OldMonitor) {
    Disable-ScheduledTask -TaskName "JarvisTradingMonitor" | Out-Null
}

Write-Host "Installed JarvisMetaTrader5 at login."
Write-Host "Installed JarvisTradingPaper with a $PaperDelaySeconds-second delay."
Write-Host "Disabled JarvisTradingMonitor to prevent duplicate runners."
