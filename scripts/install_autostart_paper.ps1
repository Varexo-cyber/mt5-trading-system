param(
    [string]$Mt5Path = "C:\Program Files\MetaTrader 5\terminal64.exe",
    [ValidateSet("paper", "experimental_live")]
    [string]$TradingOperation = "paper",
    [ValidateRange(15, 600)]
    [int]$PaperDelaySeconds = 60,
    [ValidateRange(30, 900)]
    [int]$DashboardDelaySeconds = 90
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Pythonw = Join-Path $RepoRoot ".venv-live\Scripts\pythonw.exe"
$Jarvis = Join-Path $RepoRoot "jarvis.py"
$DashboardLauncher = Join-Path $PSScriptRoot "launch_dashboard.py"
$Mt5Launcher = Join-Path $PSScriptRoot "launch_mt5_minimized.ps1"
$WindowsPowerShell = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"
$ExperimentalContract = Join-Path $RepoRoot "runtime\EXPERIMENTAL_LIVE.json"

if ($TradingOperation -eq "experimental_live" -and -not (Test-Path -LiteralPath $ExperimentalContract)) {
    throw "Experimental-live contract is missing: $ExperimentalContract"
}

if ($TradingOperation -eq "experimental_live") {
    $TradingTaskName = "JarvisTradingExperimentalLive"
    $OtherTradingTaskName = "JarvisTradingPaper"
    $TradingDescription = "Start Jarvis EXPERIMENTAL LIVE for the bound real account after MT5."
}
else {
    $TradingTaskName = "JarvisTradingPaper"
    $OtherTradingTaskName = "JarvisTradingExperimentalLive"
    $TradingDescription = "Start Jarvis PAPER trading after MT5; this task cannot submit live orders."
}

foreach (
    $RequiredFile in @(
        $Pythonw,
        $Jarvis,
        $DashboardLauncher,
        $Mt5Launcher,
        $Mt5Path,
        $WindowsPowerShell
    )
) {
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
    -Argument "`"$Jarvis`" --operation $TradingOperation" `
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
    -TaskName $TradingTaskName `
    -Action $PaperAction `
    -Trigger $PaperTrigger `
    -Settings $PaperSettings `
    -Description $TradingDescription `
    -Force | Out-Null

$DashboardTrigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$DashboardTrigger.Delay = "PT${DashboardDelaySeconds}S"
$DashboardAction = New-ScheduledTaskAction `
    -Execute $Pythonw `
    -Argument "`"$DashboardLauncher`"" `
    -WorkingDirectory $RepoRoot
$DashboardSettings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Days 3650) `
    -RestartCount 10 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -MultipleInstances IgnoreNew `
    -StartWhenAvailable `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries

Register-ScheduledTask `
    -TaskName "JarvisDashboard" `
    -Action $DashboardAction `
    -Trigger $DashboardTrigger `
    -Settings $DashboardSettings `
    -Description "Start the local Jarvis Streamlit dashboard and open it in the default browser." `
    -Force | Out-Null

$OldMonitor = Get-ScheduledTask -TaskName "JarvisTradingMonitor" -ErrorAction SilentlyContinue
if ($OldMonitor) {
    Disable-ScheduledTask -TaskName "JarvisTradingMonitor" | Out-Null
}

$OtherTradingTask = Get-ScheduledTask -TaskName $OtherTradingTaskName -ErrorAction SilentlyContinue
if ($OtherTradingTask) {
    Stop-ScheduledTask -TaskName $OtherTradingTaskName -ErrorAction SilentlyContinue
    Disable-ScheduledTask -TaskName $OtherTradingTaskName | Out-Null
}

Write-Host "Installed JarvisMetaTrader5 at login."
Write-Host "Installed $TradingTaskName with a $PaperDelaySeconds-second delay."
Write-Host "Installed JarvisDashboard with a $DashboardDelaySeconds-second delay."
Write-Host "Disabled JarvisTradingMonitor to prevent duplicate runners."
Write-Host "Disabled $OtherTradingTaskName to prevent duplicate trading runners."
