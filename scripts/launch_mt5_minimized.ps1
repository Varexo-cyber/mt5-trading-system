param(
    [Parameter(Mandatory = $true)]
    [string]$Mt5Path
)

$ErrorActionPreference = "Stop"
$ResolvedMt5 = (Resolve-Path -LiteralPath $Mt5Path).Path
$Running = Get-Process -Name "terminal64" -ErrorAction SilentlyContinue | Where-Object {
    try {
        $_.Path -and ((Resolve-Path -LiteralPath $_.Path).Path -eq $ResolvedMt5)
    }
    catch {
        $false
    }
}

if ($Running) {
    exit 0
}

Start-Process `
    -FilePath $ResolvedMt5 `
    -WorkingDirectory (Split-Path -Parent $ResolvedMt5) `
    -WindowStyle Minimized
