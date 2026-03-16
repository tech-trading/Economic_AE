param(
    [string]$ShortcutName = "Economic AE",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$startupScript = Join-Path $projectRoot "scripts\start_production_ui.ps1"
$desktopPath = [Environment]::GetFolderPath("Desktop")
$shortcutPath = Join-Path $desktopPath ("{0}.lnk" -f $ShortcutName)
$iconDir = Join-Path $projectRoot "assets"
$iconPath = Join-Path $iconDir "app_icon.ico"
$streamlitFavicon = Join-Path $projectRoot ".venv311\Lib\site-packages\streamlit\static\favicon.png"

if (-not (Test-Path $startupScript)) {
    Write-Host "No se encontro el script de inicio: $startupScript" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $iconDir)) {
    New-Item -Path $iconDir -ItemType Directory | Out-Null
}

if (-not (Test-Path $iconPath) -or $Force) {
    if (Test-Path $streamlitFavicon) {
        Add-Type -AssemblyName System.Drawing

        $bitmap = [System.Drawing.Bitmap]::FromFile($streamlitFavicon)
        $icon = [System.Drawing.Icon]::FromHandle($bitmap.GetHicon())
        $stream = [System.IO.File]::Open($iconPath, [System.IO.FileMode]::Create)

        try {
            $icon.Save($stream)
        }
        finally {
            $stream.Dispose()
            $icon.Dispose()
            $bitmap.Dispose()
        }

        Write-Host "Icono generado en: $iconPath" -ForegroundColor Cyan
    }
}

$iconForShortcut = if (Test-Path $iconPath) { $iconPath } else { (Get-Command powershell.exe).Source }

$wShell = New-Object -ComObject WScript.Shell
$shortcut = $wShell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = (Get-Command powershell.exe).Source
$shortcut.Arguments = "-ExecutionPolicy Bypass -File `"$startupScript`""
$shortcut.WorkingDirectory = $projectRoot
$shortcut.IconLocation = "$iconForShortcut,0"
$shortcut.Description = "Economic AE - Bot LIVE + UI"
$shortcut.Save()

Write-Host "Acceso directo creado en escritorio: $shortcutPath" -ForegroundColor Green
