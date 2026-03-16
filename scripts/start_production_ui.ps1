$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot ".venv311\Scripts\python.exe"

if (-not (Test-Path $pythonPath)) {
    Write-Host "No se encontro Python en .venv311. Verifica el entorno virtual." -ForegroundColor Red
    exit 1
}

$mainCmd = "Set-Location '$projectRoot'; `$env:PAPER_TRADING='false'; & '$pythonPath' -m src.main"
$uiCmd = "Set-Location '$projectRoot'; & '$pythonPath' -m streamlit run src/ui_app.py"

# Bot en modo LIVE (produccion real)
Start-Process -FilePath "powershell.exe" -ArgumentList @(
    "-NoExit",
    "-ExecutionPolicy", "Bypass",
    "-Command", $mainCmd
)

Start-Sleep -Seconds 2

# Interfaz Streamlit
Start-Process -FilePath "powershell.exe" -ArgumentList @(
    "-NoExit",
    "-ExecutionPolicy", "Bypass",
    "-Command", $uiCmd
)

Write-Host "Aplicacion iniciada: bot LIVE + IU Streamlit." -ForegroundColor Green
