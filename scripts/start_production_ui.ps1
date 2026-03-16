$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $pythonPath)) {
    $pythonCmd = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($null -eq $pythonCmd) {
        Write-Host "No se encontro Python en .venv ni en PATH. Verifica el entorno virtual." -ForegroundColor Red
        exit 1
    }
    $pythonPath = $pythonCmd.Source
}

$mainCmd = "Set-Location '$projectRoot'; `$env:PAPER_TRADING='false'; & '$pythonPath' -m src.main"
# Ejecutar Streamlit en modo headless para evitar que abra automáticamente
# el navegador y así prevenir que se abran dos ventanas (Streamlit + Start-Process).
$uiCmd = "Set-Location '$projectRoot'; & '$pythonPath' -m streamlit run src/ui_app.py --server.headless true --server.port 8501"

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

# Abrir UI en navegador por defecto para confirmar inicio visual.
Start-Sleep -Seconds 2
Start-Process "http://localhost:8501"

Write-Host "Aplicacion iniciada: bot LIVE + IU Streamlit." -ForegroundColor Green
