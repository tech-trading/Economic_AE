$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Resolve-Path (Join-Path $scriptDir "..")

Set-Location $projectRoot

$pythonExe = Join-Path $projectRoot ".venv311\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    throw "No se encontro el entorno .venv311 en $pythonExe"
}

$logDir = Join-Path $projectRoot "logs"
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir | Out-Null
}

$logPath = Join-Path $logDir "daily_snapshot.log"
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Add-Content -Path $logPath -Value "[$timestamp] Iniciando snapshot diario"

& $pythonExe -c "from src.daily_jobs import run_daily_snapshot; run_daily_snapshot()" *>> $logPath
if ($LASTEXITCODE -ne 0) {
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Add-Content -Path $logPath -Value "[$timestamp] Snapshot diario finalizo con error codigo $LASTEXITCODE"
    exit $LASTEXITCODE
}

$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
Add-Content -Path $logPath -Value "[$timestamp] Snapshot diario finalizado correctamente"
