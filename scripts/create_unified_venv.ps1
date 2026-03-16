<#
Crear un entorno virtual unificado `.venv` e instalar dependencias.

Uso (PowerShell):
  ./scripts/create_unified_venv.ps1

El script:
 - crea `.venv` en la raíz del repo si no existe
 - activa el entorno y actualiza pip
 - instala `requirements.txt`
 - instala TensorFlow y protobuf compatibles (intento seguro)
 - si hay conflictos, imprime instrucciones para resolver manualmente
#>
Param()

Set-StrictMode -Version Latest
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Push-Location $RepoRoot

$venvPath = Join-Path $RepoRoot '.venv'
if (-Not (Test-Path $venvPath)) {
    Write-Host "Creando entorno virtual en .venv..."
    python -m venv .venv
} else {
    Write-Host ".venv ya existe — se reutilizará."
}

Write-Host "Activando .venv y actualizando pip..."
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel

Write-Host "Instalando dependencias desde requirements.txt..."
if (Test-Path requirements.txt) {
    pip install -r requirements.txt
} else {
    Write-Host "No se encontró requirements.txt en la raíz."
}

# Intentar instalar TensorFlow y protobuf en versiones compatibles
Write-Host "Instalando TensorFlow y protobuf (intento)..."
try {
    pip install "tensorflow==2.11.0" "protobuf==3.20.3"
} catch {
    Write-Host "Instalación automática de TensorFlow falló. Revisa las versiones de numpy/protobuf y ajusta manualmente."
}

Write-Host "Entorno unificado preparado (verifica errores arriba)." 
Write-Host "Para limpiar entornos viejos (no ejecutado automáticamente), por ejemplo:"
Write-Host "  Remove-Item -Recurse -Force .venv-tf"
Write-Host "  Remove-Item -Recurse -Force .venv311"

Pop-Location
