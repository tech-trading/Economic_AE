$ErrorActionPreference = "Stop"

$mutexName = "Global\EconomicAE.Startup"
$createdNew = $false
$startupMutex = New-Object System.Threading.Mutex($true, $mutexName, [ref]$createdNew)
if (-not $createdNew) {
    Write-Host "Ya hay un inicio en progreso. Espera unos segundos e intenta de nuevo." -ForegroundColor Yellow
    exit 0
}

try {

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

function Get-ProjectPythonProcesses {
    param(
        [Parameter(Mandatory = $true)][string]$Pattern
    )

    Get-CimInstance Win32_Process |
        Where-Object {
            $_.Name -match "python" -and
            $_.CommandLine -match $Pattern
        } |
        Select-Object ProcessId, ParentProcessId, CreationDate, CommandLine
}

function Keep-SingleProcess {
    param(
        [object[]]$Processes,
        [Parameter(Mandatory = $true)][string]$Label
    )

    if (-not $Processes -or $Processes.Count -le 1) {
        return $Processes
    }

    $pidSet = @{}
    foreach ($proc in $Processes) {
        $pidSet[[int]$proc.ProcessId] = $true
    }

    $rootProcesses = @(
        $Processes | Where-Object {
            $parentPid = [int]($_.ParentProcessId)
            -not $pidSet.ContainsKey($parentPid)
        }
    )

    if (-not $rootProcesses -or $rootProcesses.Count -le 1) {
        return $Processes
    }

    $ordered = $rootProcesses | Sort-Object CreationDate -Descending
    $keep = $ordered[0]
    $toStop = $ordered | Select-Object -Skip 1

    foreach ($proc in $toStop) {
        try {
            Stop-Process -Id $proc.ProcessId -Force -ErrorAction Stop
            Write-Host "$Label duplicado eliminado (PID $($proc.ProcessId))." -ForegroundColor Yellow
        }
        catch {
            Write-Host "No se pudo detener $Label duplicado (PID $($proc.ProcessId)): $($_.Exception.Message)" -ForegroundColor Red
        }
    }

    return @($keep)
}

$mainProcs = @(Get-ProjectPythonProcesses -Pattern "-m\s+src\.main")
$mainProcs = @(Keep-SingleProcess -Processes $mainProcs -Label "bot LIVE")
$runningMain = if ($mainProcs.Count -gt 0) { $mainProcs[0] } else { $null }

$uiProcs = @(Get-ProjectPythonProcesses -Pattern "-m\s+streamlit\s+run\s+src/ui_app\.py")
$uiProcs = @(Keep-SingleProcess -Processes $uiProcs -Label "UI Streamlit")
$runningUi = if ($uiProcs.Count -gt 0) { $uiProcs[0] } else { $null }

if ($null -eq $runningMain) {
    # Bot en modo LIVE (produccion real)
    Start-Process -FilePath "powershell.exe" -ArgumentList @(
        "-NoExit",
        "-ExecutionPolicy", "Bypass",
        "-Command", $mainCmd
    )
    Write-Host "Bot LIVE iniciado." -ForegroundColor Green
}
else {
    Write-Host "Bot LIVE ya estaba ejecutandose (PID $($runningMain.ProcessId))." -ForegroundColor Cyan
}

Start-Sleep -Seconds 2

if ($null -eq $runningUi) {
    # Interfaz Streamlit
    Start-Process -FilePath "powershell.exe" -ArgumentList @(
        "-NoExit",
        "-ExecutionPolicy", "Bypass",
        "-Command", $uiCmd
    )
    Write-Host "UI Streamlit iniciada." -ForegroundColor Green
}
else {
    Write-Host "UI Streamlit ya estaba ejecutandose (PID $($runningUi.ProcessId))." -ForegroundColor Cyan
}

# Abrir UI en navegador por defecto para confirmar inicio visual.
Start-Sleep -Seconds 2
Start-Process "http://localhost:8501"

Write-Host "Aplicacion iniciada: bot LIVE + IU Streamlit." -ForegroundColor Green
}
finally {
    if ($startupMutex) {
        $startupMutex.ReleaseMutex() | Out-Null
        $startupMutex.Dispose()
    }
}
