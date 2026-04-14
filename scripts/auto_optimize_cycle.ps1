param(
    [int]$Hours = 24,
    [int]$IntervalMinutes = 60,
    [switch]$Loop
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonPath)) {
    throw "No se encontro Python del proyecto en .venv\Scripts\python.exe"
}

$envPath = Join-Path $projectRoot ".env"
$logDir = Join-Path $projectRoot "logs"
$logPath = Join-Path $logDir "auto_optimize_cycle.log"
$statePath = Join-Path $projectRoot "models\auto_opt_state.json"
$bestEnvPath = Join-Path $projectRoot "models\auto_opt_best.env"

if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
}

function Write-Log {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date).ToString("s"), $Message
    Add-Content -Path $logPath -Value $line
    Write-Host $line
}

function Get-FileHashSafe {
    param([string]$Path)
    if (-not (Test-Path $Path)) { return "" }
    try {
        return (Get-FileHash -Path $Path -Algorithm SHA256).Hash
    }
    catch {
        return ""
    }
}

function Restart-LiveBot {
    Write-Log "Reiniciando bot live por cambios en .env"

    $mainProcs = Get-CimInstance Win32_Process | Where-Object {
        $_.Name -eq 'python.exe' -and $_.CommandLine -like '*-m src.main*'
    }
    foreach ($p in $mainProcs) {
        try {
            Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
        }
        catch {
        }
    }

    $pidPath = Join-Path $projectRoot "logs\live_bot.pid"
    if (Test-Path $pidPath) {
        Remove-Item $pidPath -Force -ErrorAction SilentlyContinue
    }

    Start-Process -FilePath $pythonPath -ArgumentList @('-m', 'src.main') -WorkingDirectory $projectRoot | Out-Null
    Write-Log "Bot live reiniciado"
}

function Get-CurrentMetrics {
    $reportPath = Join-Path $projectRoot "models\daily_live_report.json"
    if (-not (Test-Path $reportPath)) {
        return $null
    }
    try {
        $obj = Get-Content $reportPath -Raw | ConvertFrom-Json
        $mt5 = $obj.mt5_performance
        if ($null -eq $mt5) {
            return $null
        }
        return [PSCustomObject]@{
            closed_deals = [int]($mt5.closed_deals)
            win_rate = [double]($mt5.win_rate)
            profit_factor = [double]($mt5.profit_factor)
            net_profit = [double]($mt5.net_profit)
            max_drawdown_profit = [double]($mt5.max_drawdown_profit)
        }
    }
    catch {
        return $null
    }
}

function Load-State {
    if (-not (Test-Path $statePath)) {
        return [PSCustomObject]@{}
    }
    try {
        return (Get-Content $statePath -Raw | ConvertFrom-Json)
    }
    catch {
        return [PSCustomObject]@{}
    }
}

function Save-State {
    param([object]$State)
    $dir = Split-Path -Parent $statePath
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
    ($State | ConvertTo-Json -Depth 8) | Set-Content -Path $statePath -Encoding UTF8
}

function Ensure-StateShape {
    param([object]$State)

    if ($null -eq $State) {
        $State = [PSCustomObject]@{}
    }

    if (-not ($State.PSObject.Properties.Name -contains 'best')) {
        Add-Member -InputObject $State -MemberType NoteProperty -Name best -Value $null
    }
    if (-not ($State.PSObject.Properties.Name -contains 'last')) {
        Add-Member -InputObject $State -MemberType NoteProperty -Name last -Value $null
    }

    return $State
}

function Snapshot-BestEnv {
    if (Test-Path $envPath) {
        Copy-Item -Path $envPath -Destination $bestEnvPath -Force
    }
}

function Is-Degraded {
    param(
        [object]$Current,
        [object]$Best
    )

    if ($null -eq $Current -or $null -eq $Best) { return $false }
    if ([int]$Current.closed_deals -lt 6 -or [int]$Best.closed_deals -lt 6) { return $false }

    $pfDrop = ([double]$Current.profit_factor -lt ([double]$Best.profit_factor * 0.80))
    $netDrop = ([double]$Current.net_profit -lt ([double]$Best.net_profit - [Math]::Max(2.0, [Math]::Abs([double]$Best.net_profit) * 0.20)))
    $ddRise = ([double]$Current.max_drawdown_profit -gt [Math]::Max(3.0, [double]$Best.max_drawdown_profit * 1.35))

    return ($pfDrop -or $netDrop -or $ddRise)
}

function Is-Better {
    param(
        [object]$Current,
        [object]$Best
    )

    if ($null -eq $Current) { return $false }
    if ($null -eq $Best) { return $true }

    $scoreCurrent = ([double]$Current.profit_factor * 2.0) + ([double]$Current.net_profit * 0.20) + ([double]$Current.win_rate * 2.0) - ([double]$Current.max_drawdown_profit * 0.10)
    $scoreBest = ([double]$Best.profit_factor * 2.0) + ([double]$Best.net_profit * 0.20) + ([double]$Best.win_rate * 2.0) - ([double]$Best.max_drawdown_profit * 0.10)

    if ([int]$Current.closed_deals -lt 4 -and [int]$Best.closed_deals -ge 4) {
        return $false
    }

    return ($scoreCurrent -gt $scoreBest)
}

function Restore-BestEnv {
    if (-not (Test-Path $bestEnvPath)) {
        Write-Log "No hay snapshot best env para rollback"
        return $false
    }
    Copy-Item -Path $bestEnvPath -Destination $envPath -Force
    Write-Log "Rollback aplicado: .env restaurado desde baseline estable"
    return $true
}

function Invoke-OneCycle {
    Set-Location $projectRoot

    Write-Log "Generando reporte diario (pre-check)"
    & $pythonPath "scripts\daily_live_report.py" | Out-Null

    $preMetrics = Get-CurrentMetrics
    $state = Ensure-StateShape -State (Load-State)
    $bestMetrics = if ($null -ne $state.best) { $state.best.metrics } else { $null }

    if ($null -ne $preMetrics -and $null -ne $bestMetrics -and (Is-Degraded -Current $preMetrics -Best $bestMetrics)) {
        Write-Log "Detectada degradacion contra baseline; aplicando rollback"
        if (Restore-BestEnv) {
            Restart-LiveBot
            & $pythonPath "scripts\daily_live_report.py" | Out-Null
            $postRollback = Get-CurrentMetrics
            $state.last = [PSCustomObject]@{ metrics = $postRollback; updated_at = (Get-Date).ToString("s") }
            Save-State -State $state
            return
        }
    }

    $beforeHash = Get-FileHashSafe -Path $envPath

    Write-Log "Ejecutando auto-optimizacion"
    & $pythonPath "scripts\auto_optimize_agentic.py" --hours $Hours --apply | Out-Null

    $afterHash = Get-FileHashSafe -Path $envPath

    if ($beforeHash -ne $afterHash) {
        Write-Log ".env actualizado por auto-optimizacion"
        Restart-LiveBot
    }
    else {
        Write-Log "Sin cambios en .env; no se reinicia bot"
    }

    & $pythonPath "scripts\daily_live_report.py" | Out-Null
    $postMetrics = Get-CurrentMetrics

    $state.last = [PSCustomObject]@{ metrics = $postMetrics; updated_at = (Get-Date).ToString("s") }

    $bestMetricsAfter = if ($null -ne $state.best) { $state.best.metrics } else { $null }
    if (Is-Better -Current $postMetrics -Best $bestMetricsAfter) {
        $state.best = [PSCustomObject]@{ metrics = $postMetrics; updated_at = (Get-Date).ToString("s") }
        Snapshot-BestEnv
        Write-Log "Nuevo baseline estable guardado"
    }

    Save-State -State $state
}

if ($Loop) {
    Write-Log "Iniciando ciclo continuo cada $IntervalMinutes minutos"
    while ($true) {
        try {
            Invoke-OneCycle
        }
        catch {
            Write-Log ("Error en ciclo: " + $_.Exception.Message)
        }
        Start-Sleep -Seconds ([Math]::Max(60, $IntervalMinutes * 60))
    }
}
else {
    Write-Log "Ejecucion unica del ciclo"
    Invoke-OneCycle
}
