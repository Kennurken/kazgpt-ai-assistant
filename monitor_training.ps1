# KazGPT V3 — Training monitor.
# Usage:
#   .\monitor_training.ps1           # 1 snapshot
#   .\monitor_training.ps1 -Watch    # tail log in real-time
#   .\monitor_training.ps1 -Kill     # stop training

param(
    [switch]$Watch,
    [switch]$Kill,
    [switch]$Stats
)

$Root = $PSScriptRoot
$PidFile = Join-Path $Root "logs\train_v3.pid"
$LogPattern = Join-Path $Root "logs\train_v3_*.log"

$latestLog = Get-ChildItem $LogPattern | Sort-Object LastWriteTime -Descending | Select-Object -First 1

if (-not $latestLog) {
    Write-Host "No training log found. Did you run launch_training_detached.ps1?" -ForegroundColor Yellow
    exit 1
}

if ($Kill) {
    if (Test-Path $PidFile) {
        $pid = Get-Content $PidFile
        Write-Host "Killing PID $pid..." -ForegroundColor Red
        Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
        Get-Process -Name "python*" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
        Remove-Item $PidFile -ErrorAction SilentlyContinue
        Write-Host "Done." -ForegroundColor Green
    }
    exit 0
}

# Process check
$pid = if (Test-Path $PidFile) { Get-Content $PidFile } else { $null }
$proc = if ($pid) { Get-Process -Id $pid -ErrorAction SilentlyContinue } else { $null }

Write-Host ""
Write-Host "=== KazGPT V3 Training Monitor ===" -ForegroundColor Cyan
Write-Host ""
if ($proc) {
    $runtime = (Get-Date) - $proc.StartTime
    Write-Host ("Status:   ALIVE (PID {0}, running {1:hh\:mm\:ss})" -f $pid, $runtime) -ForegroundColor Green
} else {
    Write-Host "Status:   DEAD or unknown PID" -ForegroundColor Yellow
}
Write-Host "Log file: $latestLog" -ForegroundColor Gray

# GPU stats
$gpu = (& nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader,nounits 2>$null)
if ($gpu) {
    $parts = $gpu -split ", "
    Write-Host ("GPU:      {0}MB used / {1}MB free, util {2}%, temp {3}C" -f $parts) -ForegroundColor Gray
}

# Latest loss values
Write-Host ""
Write-Host "=== Recent loss / metrics ===" -ForegroundColor Cyan
$lossLines = Get-Content $latestLog | Select-String -Pattern "'loss'" | Select-Object -Last 5
if ($lossLines) {
    foreach ($line in $lossLines) {
        # Extract loss and step
        $line.Line
    }
} else {
    Write-Host "No loss recorded yet (still loading?)" -ForegroundColor Yellow
}

# Eval loss
Write-Host ""
Write-Host "=== Recent eval_loss ===" -ForegroundColor Cyan
$evalLines = Get-Content $latestLog | Select-String -Pattern "eval_loss" | Select-Object -Last 3
if ($evalLines) {
    foreach ($line in $evalLines) { $line.Line }
} else {
    Write-Host "No eval yet (first eval at step 1000)" -ForegroundColor Yellow
}

if ($Watch) {
    Write-Host ""
    Write-Host "=== Tailing log (Ctrl+C to stop) ===" -ForegroundColor Cyan
    Get-Content $latestLog -Tail 20 -Wait
}
