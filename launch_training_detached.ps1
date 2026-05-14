# Launch training as DETACHED background process.
# Survives Claude Code session restart.
#
# Usage:
#   .\launch_training_detached.ps1
#   Get-Content logs\train_v3_*.log -Tail 50  # monitor progress
#   Get-Process -Id (Get-Content logs\train_v3.pid)  # check if alive

$Root = $PSScriptRoot
$Ts = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile = Join-Path $Root "logs\train_v3_$Ts.log"
$PidFile = Join-Path $Root "logs\train_v3.pid"
New-Item -ItemType Directory -Force -Path (Split-Path $LogFile) | Out-Null

# Stop Ollama first
Write-Host "Stopping Ollama..." -ForegroundColor Cyan
Get-Process -Name "ollama*" -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 3

# Launch training_command in detached PowerShell window
$cmd = @"
`$env:PYTHONIOENCODING = 'utf-8'
`$env:TRANSFORMERS_VERBOSITY = 'info'
`$env:TOKENIZERS_PARALLELISM = 'false'
Set-Location '$Root'
& '$Root\.venv\Scripts\python.exe' ml\train_cuda.py ``
    --model '$Root\models\qwen2.5-7b-instruct' ``
    --data ml\data_v3 ``
    --output adapters_v3 ``
    --epochs 1 ``
    --lora-rank 32 ``
    --lora-alpha 64 ``
    --batch-size 1 ``
    --grad-accum 16 ``
    --lr 2e-4 ``
    --max-seq 1024 ``
    --save-steps 1000 ``
    --eval-steps 1000 ``
    --logging-steps 25 *>&1 | Tee-Object -FilePath '$LogFile'
"@

# Spawn as new powershell window — survives parent shell death
$proc = Start-Process powershell -ArgumentList "-NoExit", "-Command", $cmd -PassThru -WindowStyle Hidden

$proc.Id | Out-File $PidFile -Encoding ascii
Write-Host "Training launched. PID: $($proc.Id)" -ForegroundColor Green
Write-Host "Log:  $LogFile" -ForegroundColor Green
Write-Host "PID:  $PidFile" -ForegroundColor Green
Write-Host ""
Write-Host "Monitor: Get-Content '$LogFile' -Tail 50 -Wait" -ForegroundColor Cyan
Write-Host "Kill:    Stop-Process -Id $($proc.Id) -Force" -ForegroundColor Cyan
