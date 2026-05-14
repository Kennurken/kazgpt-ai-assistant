# Continue V3 training — resume from checkpoint-592, add 2 more epochs.
# Strategy: constant low LR (5e-6) для gentle polishing, без agressive learning.

$Root = $PSScriptRoot
$Venv = Join-Path $Root ".venv\Scripts\python.exe"
$Ts = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile = Join-Path $Root "logs\train_v3_continue_$Ts.log"
$PidFile = Join-Path $Root "logs\train_v3.pid"

# Verify checkpoint exists
$Checkpoint = Join-Path $Root "adapters_v3_pland\checkpoint-592"
if (-not (Test-Path $Checkpoint)) {
    Write-Host "[FATAL] No checkpoint-592 found" -ForegroundColor Red
    exit 1
}

# Stop any running processes
Get-Process -Name "python*" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Get-Process -Name "ollama*" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 3

$cmd = @"
`$env:PYTHONIOENCODING = 'utf-8'
`$env:TOKENIZERS_PARALLELISM = 'false'
Set-Location '$Root'
& '$Venv' ml\train_cuda.py ``
    --model '$Root\models\qwen2.5-1.5b-instruct' ``
    --data ml\data_v3_10k ``
    --output adapters_v3_pland ``
    --epochs 4 ``
    --lora-rank 64 ``
    --lora-alpha 128 ``
    --batch-size 2 ``
    --grad-accum 8 ``
    --lr 5e-5 ``
    --max-seq 768 ``
    --save-steps 200 ``
    --eval-steps 200 ``
    --logging-steps 10 ``
    --resume '$Checkpoint' *>&1 | Tee-Object -FilePath '$LogFile'
"@

$proc = Start-Process powershell -ArgumentList "-NoExit", "-Command", $cmd -PassThru -WindowStyle Hidden
$proc.Id | Out-File $PidFile -Encoding ascii

Write-Host ""
Write-Host "=== V3 CONTINUE TRAINING LAUNCHED ===" -ForegroundColor Green
Write-Host "Resume from:  checkpoint-592 (val_loss 1.078)"
Write-Host "Target:       4 total epochs (594 more steps from 592)"
Write-Host "LR:           5e-5 (gentle polishing)"
Write-Host "ETA:          ~1.5 hours"
Write-Host "PID:          $($proc.Id)"
Write-Host "Log:          $LogFile"
Write-Host ""
Write-Host "Monitor: .\monitor_training.ps1"
