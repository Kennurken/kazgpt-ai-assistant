# PLAN D — Qwen2.5-1.5B + 10k subset + 2 epochs.
# Target: 8-10 hours (one night), val_loss 0.8-1.3.

$Root = $PSScriptRoot
$Venv = Join-Path $Root ".venv\Scripts\python.exe"
$Ts = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile = Join-Path $Root "logs\train_v3_pland_$Ts.log"
$PidFile = Join-Path $Root "logs\train_v3.pid"
New-Item -ItemType Directory -Force -Path (Split-Path $LogFile) | Out-Null

# Stop everything first
Get-Process -Name "python*" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Get-Process -Name "ollama*" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 3

# Verify model and data exist
$model = Join-Path $Root "models\qwen2.5-1.5b-instruct"
$data = Join-Path $Root "ml\data_v3_10k"
if (-not (Test-Path "$model\config.json")) {
    Write-Host "[FATAL] Qwen2.5-1.5B not downloaded yet. Wait for hf download." -ForegroundColor Red
    exit 1
}
if (-not (Test-Path "$data\train.jsonl")) {
    Write-Host "[FATAL] Subset data not generated. Run merge_datasets or create subset." -ForegroundColor Red
    exit 1
}

$cmd = @"
`$env:PYTHONIOENCODING = 'utf-8'
`$env:TOKENIZERS_PARALLELISM = 'false'
Set-Location '$Root'
& '$Venv' ml\train_cuda.py ``
    --model '$model' ``
    --data $data ``
    --output adapters_v3_pland ``
    --epochs 2 ``
    --lora-rank 64 ``
    --lora-alpha 128 ``
    --batch-size 2 ``
    --grad-accum 8 ``
    --lr 2e-4 ``
    --max-seq 768 ``
    --save-steps 200 ``
    --eval-steps 200 ``
    --logging-steps 10 *>&1 | Tee-Object -FilePath '$LogFile'
"@

$proc = Start-Process powershell -ArgumentList "-NoExit", "-Command", $cmd -PassThru -WindowStyle Hidden
$proc.Id | Out-File $PidFile -Encoding ascii

Write-Host ""
Write-Host "=== PLAN D LAUNCHED ===" -ForegroundColor Green
Write-Host "Model:  Qwen2.5-1.5B-Instruct"
Write-Host "Data:   $data (10,000 examples)"
Write-Host "Config: 2 epochs, rank=64, batch=2, grad_accum=8, max_seq=768"
Write-Host "ETA:    ~8-10 hours"
Write-Host "PID:    $($proc.Id)"
Write-Host "Log:    $LogFile"
Write-Host ""
Write-Host "Monitor: .\monitor_training.ps1 -Watch"
