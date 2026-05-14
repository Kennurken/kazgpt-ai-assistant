# PLAN B — switch to faster training config if 7B is too slow.
# Strategy: subset 76k → 15k, max_seq 1024 → 768, keep 7B model.
# Target: ~6-8 hours instead of ~30+ hours.
#
# Usage:
#   .\switch_to_plan_b.ps1            # subset + faster max_seq
#   .\switch_to_plan_b.ps1 -PlanC     # switch to Qwen2.5-3B (smaller model)

param(
    [switch]$PlanC
)

$Root = $PSScriptRoot
$Venv = Join-Path $Root ".venv\Scripts\python.exe"

Write-Host ""
Write-Host "=== SWITCHING TRAINING PLAN ===" -ForegroundColor Yellow
Write-Host ""

# 1. Kill current training
Write-Host "==> Stopping current training..." -ForegroundColor Cyan
$pidFile = Join-Path $Root "logs\train_v3.pid"
if (Test-Path $pidFile) {
    Stop-Process -Id (Get-Content $pidFile) -Force -ErrorAction SilentlyContinue
}
Get-Process -Name "python*" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 3

if ($PlanC) {
    # PLAN C: smaller model (Qwen2.5-3B)
    Write-Host "==> PLAN C: switch to Qwen2.5-3B-Instruct (3x faster)" -ForegroundColor Magenta

    $modelDir = Join-Path $Root "models\qwen2.5-3b-instruct"
    if (-not (Test-Path $modelDir)) {
        Write-Host "  Downloading Qwen2.5-3B-Instruct (~6GB)..." -ForegroundColor Gray
        & hf download Qwen/Qwen2.5-3B-Instruct --local-dir $modelDir
    }

    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    $logFile = Join-Path $Root "logs\train_v3_planc_$ts.log"

    $cmd = @"
`$env:PYTHONIOENCODING = 'utf-8'
`$env:TOKENIZERS_PARALLELISM = 'false'
Set-Location '$Root'
& '$Venv' ml\train_cuda.py ``
    --model '$modelDir' ``
    --data ml\data_v3 ``
    --output adapters_v3_planc ``
    --epochs 1 ``
    --lora-rank 32 ``
    --lora-alpha 64 ``
    --batch-size 2 ``
    --grad-accum 8 ``
    --lr 2e-4 ``
    --max-seq 1024 ``
    --save-steps 500 ``
    --eval-steps 500 ``
    --logging-steps 25 *>&1 | Tee-Object -FilePath '$logFile'
"@
    $proc = Start-Process powershell -ArgumentList "-NoExit", "-Command", $cmd -PassThru -WindowStyle Hidden
    $proc.Id | Out-File $pidFile -Encoding ascii
    Write-Host "  PLAN C running. PID: $($proc.Id). Log: $logFile" -ForegroundColor Green

} else {
    # PLAN B: subset + faster max_seq
    Write-Host "==> PLAN B: subset 15k + max_seq 768 (5x faster)" -ForegroundColor Magenta

    # 1. Generate subset
    Write-Host "  Generating 15k subset..." -ForegroundColor Gray
    $env:PYTHONIOENCODING = "utf-8"
    & $Venv -c @"
import json, random
random.seed(42)
src = r'$Root\ml\data_v3\train.jsonl'
out_dir = r'$Root\ml\data_v3_subset'
import os
os.makedirs(out_dir, exist_ok=True)

# Subsample train to 15k
with open(src, 'r', encoding='utf-8') as f:
    lines = f.readlines()
random.shuffle(lines)
subset = lines[:15000]
with open(os.path.join(out_dir, 'train.jsonl'), 'w', encoding='utf-8') as f:
    f.writelines(subset)
print(f'Subset: {len(subset)} examples saved to {out_dir}/train.jsonl')

# Reuse valid.jsonl as is
import shutil
shutil.copy(r'$Root\ml\data_v3\valid.jsonl', os.path.join(out_dir, 'valid.jsonl'))
shutil.copy(r'$Root\ml\data_v3\test.jsonl', os.path.join(out_dir, 'test.jsonl'))
"@

    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    $logFile = Join-Path $Root "logs\train_v3_planb_$ts.log"

    $cmd = @"
`$env:PYTHONIOENCODING = 'utf-8'
`$env:TOKENIZERS_PARALLELISM = 'false'
Set-Location '$Root'
& '$Venv' ml\train_cuda.py ``
    --model '$Root\models\qwen2.5-7b-instruct' ``
    --data ml\data_v3_subset ``
    --output adapters_v3_planb ``
    --epochs 1 ``
    --lora-rank 32 ``
    --lora-alpha 64 ``
    --batch-size 1 ``
    --grad-accum 8 ``
    --lr 2e-4 ``
    --max-seq 768 ``
    --save-steps 200 ``
    --eval-steps 200 ``
    --logging-steps 10 *>&1 | Tee-Object -FilePath '$logFile'
"@
    $proc = Start-Process powershell -ArgumentList "-NoExit", "-Command", $cmd -PassThru -WindowStyle Hidden
    $proc.Id | Out-File $pidFile -Encoding ascii
    Write-Host "  PLAN B running. PID: $($proc.Id). Log: $logFile" -ForegroundColor Green
}

Write-Host ""
Write-Host "ETA:" -ForegroundColor Cyan
if ($PlanC) {
    Write-Host "  PLAN C (3B + 76k): ~12-16 hours for 1 epoch"
} else {
    Write-Host "  PLAN B (7B + 15k + max_seq 768): ~6-8 hours for 1 epoch"
}
Write-Host ""
Write-Host "Monitor: .\monitor_training.ps1" -ForegroundColor Gray
