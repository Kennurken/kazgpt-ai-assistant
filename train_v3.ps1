# KazGPT V3 — One-command production training pipeline.
#
# Запуск перед сном:
#   .\train_v3.ps1
#
# Что делает:
# 1. Останавливает Ollama (освобождает VRAM)
# 2. Запускает train_cuda.py против Qwen2.5-7B-Instruct + LoRA r=64
#    на ml/data_v3 (76k examples) — 2 epochs, ~10-13 часов
# 3. Утром: fuse LoRA → fused model → GGUF Q4_K_M → Ollama create kazgpt-v3
# 4. Финальный eval vs base qwen2.5:7b
#
# Параметры можно переопределить:
#   .\train_v3.ps1 -Epochs 3 -LoraRank 32   # дольше но меньше capacity
#   .\train_v3.ps1 -DryRun                  # печатает команду, не запускает

param(
    [string]$Model = "C:\app\kazgpt-ai-assistant\models\qwen2.5-7b-instruct",
    [string]$DataDir = ".\ml\data_v3",
    [string]$OutDir = ".\adapters_v3",
    [float]$Epochs = 2.0,
    [int]$LoraRank = 64,
    [int]$LoraAlpha = 128,
    [int]$BatchSize = 1,
    [int]$GradAccum = 16,
    [float]$Lr = 2e-4,
    [int]$MaxSeq = 2048,
    [int]$SaveSteps = 500,
    [int]$EvalSteps = 500,
    [switch]$DryRun,
    [switch]$SkipFuse,
    [switch]$SkipEval
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$Venv = Join-Path $Root ".venv\Scripts"
$Python = Join-Path $Venv "python.exe"
$Ts = Get-Date -Format "yyyyMMdd_HHmmss"
$TrainLog = Join-Path $Root "logs\train_v3_$Ts.log"
New-Item -ItemType Directory -Force -Path (Split-Path $TrainLog) | Out-Null

function Step($msg) {
    Write-Host ""
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] $msg" -ForegroundColor Cyan
}

# ============================================================
# 0. Pre-flight checks
# ============================================================
Step "Pre-flight checks"

if (-not (Test-Path $Model)) {
    Write-Host "[FATAL] Model not found: $Model" -ForegroundColor Red
    Write-Host "Run: hf download Qwen/Qwen2.5-7B-Instruct --local-dir $Model" -ForegroundColor Yellow
    exit 1
}
if (-not (Test-Path "$DataDir\train.jsonl")) {
    Write-Host "[FATAL] Train data not found: $DataDir\train.jsonl" -ForegroundColor Red
    Write-Host "Run: python ml/merge_datasets.py --output $DataDir" -ForegroundColor Yellow
    exit 1
}
if (-not (Test-Path $Python)) {
    Write-Host "[FATAL] venv not found: $Python" -ForegroundColor Red
    exit 1
}

$trainCount = (Get-Content "$DataDir\train.jsonl" | Measure-Object -Line).Lines
$validCount = (Get-Content "$DataDir\valid.jsonl" | Measure-Object -Line).Lines
$stepsPerEpoch = [math]::Ceiling($trainCount / $GradAccum)
$totalSteps = [math]::Ceiling($stepsPerEpoch * $Epochs)
$etaHours = [math]::Round($totalSteps * 4 / 3600, 1)  # ~4 sec/step on 3070 Ti

Write-Host "  Model: $Model"
Write-Host "  Data:  train=$trainCount, valid=$validCount"
Write-Host "  Steps: $stepsPerEpoch/epoch × $Epochs epochs = $totalSteps total"
Write-Host "  ETA:   ~$etaHours hours"
Write-Host "  Log:   $TrainLog"

# GPU check
$gpuInfo = & nvidia-smi --query-gpu=name,memory.free --format=csv,noheader 2>$null
Write-Host "  GPU:   $gpuInfo"

# ============================================================
# 1. Stop Ollama to free VRAM
# ============================================================
Step "Stop Ollama (free VRAM)"
try {
    $ollamaProcs = Get-Process -Name "ollama*" -ErrorAction SilentlyContinue
    if ($ollamaProcs) {
        $ollamaProcs | Stop-Process -Force
        Write-Host "  Stopped: $($ollamaProcs.Count) Ollama processes"
        Start-Sleep -Seconds 3
    } else {
        Write-Host "  Ollama not running"
    }
} catch {
    Write-Host "  Warning: $_" -ForegroundColor Yellow
}

# Verify VRAM freed
$gpuFree = (& nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits) -as [int]
Write-Host "  VRAM free: ${gpuFree}MB"
if ($gpuFree -lt 7000) {
    Write-Host "[WARN] Less than 7GB VRAM free. Возможен OOM на Qwen-7B + LoRA r=64." -ForegroundColor Yellow
    Write-Host "       Закрой Chrome video / Discord overlay для запаса." -ForegroundColor Yellow
}

# ============================================================
# 2. Run training
# ============================================================
Step "Production Training"

$trainArgs = @(
    (Join-Path $Root "ml\train_cuda.py"),
    "--model", $Model,
    "--data", $DataDir,
    "--output", $OutDir,
    "--epochs", $Epochs,
    "--lora-rank", $LoraRank,
    "--lora-alpha", $LoraAlpha,
    "--batch-size", $BatchSize,
    "--grad-accum", $GradAccum,
    "--lr", $Lr,
    "--max-seq", $MaxSeq,
    "--save-steps", $SaveSteps,
    "--eval-steps", $EvalSteps,
    "--logging-steps", 25
)

Write-Host "  Command:" -ForegroundColor Gray
Write-Host "    $Python $($trainArgs -join ' ')" -ForegroundColor Gray
Write-Host ""

if ($DryRun) {
    Write-Host "[DRY RUN] Would run command above. Exiting." -ForegroundColor Yellow
    exit 0
}

$env:PYTHONIOENCODING = "utf-8"
$env:TRANSFORMERS_VERBOSITY = "info"
$env:TOKENIZERS_PARALLELISM = "false"  # избегаем warning при datasets shuffle

$trainStart = Get-Date
& $Python @trainArgs 2>&1 | Tee-Object -FilePath $TrainLog

$trainEnd = Get-Date
$trainDuration = $trainEnd - $trainStart
Write-Host ""
Write-Host "[$trainEnd] Training finished. Duration: $($trainDuration.TotalHours.ToString('F1'))h" -ForegroundColor Green

if ($LASTEXITCODE -ne 0) {
    Write-Host "[FATAL] Training failed (exit $LASTEXITCODE). См. лог: $TrainLog" -ForegroundColor Red
    exit 1
}

# ============================================================
# 3. Fuse + GGUF + Ollama create
# ============================================================
if (-not $SkipFuse) {
    Step "Fuse LoRA + Export GGUF + Ollama"
    $finalAdapter = Join-Path $OutDir "final"
    if (-not (Test-Path $finalAdapter)) {
        Write-Host "[WARN] Final adapter not found at $finalAdapter, скип fuse" -ForegroundColor Yellow
    } else {
        & $Python (Join-Path $Root "ml\fuse_and_export.py") `
            --base $Model `
            --adapter $finalAdapter `
            --output (Join-Path $Root "kazgpt-v3-merged") `
            --gguf-out (Join-Path $Root "kazgpt-v3-Q4_K_M.gguf") `
            --quantize Q4_K_M
    }
}

# ============================================================
# 4. Restart Ollama + eval
# ============================================================
if (-not $SkipEval) {
    Step "Restart Ollama + Final Eval"
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 5

    # Если есть Modelfile, создаём kazgpt-v3
    if (Test-Path (Join-Path $Root "Modelfile")) {
        & ollama create kazgpt-v3 -f (Join-Path $Root "Modelfile")
    }

    # Запускаем backend в Phase 0.1 + eval
    # (TODO: добавь сюда A/B eval базы vs V3 если хочешь)
    Write-Host "  Manually run: .\run_experiment.ps1" -ForegroundColor Cyan
}

Step "DONE"
Write-Host "Adapter: $(Join-Path $OutDir 'final')" -ForegroundColor Green
Write-Host "Log:     $TrainLog" -ForegroundColor Green
