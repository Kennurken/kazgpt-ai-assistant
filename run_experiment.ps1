# KazGPT — Automated A/B experiment runner.
#
# Flow:
#   1. Запускает Ollama (если не запущен)
#   2. Запускает backend в --baseline профиле → ждёт /api/health
#   3. Прогоняет eval против baseline → reports/{ts}_baseline.json
#   4. Останавливает backend
#   5. Запускает backend в Phase 0 (default) профиле → ждёт /api/health
#   6. Прогоняет eval против Phase 0 → reports/{ts}_phase0.json
#   7. Делает numerical diff
#   8. Открывает оба отчёта
#
# Usage:
#   .\run_experiment.ps1
#   .\run_experiment.ps1 -SkipBaseline   # только Phase 0
#   .\run_experiment.ps1 -EnableLLMJudge # с GPT-4 judge (платно, нужен OPENAI_API_KEY)

param(
    [switch]$SkipBaseline,
    [switch]$EnableLLMJudge,
    [switch]$EnableBertScore,
    [int]$Limit = 0  # 0 = все
)

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$BackendDir = Join-Path $Root "backend"
$EvalDir = Join-Path $Root "ml\eval"
$Venv = Join-Path $Root ".venv\Scripts"
$Python = Join-Path $Venv "python.exe"
$Mvn = "C:\ProgramData\chocolatey\lib\maven\apache-maven-3.9.15\bin\mvn.cmd"
$Jar = Join-Path $BackendDir "target\kazgpt-0.0.1-SNAPSHOT.jar"
$ReportsDir = Join-Path $EvalDir "reports"
$Ts = Get-Date -Format "yyyyMMdd_HHmmss"

function Wait-Health($url, $timeoutSec = 60) {
    $deadline = (Get-Date).AddSeconds($timeoutSec)
    while ((Get-Date) -lt $deadline) {
        try {
            $resp = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
            if ($resp.StatusCode -eq 200) { return $true }
        } catch {}
        Start-Sleep -Seconds 1
    }
    return $false
}

function Start-Backend($profile) {
    Write-Host "==> Starting backend (profile=$profile)..." -ForegroundColor Cyan
    $env:SPRING_PROFILES_ACTIVE = $profile
    $proc = Start-Process -FilePath "java" `
        -ArgumentList "-jar", $Jar, "--spring.profiles.active=$profile" `
        -PassThru -NoNewWindow `
        -RedirectStandardOutput (Join-Path $env:TEMP "kazgpt-backend-$profile.log") `
        -RedirectStandardError  (Join-Path $env:TEMP "kazgpt-backend-$profile.err.log")

    Write-Host "    PID: $($proc.Id), waiting /api/health..." -ForegroundColor Gray
    if (-not (Wait-Health "http://localhost:8080/api/health" 90)) {
        Write-Host "[FATAL] Backend did not start in 90s. См. $env:TEMP\kazgpt-backend-$profile.err.log" -ForegroundColor Red
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        exit 1
    }
    Write-Host "    Backend ready on :8080" -ForegroundColor Green
    return $proc
}

function Stop-Backend($proc) {
    if ($proc -and -not $proc.HasExited) {
        Write-Host "==> Stopping backend (PID $($proc.Id))..." -ForegroundColor Cyan
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 2
    }
}

function Run-Eval($modelName, $tag) {
    Write-Host "==> Running eval ($tag)..." -ForegroundColor Cyan
    $args = @(
        (Join-Path $EvalDir "run_eval.py"),
        "--models", $modelName,
        "--reports", $ReportsDir
    )
    if (-not $EnableBertScore -and -not $EnableLLMJudge) { $args += "--fast" }
    if ($EnableBertScore) { $args += "--enable-bertscore" }
    if ($EnableLLMJudge) { $args += "--enable-llm-judge" }
    if ($Limit -gt 0) { $args += @("--limit", $Limit) }

    $env:PYTHONIOENCODING = "utf-8"
    & $Python @args
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[FATAL] eval failed (exit $LASTEXITCODE)" -ForegroundColor Red
        exit 1
    }
}

# ============================================================
# 0. Sanity checks
# ============================================================
if (-not (Test-Path $Jar)) {
    Write-Host "==> JAR not found, building..." -ForegroundColor Yellow
    & $Mvn -f (Join-Path $BackendDir "pom.xml") package -DskipTests -q
    if ($LASTEXITCODE -ne 0) { Write-Host "[FATAL] mvn package failed" -ForegroundColor Red; exit 1 }
}

# Ollama check
$ollama = Get-Command ollama -ErrorAction SilentlyContinue
if (-not $ollama) {
    $ollama = Test-Path "C:\Users\user\AppData\Local\Programs\Ollama\ollama.exe"
}
if (-not $ollama) {
    Write-Host "[FATAL] Ollama not installed. Run: choco install ollama -y" -ForegroundColor Red
    exit 1
}

# Ollama serve check
try {
    Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -TimeoutSec 2 -UseBasicParsing | Out-Null
    Write-Host "==> Ollama already running" -ForegroundColor Green
} catch {
    Write-Host "==> Starting Ollama in background..." -ForegroundColor Cyan
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
    if (-not (Wait-Health "http://localhost:11434/api/tags" 30)) {
        Write-Host "[FATAL] Ollama did not start" -ForegroundColor Red
        exit 1
    }
    Write-Host "==> Ollama ready" -ForegroundColor Green
}

# Model check
$tags = (Invoke-WebRequest "http://localhost:11434/api/tags" -UseBasicParsing).Content | ConvertFrom-Json
if (-not ($tags.models | Where-Object { $_.name -match "qwen2.5" })) {
    Write-Host "==> qwen2.5:7b not pulled. Run: ollama pull qwen2.5:7b" -ForegroundColor Yellow
    Write-Host "    (это ~5GB, может занять 5-15 минут)"
    $ans = Read-Host "Скачать сейчас? [y/N]"
    if ($ans -eq "y" -or $ans -eq "Y") {
        & ollama pull qwen2.5:7b
        if ($LASTEXITCODE -ne 0) { exit 1 }
    } else {
        exit 1
    }
}

# ============================================================
# 1. Baseline run
# ============================================================
if (-not $SkipBaseline) {
    Write-Host "`n========== BASELINE RUN ==========" -ForegroundColor Magenta
    $proc = Start-Backend "baseline"
    try {
        Run-Eval "base" "baseline"
    } finally {
        Stop-Backend $proc
    }
}

# ============================================================
# 2. Phase 0 run
# ============================================================
Write-Host "`n========== PHASE 0 RUN ==========" -ForegroundColor Magenta
$proc = Start-Backend "default"
try {
    Run-Eval "base" "phase0"
} finally {
    Stop-Backend $proc
}

# ============================================================
# 3. Compare
# ============================================================
Write-Host "`n========== COMPARE ==========" -ForegroundColor Magenta
$reports = Get-ChildItem $ReportsDir -Filter "*.json" | Sort-Object LastWriteTime -Descending | Select-Object -First 2
if ($reports.Count -lt 2) {
    Write-Host "Need 2 reports for diff. Run without -SkipBaseline." -ForegroundColor Yellow
    exit 0
}

$newer = Get-Content $reports[0].FullName | ConvertFrom-Json
$older = Get-Content $reports[1].FullName | ConvertFrom-Json

Write-Host "`nBaseline ($($reports[1].Name)) vs Phase 0 ($($reports[0].Name))" -ForegroundColor Cyan
Write-Host "Metric              | baseline | phase 0  | delta"
Write-Host "--------------------|----------|----------|------"
$keys = @("bleu_avg", "rouge_l_avg", "bertscore_avg", "loop_pct", "kz_purity_avg", "assertions_pct", "ttft_p50_ms")
foreach ($k in $keys) {
    $a = $older.summary.base.$k
    $b = $newer.summary.base.$k
    $delta = if ($a -ne $null -and $b -ne $null) { [math]::Round($b - $a, 3) } else { "n/a" }
    Write-Host ("{0,-20}| {1,-9}| {2,-9}| {3}" -f $k, $a, $b, $delta)
}

Write-Host "`nReports:"
Write-Host "  Baseline: $($reports[1].FullName)"
Write-Host "  Phase 0:  $($reports[0].FullName)"
Write-Host "`nMarkdown отчёты в той же папке (.md)"
