# KazGPT V3 — EMERGENCY EVAL.
# Используется если training не завершился за выделенное время,
# но есть intermediate checkpoint. Берёт последний и делает fuse+GGUF+Ollama+eval.
#
# Usage:
#   .\emergency_eval.ps1                        # авто-выбор последнего checkpoint
#   .\emergency_eval.ps1 -Checkpoint adapters_v3\checkpoint-1000

param(
    [string]$Checkpoint = "",
    [switch]$SkipFuse,
    [switch]$SkipEval,
    [switch]$KillTraining
)

$Root = $PSScriptRoot
$Venv = Join-Path $Root ".venv\Scripts\python.exe"
$AdaptersDir = Join-Path $Root "adapters_v3"

# 1. Опционально убить training
if ($KillTraining) {
    Write-Host "==> Killing training..." -ForegroundColor Red
    $pidFile = Join-Path $Root "logs\train_v3.pid"
    if (Test-Path $pidFile) {
        Stop-Process -Id (Get-Content $pidFile) -Force -ErrorAction SilentlyContinue
    }
    Get-Process -Name "python*" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 3
}

# 2. Auto-select checkpoint
if (-not $Checkpoint) {
    # Prefer "final", else latest checkpoint-N
    if (Test-Path (Join-Path $AdaptersDir "final")) {
        $Checkpoint = Join-Path $AdaptersDir "final"
    } else {
        $latest = Get-ChildItem $AdaptersDir -Directory -Filter "checkpoint-*" -ErrorAction SilentlyContinue |
                  Sort-Object { [int]($_.Name -replace "checkpoint-", "") } -Descending |
                  Select-Object -First 1
        if ($latest) {
            $Checkpoint = $latest.FullName
        }
    }
}

if (-not $Checkpoint -or -not (Test-Path $Checkpoint)) {
    Write-Host "[FATAL] No checkpoint found in $AdaptersDir" -ForegroundColor Red
    Write-Host "  Available: $(Get-ChildItem $AdaptersDir -Directory -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Name)" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "=== KazGPT V3 Emergency Eval ===" -ForegroundColor Cyan
Write-Host "Checkpoint: $Checkpoint" -ForegroundColor Green

# Check what step this is
$stepNum = if ($Checkpoint -match "checkpoint-(\d+)") { [int]$matches[1] } else { "final" }
Write-Host "Step:       $stepNum"

# 3. Fuse LoRA + base → merged HF model
$mergedDir = Join-Path $Root "kazgpt-v3-merged"
if (-not $SkipFuse) {
    Write-Host ""
    Write-Host "==> Fuse LoRA into base..." -ForegroundColor Cyan
    & $Venv (Join-Path $Root "ml\fuse_and_export.py") `
        --base (Join-Path $Root "models\qwen2.5-7b-instruct") `
        --adapter $Checkpoint `
        --output $mergedDir `
        --gguf-out (Join-Path $Root "kazgpt-v3.gguf") `
        --quantize Q4_K_M `
        --llama-cpp (Join-Path $Root "llama.cpp")
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[WARN] Fuse failed; пытаемся только HF merge без GGUF" -ForegroundColor Yellow
        & $Venv (Join-Path $Root "ml\fuse_and_export.py") `
            --base (Join-Path $Root "models\qwen2.5-7b-instruct") `
            --adapter $Checkpoint `
            --output $mergedDir `
            --skip-gguf
    }
}

# 4. Eval through Python (без backend для скорости)
if (-not $SkipEval) {
    Write-Host ""
    Write-Host "==> Quick eval via merged HF model directly..." -ForegroundColor Cyan

    $env:PYTHONIOENCODING = "utf-8"
    & $Venv -c @"
import sys, json
from pathlib import Path
sys.path.insert(0, r'$Root\ml\eval')
from metrics.loop_detector import detect_loop
from metrics.kz_purity import kz_purity
from metrics.automatic import compute_all

# Load merged model
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
print('Loading merged model...')
tok = AutoTokenizer.from_pretrained(r'$mergedDir')
m = AutoModelForCausalLM.from_pretrained(r'$mergedDir', torch_dtype=torch.bfloat16, device_map='auto')
m.eval()

# Load golden_set
items = []
with open(r'$Root\ml\eval\golden_set.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            items.append(json.loads(line))

print(f'Evaluating {len(items)} golden items...')
results = []
for it in items:
    msgs = [
        {'role': 'system', 'content': 'Сен — KazGPT, қазақ тілінде сөйлейтін AI-көмекші. Қысқа, анық жауап бер.'},
        {'role': 'user', 'content': it['question']}
    ]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors='pt').to(m.device)
    with torch.no_grad():
        out = m.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.3, top_p=0.85, top_k=40, repetition_penalty=1.15)
    resp = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    loop = detect_loop(resp)
    purity = kz_purity(resp)
    auto = compute_all(resp, it['reference_answers'], it.get('must_contain_any', []), it.get('must_not_contain', []), skip_bertscore=True)

    marker = '!' if loop['is_loop'] else '+'
    print(f'  [{marker}] {it[\"id\"]:<20} bleu={auto[\"bleu\"]} loop={loop[\"repetition_rate\"]} purity={purity[\"purity\"]}')
    results.append({'id': it['id'], 'response': resp, 'auto': auto, 'loop': loop, 'purity': purity})

# Save
from datetime import datetime
ts = datetime.now().strftime('%Y%m%d_%H%M%S')
out_path = Path(r'$Root\ml\eval\reports') / f'v3_emergency_{ts}.json'
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f'\nSaved: {out_path}')

# Summary
bleus = [r['auto']['bleu'] for r in results if r['auto']['bleu'] is not None]
purities = [r['purity']['purity'] for r in results]
loops = sum(1 for r in results if r['loop']['is_loop'])
print(f'\nSUMMARY: avg BLEU={sum(bleus)/len(bleus):.2f}, avg purity={sum(purities)/len(purities):.3f}, loops={loops}/{len(results)}')
"@
}

Write-Host ""
Write-Host "==> DONE" -ForegroundColor Green
Write-Host "Merged: $mergedDir"
Write-Host "Report: $Root\ml\eval\reports\v3_emergency_*.json"
