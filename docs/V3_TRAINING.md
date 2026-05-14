# KazGPT V3 — Production Training Pipeline

**Цель:** val loss < 0.5, грамотный казахский ответ без ошибок, без аномалий.

---

## TL;DR

| Аспект | V2 (MacBook) | V3 (RTX 3070 Ti) |
|--------|:---:|:---:|
| Base model | Qwen2.5-1.5B (4bit) | Qwen2.5-7B-Instruct (4bit) |
| Method | MLX LoRA r=4 | QLoRA r=64 (transformers+peft) |
| Train data | 173 examples | **76,305 examples** |
| Total dataset | 215 | **82,941** |
| Iters | 200 | ~10,000 steps |
| Wall time | 4 минуты | ~10-13 часов |
| Val loss | 1.98 (final) | **target: 0.4-0.7** |

V3 — это **354× больше данных + 5× больше модели** на правильном железе.

---

## 1. Dataset (76k examples)

### Источники и фильтрация

| Источник | Raw | После фильтрации | Качество |
|----------|:---:|:---:|:---:|
| `AmanMussa/kazakh-instruction-v2` | 52,201 | ~50k | Community-validated, чистый KZ Alpaca |
| `saillab/alpaca-kazakh-cleaned` | 52,002 | ~24k | Cleaned Alpaca, отброшен 7% EN-noise |
| Наш Wiki+synthetic | 4,976 | ~2k | First-sentence Q&A |
| `saillab/alpaca_kazakh_taco` | 49,601 | ❌ 0 | **Отброшен** — 50% English mixing |

**ИТОГО:** 82,941 quality examples (train 76,305 / valid 4,147 / test 2,489)

### Pipeline фильтрации (`ml/merge_datasets.py`)

1. **Загружаем 3 источника параллельно** через HF datasets
2. **English-noise detection**: regex против `"Instruction in English"`, `"<|start_header_id|>"`, etc.
3. **Длина**: prompt 2–2000 слов, output 3–1500 слов
4. **KZ purity ≥ 0.85**: метрика из `ml/eval/metrics/kz_purity.py` (доля казахских слов в output)
5. **Dedup**: hash(prompt.lower() + output.lower())
6. **Stratified split**: 92% train / 5% valid / 3% test

Это даёт **гомогенный, грамотный, instruction-following** датасет.

---

## 2. Model + Training Config

### Base: Qwen2.5-7B-Instruct (public, без gating)

**Почему Qwen2.5, а не KazLLM-1.0-8B:**
- KazLLM-8B gated через ISSAI (~1-30 мин wait на approval, иногда сутки)
- Qwen2.5 имеет **великолепный multilingual** baseline (включая казахский) благодаря тренировке на 7T+ tokens
- Qwen2.5-7B уже instruction-tuned → safer LoRA сверху (не разрушает следование инструкциям)

### QLoRA параметры (`ml/train_cuda.py` defaults)

```python
LORA_RANK = 64          # большая capacity для 76k данных
LORA_ALPHA = 128        # = 2 × rank (best practice)
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
                        # all-linear — максимальное покрытие attention + FFN

QUANTIZATION = 4-bit NF4 (BitsAndBytes)
DOUBLE_QUANT = True     # дополнительная компрессия для VRAM

EPOCHS = 2.0
BATCH_SIZE = 1
GRAD_ACCUM = 16         # effective batch = 16
LR = 2e-4 → 2e-6        # cosine schedule
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01
MAX_SEQ = 2048

BF16 = True             # лучше fp16 на Ampere (RTX 3070 Ti)
GRADIENT_CHECKPOINTING = True  # -40% VRAM, +30% compute
PACKING = True          # упаковка коротких examples в длинные → 2-3× speed
EARLY_STOPPING = patience=3, на eval_loss
SAVE_BEST_ONLY = True
```

### VRAM math для RTX 3070 Ti (8.6GB)

```
4-bit base weights:  4.5 GB
LoRA r=64 adapter:   0.6 GB
Activations (bf16):  1.0 GB
Gradients:           0.4 GB
KV cache (2048 seq): 0.5 GB
Optimizer states:    0.1 GB (LoRA only)
                    ──────
Peak VRAM:          ~7.1 GB  ← с запасом 1.5GB
```

---

## 3. Pipeline (one command)

```powershell
# Перед запуском: убедись что Ollama выгрузил все модели
.\train_v3.ps1
```

Внутри происходит:
1. **Pre-flight**: проверка модели, данных, VRAM
2. **Stop Ollama** → free ~887MB VRAM
3. **Train**: `python ml/train_cuda.py --model ... --data ml/data_v3 --epochs 2 ...`
4. **Logging**: TensorBoard + текстовый лог `logs/train_v3_{ts}.log`
5. **Checkpoints**: каждые 500 шагов в `adapters_v3/checkpoint-N/`
6. **Eval**: каждые 500 шагов на valid set (4k examples)
7. **Fuse**: после training → LoRA merge → fused HF model
8. **GGUF**: convert fused → `kazgpt-v3-Q4_K_M.gguf` (через llama.cpp)
9. **Ollama**: `ollama create kazgpt-v3 -f Modelfile` → готово к inference
10. **Final eval**: A/B сравнение v3 vs base через `run_experiment.ps1`

ETA: **10-13 часов на ночь**, утром готовый артефакт.

---

## 4. Что меряем (как поймём что получилось)

Все метрики через `ml/eval/run_eval.py`:

| Метрика | Phase 0.1 baseline | V3 target |
|---------|:---:|:---:|
| BLEU | 17.36 | > 25 |
| ROUGE-L | 0.12 | > 0.20 |
| BERTScore (multilingual) | — | > 0.78 |
| LLM-judge (GPT-4o-mini, 1-5) | — | > 4.0 |
| Loop% | 0% | 0% |
| KZ purity | 0.904 | > 0.95 |
| Assertions% | 83.3% | > 90% |
| Val loss (training metric) | — | **0.4-0.7** |

### KazMMLU (future)

Стандартный benchmark из paper [arXiv:2502.12829](https://arxiv.org/abs/2502.12829).
Подключим после первого V3 чекпоинта — даст cross-comparable числа с ISSAI работами.

---

## 5. Riskы и митигация

| Риск | Митигация |
|------|-----------|
| OOM на 7B + LoRA r=64 | Pre-flight smoke test на 5 шагов; fallback на r=32 |
| `bitsandbytes` Windows bug | Установлено через wheels, протестировано на CUDA 12.1 |
| Overfit к 2-й эпохе | `EarlyStoppingCallback` + `save_best_only` |
| Качество датасета (typos в AmanMussa) | KZ-purity filter ≥0.85, English-noise detector |
| Long-tail из Wikipedia | Random shuffle, длина-фильтр |

---

## 6. Воспроизводимость

Всё детерминированно (`seed=42` в `train_cuda.py`):
- `prepare_data.py` → одни и те же splits
- `merge_datasets.py` → одни и те же фильтры
- `train_cuda.py` → один и тот же loss curve

**Git состояние**: коммит [aea4fdf](https://github.com/Kennurken/kazgpt-ai-assistant/commit/aea4fdf) — full Phase 0.1 + V3 scripts.

**После V3**: новый коммит с adapter weights + GGUF + final eval reports.

---

## 7. Сравнение с KazLLM-1.0-8B (industrial baseline)

KazLLM от ISSAI Nazarbayev University — **state-of-the-art казахская LLM** (Nov 2024).

| Параметр | KazLLM-8B | Наш V3 |
|----------|:---:|:---:|
| Base | Llama-3.1-8B | Qwen2.5-7B-Instruct |
| Pre-training corpus | ~50GB KZ (closed) | Qwen pretraining (multilingual, ~7T tokens) |
| Fine-tune dataset | Не раскрыт | 76k AmanMussa + saillab cleaned |
| Hardware | Кластер | 1× RTX 3070 Ti |
| Cost | Industrial | $0 (open data + домашний ПК) |

**Наша уникальность:**
- Воспроизводимо одной командой
- Полностью open data (community datasets)
- Меньшая модель, но fine-tuned под short instruction format
- Eval harness с numerical comparison (BLEU/ROUGE/BERTScore/LLM-judge/loop_detector)

Это и есть демонстрация **practical ML инженерии**, а не догон ISSAI по объёму данных.
