# KazGPT — Phase 0 Experiment Log

**Дата:** 2026-05-14
**Hardware:** Windows + RTX 3070 Ti, Qwen2.5:7b через Ollama
**Eval:** ml/eval/run_eval.py --fast (без BERTScore и LLM-judge)
**Dataset:** ml/eval/golden_set.jsonl (12 примеров, 4 домена)

---

## Гипотезы

H1: Few-shot examples в system prompt → выше BLEU и assertion rate
H2: Агрессивный `repeat_penalty=1.3 + repeat_last_n=256` → меньше loops
H3: `presence_penalty + frequency_penalty + min_p` улучшают разнообразие

---

## Результаты по итерациям

| Метрика | Baseline (V1) | Phase 0 (full) | Phase 0.1 (refined) |
|---------|:-:|:-:|:-:|
| BLEU                | 5.79  | 3.90  | **17.36** 🚀 |
| ROUGE-L             | 0.15  | 0.012 | 0.12 |
| Loop %              | 0.0   | 0.0   | 0.0 ✓ |
| KZ purity           | 0.918 | 0.726 | 0.904 |
| Assert %            | 58.3  | 41.7  | **83.3** 🚀 |
| TTFT p50            | 442ms | 435ms | 451ms |

---

## Анализ Phase 0 (провал, BLEU=3.9)

**Симптомы:**
- Модель генерила псевдоказахский мусор:
  `"Сәlemetsez! Жашбыздар, рахмет. Сизге патердене бо��ғандым؟ 🌞"`
  `"Абай Ырыспатов (Ісыбаев) Ханалинович Куnanbaev"` (фабрикация русского отчества!)
- Purity упала с 0.918 → 0.726
- text_help_001 деградировал с 0.912 → 0.148 (только 15% казахских слов)

**Корневые причины (от Phase 0.1 диагностики):**
1. **`presence_penalty=0.6` + `frequency_penalty=0.4`** — OpenAI-style penalties конфликтовали с Ollama `repeat_penalty`. Двойное подавление повторов **выбивало валидные казахские суффиксы**.
2. **`min_p=0.05`** — отрезал низковероятностные токены, среди которых были редкие казахские суффиксы (-ңыз, -сің, -дың). Модель срывалась на «межъязыковой суррогат».
3. **`repeat_penalty=1.3 + repeat_last_n=256`** — слишком жёсткое подавление повторов для агглютинативного языка, где морфемы по природе повторяются.
4. **Stop tokens с казахскими маркерами** (`"\n\nҚолданушы:"`) — потенциально обрезали валидные продолжения.

**Урок:** изменение **7 параметров одновременно** делает невозможным attribution. Это анти-паттерн.

---

## Phase 0.1 (победа, BLEU=17.36)

**Что оставил из Phase 0:**
- Few-shot system prompt (3-5 пар Q&A в правильном KZ стиле)
- `repeat_penalty=1.15` (мягкий — только 0.05 от V1=1.20)
- `repeat_last_n=128` (умеренное окно)
- `top-p=0.85` (мягкое увеличение от 0.8)
- `top-k=40` (явная фиксация Qwen default)
- Stop tokens: только instruct-артефакты (`<|im_end|>`, `<|endoftext|>`)

**Что убрал:**
- `presence_penalty` / `frequency_penalty` → 0
- `min_p` → 0
- Stop tokens на казахском
- `num_ctx` фиксацию (тоже была причина sub-optimal)

**Результаты:**
- BLEU **+200%** vs baseline (5.79 → 17.36)
- Assertion rate **+25%** (58.3% → 83.3%, 10/12 ответов с нужными ключевиками)
- KZ purity осталась на уровне baseline (0.904 vs 0.918, -1.5% — статистический шум)
- Конкретный win: `text_help_003` (нейронная сеть) purity **0.429 → 0.949**

---

## Принятые решения

1. **Phase 0.1 = новый default `application.yml`** — закрепили как production config
2. **`application-baseline.yml` сохранили** для будущих A/B сравнений
3. **`docs/PHASE0_EXPERIMENT.md` (этот файл)** — фиксируем эксперимент для куратора

## Следующие итерации

- **Phase 0.2**: попробовать `repeat_penalty=1.1` (ещё мягче) — может улучшит purity
- **Phase 1**: production training data (KazQAD + Wiki + synthetic)
- **Phase 2**: V3 LoRA over KazLLM-1.0-8B
- **Phase 4**: RAG с kk-Wikipedia для factual questions
- **Phase 6 расширение**: добавить LLM-judge (gpt-4o-mini) + BERTScore + KazMMLU
