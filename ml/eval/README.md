# KazGPT Eval Harness

Цель: измерять качество ответов KazGPT **численно**, чтобы каждое изменение
(новый fine-tune, новые sampling-параметры, новый system prompt, RAG) можно
было сравнить с baseline по объективным метрикам.

## Что измеряем

| Метрика | Что показывает | Стоимость | Целевое |
|---------|----------------|-----------|---------|
| BLEU-4 | Lexical overlap с эталоном | бесплатно | ≥ 15 |
| ROUGE-L | LCS overlap | бесплатно | ≥ 0.30 |
| BERTScore | Семантическое сходство (multilingual) | ~250MB модель, ~1s/пример | ≥ 0.75 |
| Loop rate | % ответов с повторяющимися 4-граммами | бесплатно | ≤ 5% |
| KZ purity | % казахских слов в ответе | бесплатно | ≥ 90% |
| Assertions | % ответов с обязательными ключевыми словами | бесплатно | ≥ 80% |
| TTFT p50 | Время до первого токена (UX) | бесплатно | ≤ 1.5s |
| LLM-judge | GPT-4 оценка 1-5 (coherence, fluency, correctness) | ~$0.01/пример | ≥ 4.0 |

## Установка

```bash
cd ml/eval
python -m venv .venv
source .venv/bin/activate          # или .venv\Scripts\activate на Windows
pip install -r requirements.txt
```

(Минимальный запуск работает без `bert-score` / `openai`, метрики просто отключатся.)

## Запуск

Убедись, что **бэкенд запущен** (`mvn spring-boot:run` в `backend/`) и хотя бы
одна модель доступна (Ollama для `base`, MLX для `v2`).

```bash
# Быстрый смоук-тест: 1 модель, без тяжёлых метрик
python run_eval.py --models base --fast

# Полный прогон с BERTScore
python run_eval.py --models base v2 --enable-bertscore

# С LLM-judge (нужен OPENAI_API_KEY и заполненный JUDGE_PROMPT)
export OPENAI_API_KEY=sk-...
python run_eval.py --models base --enable-llm-judge --judge-provider openai

# Фильтр по домену
python run_eval.py --models base --domain kz_knowledge

# Ограничить число примеров (дебаг)
python run_eval.py --models base --limit 5
```

Результаты:
- `reports/{timestamp}.json` — машинно-читаемый, для CI/diff между прогонами
- `reports/{timestamp}.md` — для людей, с таблицей и failures

## Структура golden set

`golden_set.jsonl` — по одному JSON на строку. Схема:

```json
{
  "id": "kz_knowledge_001",
  "domain": "kz_knowledge",
  "question": "Қазақстанның қазіргі астанасы қандай қала?",
  "reference_answers": ["Астана — Қазақстанның астанасы.", "Қазақстанның астанасы Астана."],
  "must_contain_any": ["Астана"],
  "must_not_contain": ["Нұр-Сұлтан"],
  "tags": ["geography", "factual", "easy"]
}
```

**Домены сейчас (4):**
- `daily_kazakh` — повседневный казахский, code-switching (KZ ↔ RU ↔ EN)
- `kz_knowledge` — знания о Казахстане (история, география, культура)
- `kz_language` — переводы, грамматика, пословицы
- `text_help` — резюме, перефраз, объяснения

## TODO для пользователя

1. **Расширить golden_set.jsonl с 12 до 50 примеров** — по 10–15 на домен.
   Сейчас 12 примеров — достаточно для проверки пайплайна, но мало для надёжной метрики.

2. **Заполнить `JUDGE_PROMPT`** в `metrics/llm_judge.py` — это твой авторский
   prompt-engineering, который определит как GPT-4 будет оценивать ответы.
   Смотри детальные комментарии в начале файла.

3. **Запустить baseline** прогон на текущей `base` модели (Qwen2.5:7b) — это
   будет точка отсчёта для всех будущих улучшений.

## Интеграция с CI

После того как baseline стабилизируется, можно добавить в `.github/workflows/`:
```yaml
- name: KazGPT eval gate
  run: |
    python ml/eval/run_eval.py --models base --fast
    # парсить reports/*.json и проваливать билд при регрессе > 5%
```

## Архитектурное решение

- **Hit through backend** (не напрямую в Ollama) — измеряем всю систему,
  включая system prompt, SSE парсинг, обработку истории.
- **Graceful degradation** для метрик — если пакет/API недоступен, метрика
  возвращает None, прогон продолжается.
- **JSONL формат** для golden set — легко добавлять новые примеры через
  любой текстовый редактор, легко diff'ать в git.
