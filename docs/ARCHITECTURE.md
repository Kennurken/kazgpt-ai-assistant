# KazGPT — Архитектура

## Высокоуровневый data flow

```
User → Browser → Spring Boot → [Ollama OR MLX server] → LLM → Stream → Browser
                       ↓ (если demo-mode)
                  Cache lookup → Stream from JSON
```

## Компоненты

### 1. Browser (vanilla JS)
- **Файлы:** `static/index.html`, `style.css`, `app.js`
- **Функции:** UI чата, стриминг через `fetch` + `ReadableStream`, хранение истории в `localStorage`, переключение моделей
- **Почему vanilla JS:** ноль билд-системы, ноль зависимостей, мгновенный старт

### 2. Spring Boot Backend
- **Application:** `KazGptApplication.java`
- **Конфиг:** `KazGptProperties.java` биндит `kazgpt.*` из `application.yml`
- **Контроллер:** `ChatController.java`
  - `POST /api/chat/stream` — основной streaming endpoint, возвращает `Flux<String>` с MediaType `text/event-stream`
  - `GET /api/health` — пингует Ollama и MLX server, возвращает их статус
  - `GET /api/models` — список доступных моделей
- **Сервисы:**
  - `ChatService` — главная логика. Выбирает рантайм (Ollama/MLX), строит messages, делает streaming POST через WebClient, парсит JSONL / SSE.
  - `CacheService` — загружает `cached_responses.json` при старте, ищет совпадения через Levenshtein.

### 3. LLM Runtimes
- **Ollama (port 11434):** держит `qwen2.5:7b`. API: `POST /api/chat` с `stream: true`, ответ JSONL.
- **mlx-lm server (port 11435):** держит Qwen2.5-1.5B + LoRA адаптер. API: `POST /v1/chat/completions` (OpenAI-совместимый), ответ SSE.

### 4. ML Pipeline
- **prepare_data.py** — скачивает KazQAD через HuggingFace datasets, конвертирует в jsonl с `{prompt, completion}`, режет на train/valid/test.
- **config.yaml + train.sh** — запуск `mlx_lm.lora` с LoRA на 4 слоях, batch_size=1, 200 итераций.
- **analyze.py** — парсит `train.log`, рисует график loss, генерирует `before_after.md`.

## Trade-offs

| Решение | Что выиграли | Что отдали |
|---------|--------------|------------|
| WebClient вместо RestTemplate | Non-blocking stream | Чуть больше boilerplate |
| Vanilla JS | Нет билда, ноль deps | Нет реактивности, всё на руках |
| Qwen2.5-1.5B 4bit для fine-tune | Влезает в 16GB, быстро учится | Качество ниже чем у 7B |
| Qwen2.5:7b для inference | Хорошее качество демо | Медленнее на M2 |
| LoRA вместо full fine-tune | 90% эффекта за 5% памяти | Адаптер нужно отдельно держать |
| MLX вместо PyTorch на Mac | В 2-3 раза быстрее на Apple Silicon | Уже экосистема vs PyTorch |
| Demo-cache fallback | Защита от падений на демо | Не настоящий ответ модели |

## Что бы поменяли при большем времени
- Full fine-tune на 5000+ итераций
- 7B fine-tune на серверной GPU
- Расширенный датасет (legal + medical + edu)
- Полноценная оценка через KazMMLU benchmark
- Whisper для голосового ввода
- Мобильный клиент

## Безопасность от "галлюцинаций"
1. **Temperature 0.3** — снижает креативность модели
2. **Repeat penalty 1.2** — против циклов
3. **Strict system prompt** — явный запрет придумывать
4. **Pre-tested questions** — demo вопросы прогнаны заранее
5. **Demo-cache** — последняя страховка на защите
