# KazGPT — Контекст для Claude Code

## Что это
Локальный AI-ассистент с поддержкой казахского языка. Bun/Java backend + vanilla JS frontend + локальные LLM модели через Ollama и MLX.

## Стек
- **Backend:** Java 17, Spring Boot 3.5.5 (Maven). Web + WebFlux (для WebClient streaming).
- **Frontend:** vanilla HTML/CSS/JS — без фреймворков, без билд-системы.
- **LLM inference:**
  - V1 (base): Ollama runtime на `localhost:11434`, модель `qwen2.5:7b`
  - V2 (fine-tuned): mlx-lm server на `localhost:11435`, модель `mlx-community/Qwen2.5-1.5B-Instruct-4bit` + LoRA адаптер
- **Fine-tune:** Apple MLX framework, LoRA, обучение на M2 16GB.

## Структура
```
kazgpt-ai-assistant/
├── CLAUDE.md, README.md, .gitignore
├── backend/                  ← Spring Boot
│   ├── pom.xml
│   └── src/main/
│       ├── java/kz/kazgpt/
│       │   ├── KazGptApplication.java
│       │   ├── config/       ← KazGptProperties, WebConfig (CORS)
│       │   ├── controller/   ← ChatController (SSE endpoints)
│       │   ├── service/      ← ChatService (streaming), CacheService
│       │   └── model/        ← ChatRequest, Message, CachedResponse
│       └── resources/
│           ├── application.yml
│           ├── cached_responses.json    ← демо-кэш для fallback
│           └── static/                  ← фронт (index.html, style.css, app.js)
├── ml/                       ← Python для fine-tune
│   ├── prepare_data.py       ← скачивает issai/kazqad
│   ├── config.yaml           ← параметры MLX LoRA
│   ├── train.sh              ← запуск обучения
│   ├── analyze.py            ← график loss + before/after
│   └── data/                 ← train/valid/test jsonl (генерируется)
├── demo/                     ← cached_responses backup, видео-демо
└── docs/                     ← training_curve.png, slides, ARCHITECTURE.md
```

## Конвенции
- Все Java классы — пакет `kz.kazgpt.*`
- API под `/api/*`
- SSE стрим через `produces = TEXT_EVENT_STREAM_VALUE`, `Flux<String>`
- Системный промпт в `application.yml`, на казахском, жёсткий (запрещает галлюцинации)
- Низкая температура (0.3) для предсказуемости

## Безопасность от "бреда" на демо
1. **Низкая temperature (0.3)** в application.yml
2. **Жёсткий system prompt** запрещает галлюцинации
3. **Demo-cache fallback** — `kazgpt.demo-mode: true` включает ответы из cached_responses.json
4. **15 проверенных вопросов** в кэше — полное покрытие demo-сценария

## Запуск
```bash
# В отдельных терминалах:
# 1. Ollama
brew services start ollama   # или: ollama serve

# 2. Backend
cd backend && mvn spring-boot:run

# 3. (Опционально) MLX server для v2 — после fine-tune
cd ml && python -m mlx_lm.server \
  --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
  --adapter-path ./adapters \
  --port 11435

# Открыть: http://localhost:8080
```

## Endpoints
- `POST /api/chat/stream` — SSE стрим, body `{message, history, model}`, model = "base" | "v2"
- `GET /api/health` — `{status, ollamaUp, mlxServerUp, demoMode, uptimeSeconds, version}`
- `GET /api/models` — список моделей и текущий default

## Current state
- [x] V1 backend (Spring Boot + Ollama integration)
- [x] V1 frontend (chat UI с streaming + model switcher + history)
- [x] Demo-cache fallback с 15 готовыми Q&A
- [x] Документация
- [ ] V2: fine-tune Qwen2.5-1.5B на KazQAD (запуск `ml/train.sh`, ~40 мин)
- [ ] Презентация (4 слайда)
- [ ] Видео-бэкап демо
