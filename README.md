# KazGPT — Локальный казахоязычный AI-ассистент

> Қазақ тілінде сөйлейтін жасанды интеллект. Без облаков. Без интернета.

## Аннотация

Казахский — один из **low-resource языков** в современном NLP. Крупные коммерческие модели (GPT-4, Gemini, Claude) допускают ошибки в грамматике и теряются на казахоязычных запросах. Существующие облачные решения создают проблемы приватности данных и недоступны без интернета.

**KazGPT** — это локальный AI-ассистент, работающий полностью на ноутбуке пользователя. В основе — модель Qwen2.5, дообученная (LoRA) на казахскоязычном датасете [KazQAD](https://huggingface.co/datasets/issai/kazqad) от лаборатории ISSAI (Назарбаев Университет). Бэкенд написан на Java Spring Boot, фронтенд — vanilla JS. Inference выполняется через Ollama (база) и MLX server (fine-tuned версия), что обеспечивает работу даже без доступа к сети.

## Архитектура

```
┌────────────────────────────────────────────┐
│  Browser (vanilla JS, localStorage)        │
│  • Streaming chat UI                      │
│  • Model switcher (base ↔ v2)             │
└────────────┬───────────────────────────────┘
             │ HTTP / SSE
             ▼
┌────────────────────────────────────────────┐
│  Spring Boot Backend (port 8080)           │
│  • POST /api/chat/stream  (SSE)           │
│  • GET  /api/health, /api/models          │
│  • Demo-cache fallback (страховка)        │
└─────┬─────────────────────────┬────────────┘
      │                         │
      ▼                         ▼
┌──────────────────┐    ┌──────────────────────┐
│ Ollama (11434)   │    │ mlx-lm server (11435)│
│ qwen2.5:7b       │    │ Qwen2.5-1.5B + LoRA  │
└──────────────────┘    └──────────────────────┘
```

## Технологии

| Компонент | Технология | Обоснование |
|-----------|------------|-------------|
| Backend | Java 17 + Spring Boot 3.5 | Стандарт enterprise, отличная поддержка reactive streams |
| Streaming | WebClient + SSE | Не блокирует поток, естественно для LLM stream |
| Frontend | Vanilla JS | Минимум зависимостей, никакого билда |
| Base LLM | Qwen2.5:7b (Ollama) | Хорошее качество на казахском из коробки |
| Fine-tune | MLX + LoRA | Нативно для Apple Silicon, 4bit quant влезает в 16GB |
| Dataset | KazQAD (ISSAI NU) | Открытый, специализированный на Kazakh QA |

## Запуск

```bash
# 1. Зависимости
brew install openjdk@17 maven ollama python@3.11
brew services start ollama
ollama pull qwen2.5:7b

# 2. Бэкенд
cd backend && mvn spring-boot:run
# Открыть http://localhost:8080

# 3. (Опционально) Fine-tune для v2
cd ml && ./train.sh

# 4. (Опционально) Запуск fine-tuned модели
python -m mlx_lm.server \
  --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
  --adapter-path ./adapters \
  --port 11435
```

## API

### POST /api/chat/stream
Стриминговый чат-эндпоинт.
```json
{
  "message": "Алматы туралы айтып бер",
  "history": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
  "model": "base"
}
```
Ответ: `text/event-stream` с чанками текста.

### GET /api/health
```json
{
  "status": "ok",
  "version": "1.0",
  "ollamaUp": true,
  "mlxServerUp": false,
  "demoMode": false,
  "uptimeSeconds": 142
}
```

### GET /api/models
Возвращает доступные модели.

## Roadmap

- **v1.0 (текущая):** Spring Boot бэкенд, Ollama base, demo-cache, streaming UI
- **v2.0 (в процессе):** Fine-tuned Qwen2.5-1.5B на KazQAD, переключение моделей
- **v3.0:** Whisper для распознавания казахской речи (ASR)
- **v4.0:** Мобильное приложение (React Native)
- **v5.0:** Расширение датасета (legal/medical/edu domains)

## Источники

- **KazQAD Dataset.** ISSAI, Nazarbayev University. [huggingface.co/datasets/issai/kazqad](https://huggingface.co/datasets/issai/kazqad)
- **KazLLM.** ISSAI, Nazarbayev University. [issai.nu.edu.kz/kazllm](https://issai.nu.edu.kz/kazllm/)
- **Qwen2.5 Technical Report.** Qwen Team, 2025. arXiv:2412.15115
- **MLX framework.** Apple ML Research. [github.com/ml-explore/mlx](https://github.com/ml-explore/mlx)
- **Spring AI / WebFlux.** [docs.spring.io/spring-ai](https://docs.spring.io/spring-ai/)

## Лицензия

MIT.
