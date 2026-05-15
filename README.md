# KazGPT — Local AI Assistant for the Kazakh Language

> Built by a Kazakhstani developer, for Kazakhstani people. Runs fully offline. No cloud, no data leaks, no subscriptions.

---

## What's This?

Kazakh is one of those languages that big AI companies basically ignore. GPT-4, Gemini, Claude — they all stumble on Kazakh grammar, mix it with Russian, or just give up. That's the problem KazGPT tries to solve.

**KazGPT** is a locally-running AI assistant that speaks Kazakh fluently. It's built on top of Qwen2.5, fine-tuned with [KazQAD](https://huggingface.co/datasets/issai/kazqad) (a Kazakh QA dataset from ISSAI / Nazarbayev University), and packaged into a clean Spring Boot + vanilla JS web app. Everything runs on your machine — no internet required after setup.

Current release: **V5** — QLoRA fine-tuned Qwen2.5-7B, 4.4 GB GGUF, runs via Ollama on any Mac (M1/M2/M3 = fast, Intel = works).

---

## Architecture

```
┌────────────────────────────────────────────┐
│  Browser (vanilla JS, localStorage)        │
│  • Streaming chat UI                       │
│  • Model switcher (base ↔ v2)             │
└────────────┬───────────────────────────────┘
             │ HTTP / SSE
             ▼
┌────────────────────────────────────────────┐
│  Spring Boot Backend (port 8080)           │
│  • POST /api/chat/stream  (SSE)           │
│  • GET  /api/health, /api/models          │
│  • Demo-cache fallback                    │
└─────┬─────────────────────────┬────────────┘
      │                         │
      ▼                         ▼
┌──────────────────┐    ┌──────────────────────┐
│ Ollama (11434)   │    │ mlx-lm server (11435)│
│ qwen2.5:7b       │    │ Qwen2.5-7B + QLoRA   │
└──────────────────┘    └──────────────────────┘
```

---

## Tech Stack

| Layer | Tech | Why |
|-------|------|-----|
| Backend | Java 17 + Spring Boot 3.5 | Solid, reactive streams, SSE support out of the box |
| Streaming | WebClient + SSE | Non-blocking, perfect for LLM token streaming |
| Frontend | Vanilla JS | Zero dependencies, zero build step |
| Base LLM | Qwen2.5:7b via Ollama | Strong Kazakh baseline, runs on consumer hardware |
| Fine-tune | QLoRA + MLX | Native Apple Silicon support, fits in 16 GB RAM |
| Dataset | KazQAD (ISSAI, NU) | Open, high-quality Kazakh QA pairs (~16k examples) |

---

## Quick Start

```bash
# 1. Install dependencies
brew install openjdk@17 maven ollama

# 2. Start Ollama + pull base model
brew services start ollama
ollama pull qwen2.5:7b

# 3. Run the backend
cd backend && mvn spring-boot:run

# Open http://localhost:8080 — that's it!
```

### Want to use the fine-tuned V5 model?

```bash
# Download KazGPT-v5-Mac.tar (see Releases)
cd KazGPT-v5-Mac
./setup.sh   # registers the model in Ollama (one-time)
./start.sh   # launches everything + opens browser
```

### Want to train your own fine-tune?

```bash
cd ml
pip install -r requirements.txt
./train.sh   # ~40 min on M2 16GB
```

---

## API

### `POST /api/chat/stream`
Streaming chat endpoint — returns `text/event-stream`.

```json
{
  "message": "Алматы туралы айтып бер",
  "history": [{ "role": "user", "content": "..." }, { "role": "assistant", "content": "..." }],
  "model": "base"
}
```

### `GET /api/health`
```json
{
  "status": "ok",
  "version": "5.0",
  "ollamaUp": true,
  "mlxServerUp": false,
  "demoMode": false,
  "uptimeSeconds": 142
}
```

### `GET /api/models`
Returns available models and current default.

---

## Model Details (V5)

| Parameter | Value |
|-----------|-------|
| Base model | Qwen2.5-7B-Instruct |
| Fine-tuning | QLoRA (r=64, α=128) |
| Trainable params | 161M / 7.77B (2.08%) |
| Quantization | Q4_K_M — 4.4 GB |
| Language | Kazakh only |
| Temperature | 0.27 (conservative, anti-hallucination) |
| Context | 4096 tokens |
| Dataset | KazQAD — ~16k examples (ISSAI, NU) |

---

## Roadmap

- ✅ **V1** — Spring Boot backend + Ollama base model + streaming UI
- ✅ **V2** — Fine-tuned Qwen2.5-1.5B on KazQAD + model switcher
- ✅ **V3** — QLoRA fine-tune on full Qwen2.5-7B
- ✅ **V4** — GGUF quantization + Ollama packaging
- ✅ **V5** — Portable Mac release (tar bundle, setup + start scripts)
- 🔜 **V6** — Kazakh speech recognition (Whisper ASR integration)
- 🔜 **V7** — Mobile app (React Native)
- 🔜 **V8** — Domain expansion (legal / medical / education)

---

## References

- **KazQAD Dataset.** ISSAI, Nazarbayev University. [huggingface.co/datasets/issai/kazqad](https://huggingface.co/datasets/issai/kazqad)
- **KazLLM.** ISSAI, Nazarbayev University. [issai.nu.edu.kz/kazllm](https://issai.nu.edu.kz/kazllm/)
- **Qwen2.5 Technical Report.** Qwen Team, 2025. arXiv:2412.15115
- **MLX Framework.** Apple ML Research. [github.com/ml-explore/mlx](https://github.com/ml-explore/mlx)
- **Spring WebFlux.** [docs.spring.io](https://docs.spring.io/spring-framework/docs/current/reference/html/web-reactive.html)

---

## Author & Copyright

**Қыдырбек Елдос (Eldos Kydyrbek)**
ML/LLM Developer — Kazakhstan 🇰🇿
Master's student, Korkyt Ata Kyzylorda University

All rights reserved. The KazGPT name, fine-tuned model weights, training pipeline, and web application code in this repository are the intellectual property of **Eldos Kydyrbek**, created as an original academic and research project.

Co-contributor: **Temirlan Saduaqas**

---

## License

MIT License — free to use, modify, and distribute with attribution.

```
Copyright (c) 2025–2026 Eldos Kydyrbek (Қыдырбек Елдос)
```

> If you use KazGPT in your research or product, please credit the author.
