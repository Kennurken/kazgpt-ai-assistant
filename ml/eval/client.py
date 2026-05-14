"""HTTP/SSE клиент для бэкенда KazGPT.

Делает запрос к /api/chat/stream, возвращает полный ответ + замеры времени.
Параллельно собирает: time-to-first-token (TTFT), total time, raw tokens.

Используется во всех метриках через единый интерфейс ChatResponse.
"""

import json
import time
import urllib.request
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ChatResponse:
    text: str = ""
    ttft_ms: Optional[float] = None  # time to first token (милисекунды)
    total_ms: float = 0.0
    token_count: int = 0  # грубо: количество SSE-чанков
    raw_chunks: List[str] = field(default_factory=list)
    error: Optional[str] = None


def chat(
    backend_url: str,
    message: str,
    model: str = "base",
    history: Optional[list] = None,
    timeout: float = 90.0,
) -> ChatResponse:
    """Один запрос к KazGPT-бэкенду, SSE-стрим.

    Args:
        backend_url: например http://localhost:8080/api/chat/stream
        message: сообщение пользователя
        model: 'base' (Ollama 7B) или 'v2' (MLX 1.5B + LoRA) или 'kazgpt' (после fuse)
        history: список сообщений [{role, content}], по умолчанию []
        timeout: жёсткий таймаут в секундах

    Returns:
        ChatResponse — даже при ошибке вернёт объект с полем .error
    """
    history = history or []
    payload = json.dumps(
        {"message": message, "history": history, "model": model}
    ).encode("utf-8")

    req = urllib.request.Request(
        backend_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    resp = ChatResponse()
    start = time.perf_counter()

    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            buf = ""
            while True:
                chunk = r.read(256)
                if not chunk:
                    break
                buf += chunk.decode("utf-8", errors="replace")
                while "\n\n" in buf:
                    event, buf = buf.split("\n\n", 1)
                    for raw in event.split("\n"):
                        if raw.startswith("data:"):
                            token = raw[5:]  # ВАЖНО: без trimStart — пробелы значимы
                            if resp.ttft_ms is None and token.strip():
                                resp.ttft_ms = (time.perf_counter() - start) * 1000
                            resp.text += token
                            resp.token_count += 1
                            resp.raw_chunks.append(token)
    except Exception as e:
        resp.error = f"{type(e).__name__}: {e}"

    resp.total_ms = (time.perf_counter() - start) * 1000
    resp.text = resp.text.strip()
    return resp


if __name__ == "__main__":
    # Быстрый тест: python client.py
    r = chat("http://localhost:8080/api/chat/stream", "Сәлем!", model="base")
    print(f"TTFT: {r.ttft_ms:.0f}ms | Total: {r.total_ms:.0f}ms | Tokens: {r.token_count}")
    print(f"Response: {r.text}")
    if r.error:
        print(f"ERROR: {r.error}")
