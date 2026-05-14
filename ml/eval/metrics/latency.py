"""Агрегатор латентностей по списку ответов.

Берёт ttft_ms и total_ms из ChatResponse (см. eval/client.py) и считает:
- p50, p95, p99 для TTFT (time to first token) — критично для UX
- p50, p95 для total time
- tokens/sec (грубо: token_count / (total_ms / 1000))
"""

from statistics import quantiles, mean
from typing import List, Optional


def percentile(values: List[float], p: float) -> Optional[float]:
    """p ∈ [0, 1]. Возвращает квантиль или None если пусто."""
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    # quantiles делит на N+1 равных частей, индексируется с 0
    qs = quantiles(values, n=100)
    idx = int(p * 100) - 1
    idx = max(0, min(len(qs) - 1, idx))
    return round(qs[idx], 1)


def aggregate(responses) -> dict:
    """responses — list[ChatResponse]."""
    ttfts = [r.ttft_ms for r in responses if r.ttft_ms is not None and not r.error]
    totals = [r.total_ms for r in responses if not r.error]
    tps = [
        r.token_count / (r.total_ms / 1000.0)
        for r in responses
        if r.total_ms > 0 and not r.error
    ]
    errors = sum(1 for r in responses if r.error)

    return {
        "ttft_p50_ms": percentile(ttfts, 0.50),
        "ttft_p95_ms": percentile(ttfts, 0.95),
        "ttft_p99_ms": percentile(ttfts, 0.99),
        "total_p50_ms": percentile(totals, 0.50),
        "total_p95_ms": percentile(totals, 0.95),
        "tokens_per_sec_avg": round(mean(tps), 1) if tps else None,
        "error_count": errors,
        "total_samples": len(responses),
    }
