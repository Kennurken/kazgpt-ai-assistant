"""LLM-as-judge: используем GPT-4 (или Claude) как арбитра качества.

Стандартный подход в индустрии (см. MT-Bench, AlpacaEval, Anthropic's Constitutional AI):
- Сильная модель оценивает ответы слабой по структурированному промпту
- Возвращает числовой скор + объяснение
- Усреднение по golden set даёт стабильную метрику качества

Поддерживаемые провайдеры:
- OpenAI (gpt-4, gpt-4o, gpt-4o-mini)
- Anthropic (claude-3-5-sonnet) — если установлен anthropic SDK

Ключи берутся из env: OPENAI_API_KEY или ANTHROPIC_API_KEY.

ВАЖНО: метрика медленная (1-3 сек на пример) и платная.
В eval_config.yaml включать через `enable_llm_judge: true` только для важных прогонов.

============================================================
TODO ДЛЯ ПОЛЬЗОВАТЕЛЯ — здесь твоя экспертиза в казахском языке
определит, КАК будут оцениваться ВСЕ будущие эксперименты.
============================================================

Заполни константу JUDGE_PROMPT ниже. Это промпт, который GPT-4 увидит
для каждого ответа KazGPT. Подумай о таких вопросах:

  1. По каким критериям ChatGPT-как-судья должен оценивать ответ KazGPT?
     Варианты (можно комбинировать):
       a) Coherence (связность, отсутствие циклов)
       b) Fluency (естественность казахского)
       c) Correctness (фактическая правильность)
       d) Helpfulness (полезность для пользователя)
       e) Brevity (краткость, нет «воды»)
       f) Cultural appropriateness (казахские реалии, тон, вежливость)

  2. Какая шкала? Распространённые: 1-5 (Likert) или 1-10.
     1-5 проще, 1-10 даёт больше разрешения, но судьи путаются.

  3. Что важнее в твоём конкретном проекте?
     - Защита перед куратором → больше веса fluency и cultural
     - Production-стандарт → больше веса correctness и helpfulness

  4. Должен ли судья давать объяснение или только число?
     Объяснение полезно для отладки, но удваивает стоимость.

Контракт: промпт ДОЛЖЕН возвращать JSON в формате
    {"score": <число>, "reasoning": "<строка>"}

иначе parse_judge_response() выдаст ошибку.

Пример минимального промпта (тебе нужно лучше):

    JUDGE_PROMPT = '''
    You are an expert evaluator of Kazakh-language AI responses.
    Score the candidate answer from 1 to 5...
    [твоя версия здесь]
    Return JSON: {"score": int, "reasoning": str}
    '''

5-10 строк казахо-ориентированного prompt-инжиниринга — и метрика готова.
"""

import json
import os
import re
from typing import Dict, List, Optional

# ============================================================
# >>> TODO: заполни JUDGE_PROMPT ниже (см. большой комментарий выше)
# Это место, где твоя экспертиза в казахском языке определит,
# как будут сравниваться все будущие версии KazGPT.
# ============================================================
JUDGE_PROMPT = """
You are an expert evaluator of Kazakh-language AI assistant responses.

Question: {question}

Reference answer(s) (one or more valid forms):
{references}

Candidate answer from the model:
{candidate}

TODO USER: define your evaluation criteria here.
Until you fill this in, the LLM judge returns score=0.
"""


def _format_prompt(question: str, candidate: str, references: List[str]) -> str:
    refs_block = "\n".join(f"- {r}" for r in references)
    return JUDGE_PROMPT.format(question=question, candidate=candidate, references=refs_block)


def _call_openai(prompt: str, model: str = "gpt-4o-mini") -> Optional[str]:
    """Вызов OpenAI ChatCompletion. Возвращает текст ответа или None при ошибке."""
    try:
        from openai import OpenAI
    except ImportError:
        return None

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    client = OpenAI(api_key=api_key)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,  # для воспроизводимости
            max_tokens=400,
        )
        return resp.choices[0].message.content
    except Exception as e:
        print(f"[llm_judge] OpenAI error: {e}")
        return None


def _call_anthropic(prompt: str, model: str = "claude-3-5-sonnet-latest") -> Optional[str]:
    """Опциональный бэкенд — Claude от Anthropic."""
    try:
        from anthropic import Anthropic
    except ImportError:
        return None

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    client = Anthropic(api_key=api_key)
    try:
        resp = client.messages.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.0,
        )
        return resp.content[0].text if resp.content else None
    except Exception as e:
        print(f"[llm_judge] Anthropic error: {e}")
        return None


def _parse_judge_response(raw: str) -> Dict:
    """Извлекает JSON из ответа судьи. Терпим к лишнему тексту вокруг JSON."""
    # Ищем первый {...} блок
    match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if not match:
        return {"score": 0.0, "reasoning": f"[parse error] no JSON found in: {raw[:120]}"}
    try:
        data = json.loads(match.group(0))
        score = float(data.get("score", 0))
        reasoning = str(data.get("reasoning", ""))
        return {"score": score, "reasoning": reasoning}
    except (json.JSONDecodeError, ValueError) as e:
        return {"score": 0.0, "reasoning": f"[parse error] {e}: {raw[:120]}"}


def judge(
    question: str,
    candidate: str,
    references: List[str],
    provider: str = "openai",
    model: Optional[str] = None,
) -> Dict:
    """Просит LLM оценить candidate-ответ относительно references.

    Returns:
        {"score": float, "reasoning": str, "provider": str, "model": str}
    """
    # Защита: если JUDGE_PROMPT не заполнен пользователем, скоринг отключаем
    if "TODO USER:" in JUDGE_PROMPT:
        return {
            "score": 0.0,
            "reasoning": "[disabled] JUDGE_PROMPT not configured. See llm_judge.py top comment.",
            "provider": provider,
            "model": model or "n/a",
        }

    prompt = _format_prompt(question, candidate, references)

    if provider == "openai":
        model = model or "gpt-4o-mini"
        raw = _call_openai(prompt, model)
    elif provider == "anthropic":
        model = model or "claude-3-5-sonnet-latest"
        raw = _call_anthropic(prompt, model)
    else:
        return {"score": 0.0, "reasoning": f"unknown provider: {provider}", "provider": provider, "model": "n/a"}

    if raw is None:
        return {"score": 0.0, "reasoning": "[api unavailable or key missing]", "provider": provider, "model": model}

    parsed = _parse_judge_response(raw)
    parsed["provider"] = provider
    parsed["model"] = model
    return parsed


if __name__ == "__main__":
    r = judge(
        "Қазақстанның астанасы қандай?",
        "Қазақстанның астанасы — Астана.",
        ["Астана — Қазақстанның астанасы."],
    )
    print(r)
