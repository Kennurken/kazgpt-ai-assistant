"""Детектор зацикливания (повторяющихся n-грамм).

Главный симптом проблемного fine-tune у KazGPT V2 — модель повторяет
одну и ту же n-грамму много раз («12-юш 12-юш 12-юш...»).

Алгоритм:
1. Разбиваем ответ на слова (нижний регистр, без пунктуации).
2. Скользящим окном собираем все n-граммы.
3. Считаем самую частую n-грамму.
4. Если она повторяется > threshold раз — это loop.
5. Возвращаем repetition_rate = max_count / total_ngrams.

Метрика возвращается в диапазоне [0.0, 1.0]:
  0.0 — все n-граммы уникальны (идеально)
  1.0 — весь ответ — одна повторяющаяся фраза (катастрофа)

Целевое значение для production KazGPT: repetition_rate < 0.10.
"""

import re
from collections import Counter
from typing import Dict


def _tokenize(text: str) -> list:
    """Нормализуем: lowercase, выкидываем пунктуацию, режем по пробелам."""
    text = text.lower()
    text = re.sub(r"[^\w\sЀ-ӿӒӓӨөҮүҚқҢңҺһҒғІі]+", " ", text, flags=re.UNICODE)
    return [tok for tok in text.split() if tok]


def detect_loop(text: str, n: int = 4, threshold: int = 3) -> Dict:
    """Анализирует текст на повторы n-грамм.

    Args:
        text: ответ модели
        n: размер n-граммы (4 = четыре слова подряд)
        threshold: если самая частая n-грамма встречается >=threshold раз — это loop

    Returns:
        dict с полями:
          - is_loop: bool
          - repetition_rate: float в [0, 1]  (всегда 0.0 если слов меньше чем 2n — мало данных)
          - most_common_ngram: str (для отладки)
          - max_count: int
    """
    tokens = _tokenize(text)
    # Минимум 2*n токенов — иначе один уникальный n-gram даст rate=1.0, что вводит в заблуждение
    if len(tokens) < 2 * n:
        return {
            "is_loop": False,
            "repetition_rate": 0.0,
            "most_common_ngram": "",
            "max_count": 0,
        }

    ngrams = [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    counter = Counter(ngrams)
    most_common_ngram, max_count = counter.most_common(1)[0]
    repetition_rate = max_count / len(ngrams)

    return {
        # threshold=3 → 3 и более повторов это loop (3× — уже явно зацикленность)
        "is_loop": max_count >= threshold,
        "repetition_rate": round(repetition_rate, 3),
        "most_common_ngram": most_common_ngram,
        "max_count": max_count,
    }


if __name__ == "__main__":
    # Само-тест: проблемные случаи из V2
    cases = [
        ("Сәлеметсіз! Мен KazGPT көмекшісімін.", "OK — нет повторов"),
        ("12-юш 12-юш 12-юш 12-юш 12-юш", "LOOP — повтор 12-юш"),
        ("қаласын қолдануға қаласын қолдануға қаласын қолдануға", "LOOP — повтор фразы"),
        ("", "Empty"),
    ]
    for text, label in cases:
        r = detect_loop(text)
        print(f"[{label}] is_loop={r['is_loop']}, rate={r['repetition_rate']}, ngram='{r['most_common_ngram']}'")
