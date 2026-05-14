"""Автоматические лексические и семантические метрики.

- BLEU-4: ngram precision против эталонов (sacrebleu)
- ROUGE-L: longest common subsequence (rouge-score)
- BERTScore: семантическое сходство через multilingual BERT
  (paraphrase-multilingual-MiniLM-L12-v2 — поддерживает казахский)

Дизайн:
- Импорты ленивые → файл импортируется даже без установленных пакетов,
  отдельные метрики просто помечаются "unavailable".
- Все метрики принимают (prediction, references) где references — список валидных эталонов.
  Возвращают лучший скор по всем эталонам (так делают принятые бенчмарки).

Особенности для казахского:
- BLEU-4 для kz-языка часто очень низкий (морфология агглютинативная) — это нормально.
  Главная метрика всё-таки BERTScore и LLM-judge, BLEU — sanity check.
"""

from typing import Dict, List, Optional


def bleu_score(prediction: str, references: List[str]) -> Optional[float]:
    """BLEU-4 против лучшего эталона. Возвращает 0-100 или None если sacrebleu не установлен."""
    try:
        import sacrebleu
    except ImportError:
        return None

    if not prediction or not references:
        return 0.0

    # sacrebleu принимает list of list of references (corpus-level), мы передаём один пример
    best = 0.0
    for ref in references:
        result = sacrebleu.sentence_bleu(prediction, [ref])
        best = max(best, result.score)
    return round(best, 2)


def rouge_l_score(prediction: str, references: List[str]) -> Optional[float]:
    """ROUGE-L F1 против лучшего эталона. Возвращает 0-1 или None."""
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        return None

    if not prediction or not references:
        return 0.0

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    best = 0.0
    for ref in references:
        s = scorer.score(ref, prediction)["rougeL"].fmeasure
        best = max(best, s)
    return round(best, 3)


# BERTScore модель загружается один раз и кешируется
_bertscore_model = None


def bertscore(prediction: str, references: List[str]) -> Optional[float]:
    """BERTScore F1 (multilingual) против лучшего эталона. Тяжёлый: загружает модель ~250MB."""
    global _bertscore_model
    try:
        from bert_score import BERTScorer
    except ImportError:
        return None

    if not prediction or not references:
        return 0.0

    if _bertscore_model is None:
        # Multilingual MiniLM хорошо поддерживает казахский и быстрее обычного BERT
        _bertscore_model = BERTScorer(
            model_type="microsoft/Multilingual-MiniLM-L12-H384",
            num_layers=12,
        )

    # bert_score принимает батчи
    P, R, F1 = _bertscore_model.score([prediction] * len(references), references)
    return round(float(F1.max()), 3)


def assertions_check(
    prediction: str,
    must_contain_any: List[str],
    must_not_contain: List[str],
) -> Dict:
    """Простой rule-based чек: содержит ли ответ нужные ключевые слова и не содержит ли запретных.

    Главная польза для KazGPT — на factual questions ловит явные галлюцинации.
    Пример: на вопрос «столица Казахстана?» ответ должен содержать «Астана»
    и не должен называть Алматы столицей.
    """
    pred_lower = prediction.lower()

    contains_required = True
    if must_contain_any:
        contains_required = any(kw.lower() in pred_lower for kw in must_contain_any)

    has_forbidden = []
    for kw in must_not_contain or []:
        if kw.lower() in pred_lower:
            has_forbidden.append(kw)

    return {
        "passed": contains_required and not has_forbidden,
        "contains_required": contains_required,
        "forbidden_found": has_forbidden,
    }


def compute_all(
    prediction: str,
    references: List[str],
    must_contain_any: Optional[List[str]] = None,
    must_not_contain: Optional[List[str]] = None,
    skip_bertscore: bool = False,
) -> Dict:
    """Считает все автоматические метрики за один проход.

    skip_bertscore=True ускоряет CI (BERTScore ~1s на пример из-за GPU/CPU модели).
    """
    return {
        "bleu": bleu_score(prediction, references),
        "rouge_l": rouge_l_score(prediction, references),
        "bertscore": None if skip_bertscore else bertscore(prediction, references),
        "assertions": assertions_check(prediction, must_contain_any or [], must_not_contain or []),
    }


if __name__ == "__main__":
    pred = "Қазақстанның астанасы — Астана қаласы."
    refs = ["Астана — Қазақстанның астанасы.", "Қазақстанның астанасы Астана."]
    print(compute_all(pred, refs, must_contain_any=["Астана"], must_not_contain=["Нұр-Сұлтан"]))
