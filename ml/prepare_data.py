"""Подготовка казахских данных для MLX LoRA fine-tune.

Скачивает KazQAD от ISSAI (Nazarbayev University), конвертирует в формат
MLX LoRA (jsonl с полями prompt/completion), разбивает на train/valid/test.

Если KazQAD недоступен — фоллбек на multidomain-kazakh-dataset.
"""

import json
import random
import sys
from pathlib import Path

random.seed(42)

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

TRAIN_LIMIT = 500
VALID_LIMIT = 50
TEST_LIMIT = 50


def try_load_kazqad():
    from datasets import load_dataset
    print("=> Попытка загрузки issai/kazqad ...")
    try:
        ds = load_dataset("issai/kazqad", split="train")
        records = []
        for item in ds:
            q = item.get("question") or item.get("instruction")
            a = item.get("answers") or item.get("answer") or item.get("output")
            if isinstance(a, dict):
                texts = a.get("text") or []
                a = texts[0] if texts else None
            elif isinstance(a, list) and a:
                a = a[0]
            if q and a:
                records.append({"prompt": q.strip(), "completion": " " + str(a).strip()})
        return records
    except Exception as e:
        print(f"   KazQAD недоступен: {e}")
        return None


def try_load_multidomain():
    from datasets import load_dataset
    print("=> Фоллбек: kz-transformers/multidomain-kazakh-dataset ...")
    try:
        ds = load_dataset("kz-transformers/multidomain-kazakh-dataset", split="train", streaming=True)
        records = []
        for i, item in enumerate(ds):
            if i >= 1000:
                break
            text = item.get("text") or item.get("content") or ""
            if not text or len(text) < 80:
                continue
            words = text.split()
            if len(words) < 30:
                continue
            half = len(words) // 2
            prompt = " ".join(words[:half])
            completion = " " + " ".join(words[half:half + 100])
            records.append({"prompt": prompt, "completion": completion})
        return records
    except Exception as e:
        print(f"   Фоллбек тоже не сработал: {e}")
        return None


def add_manual_examples():
    """Качественные ручные примеры для дообучения стиля KazGPT."""
    return [
        {"prompt": "Сәлем", "completion": " Сәлеметсіз! Мен KazGPT көмекшісімін. Сізге қалай көмектесе аламын?"},
        {"prompt": "Өзің туралы айт", "completion": " Мен KazGPT — қазақ тілінде сөйлейтін жасанды интеллект көмекшісімін."},
        {"prompt": "Алматы туралы айт", "completion": " Алматы — Қазақстанның ең үлкен қаласы, мәдени орталығы. Іле Алатауының баурайында орналасқан."},
        {"prompt": "Қазақстанның астанасы", "completion": " Қазақстанның астанасы — Астана қаласы. 1997 жылдан бастап астана."},
        {"prompt": "Нейрондық желі дегеніміз не?", "completion": " Нейрондық желі — адам миының жұмысын үлгілейтін жасанды интеллект моделі."},
    ]


def split_and_save(records):
    random.shuffle(records)
    n = len(records)

    if n >= TRAIN_LIMIT + VALID_LIMIT + TEST_LIMIT:
        train = records[:TRAIN_LIMIT]
        valid = records[TRAIN_LIMIT:TRAIN_LIMIT + VALID_LIMIT]
        test = records[TRAIN_LIMIT + VALID_LIMIT:TRAIN_LIMIT + VALID_LIMIT + TEST_LIMIT]
    else:
        v_size = max(20, n // 10)
        t_size = max(20, n // 10)
        valid = records[:v_size]
        test = records[v_size:v_size + t_size]
        train = records[v_size + t_size:]

    for name, data in [("train", train), ("valid", valid), ("test", test)]:
        path = DATA_DIR / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for rec in data:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"   {path.name}: {len(data)} examples")

    print(f"\n=> 3 случайных примера из train:")
    for i, rec in enumerate(random.sample(train, min(3, len(train))), 1):
        p = rec["prompt"][:80] + ("..." if len(rec["prompt"]) > 80 else "")
        c = rec["completion"][:80] + ("..." if len(rec["completion"]) > 80 else "")
        print(f"   [{i}] PROMPT: {p}")
        print(f"       COMPLETION: {c}")


def main():
    print("KazGPT — Подготовка данных для fine-tune")
    print("=" * 50)

    records = try_load_kazqad()
    if not records:
        records = try_load_multidomain()
    if not records:
        print("ERROR: Не удалось загрузить ни один датасет.")
        sys.exit(1)

    records.extend(add_manual_examples() * 10)

    print(f"\n=> Всего записей: {len(records)}")
    print(f"=> Сохраняем в: {DATA_DIR}")

    split_and_save(records)

    print("\n=> Готово.")


if __name__ == "__main__":
    main()
