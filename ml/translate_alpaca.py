#!/usr/bin/env python3
"""
KazGPT — Переводчик Alpaca 52k → казахский

Скачивает stanford_alpaca и переводит instruction/output на казахский
через deep-translator (Google Translate API, бесплатный эндпоинт).

Запуск:
    python translate_alpaca.py
    python translate_alpaca.py --limit 5000       # первые 5000 пар
    python translate_alpaca.py --limit 500 --fast # тест
    python translate_alpaca.py --workers 3        # параллельно

ВАЖНО:
  - Не перегружай Google API: вшит rate-limit (1.5 сек / запрос)
  - При ошибке перевода запись пропускается
  - Результат → ml/data/alpaca_kk.jsonl
"""

import json, time, argparse, random, sys
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

random.seed(42)
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = DATA_DIR / "alpaca_kk.jsonl"

# ── Обёртка переводчика ───────────────────────────────────────────
def get_translator():
    try:
        from deep_translator import GoogleTranslator
        return GoogleTranslator(source="en", target="kk")
    except ImportError:
        print("❌ Установи: pip install deep-translator")
        sys.exit(1)

def translate(text: str, tr, max_retries=3) -> str | None:
    """Переводим текст с en→kk, возвращаем None при ошибке."""
    text = text.strip()
    if not text:
        return ""
    for attempt in range(max_retries):
        try:
            result = tr.translate(text)
            return result
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return None

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

# ══════════════════════════════════════════════════════════════════
# ЗАГРУЗКА ALPACA
# ══════════════════════════════════════════════════════════════════

def load_alpaca_hf(limit: int) -> list[dict]:
    """Грузим tatsu-lab/alpaca через HuggingFace datasets."""
    log("📦 Загрузка tatsu-lab/alpaca с HuggingFace...")
    try:
        from datasets import load_dataset
        ds = load_dataset("tatsu-lab/alpaca", split="train")
        records = []
        for item in ds:
            instruction = (item.get("instruction") or "").strip()
            inp         = (item.get("input") or "").strip()
            output      = (item.get("output") or "").strip()
            if not instruction or not output:
                continue
            # Если есть input — дописываем к instruction
            if inp:
                full_instruction = f"{instruction}\n\nInput: {inp}"
            else:
                full_instruction = instruction
            records.append({"instruction": full_instruction, "output": output})
            if len(records) >= limit:
                break
        log(f"   Загружено: {len(records):,} примеров")
        return records
    except Exception as e:
        log(f"❌ Ошибка HF: {e}")
        # Пробуем JSON запасной вариант
        return load_alpaca_json(limit)

def load_alpaca_json(limit: int) -> list[dict]:
    """Запасной вариант — прямой JSON с GitHub."""
    import urllib.request
    url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
    log(f"📦 Загрузка alpaca_data.json с GitHub...")
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            data = json.loads(r.read().decode())
        records = []
        for item in data:
            instruction = (item.get("instruction") or "").strip()
            inp         = (item.get("input") or "").strip()
            output      = (item.get("output") or "").strip()
            if not instruction or not output:
                continue
            if inp:
                instruction = f"{instruction}\n\nInput: {inp}"
            records.append({"instruction": instruction, "output": output})
            if len(records) >= limit:
                break
        log(f"   Загружено: {len(records):,} примеров")
        return records
    except Exception as e:
        log(f"❌ JSON fallback ошибка: {e}")
        return []

# ══════════════════════════════════════════════════════════════════
# ПЕРЕВОД
# ══════════════════════════════════════════════════════════════════

def translate_record(item: dict, tr, delay: float) -> dict | None:
    """Переводим один пример. Возвращает None при ошибке."""
    prompt_kk = translate(item["instruction"], tr)
    time.sleep(delay)
    if not prompt_kk:
        return None

    output_kk = translate(item["output"], tr)
    time.sleep(delay)
    if not output_kk:
        return None

    return {
        "prompt":     prompt_kk.strip(),
        "completion": " " + output_kk.strip(),
        "source":     "alpaca_translated"
    }

def run_translation(records: list, delay: float, workers: int, out_file: Path) -> int:
    """Переводим с checkpoint-записью каждые 50 примеров."""
    tr = get_translator()

    # Проверяем checkpoint — уже переведённые
    done = 0
    if out_file.exists():
        with open(out_file, encoding="utf-8") as f:
            done = sum(1 for l in f if l.strip())
        if done > 0:
            log(f"♻️  Найден checkpoint: {done:,} уже переведено, продолжаем...")
            records = records[done:]

    if not records:
        log("✅ Все примеры уже переведены!")
        return done

    log(f"🔄 Перевод {len(records):,} примеров (задержка {delay:.1f}с, workers={workers})...")

    total_written = done
    batch = []

    def flush(batch):
        nonlocal total_written
        with open(out_file, "a", encoding="utf-8") as f:
            for r in batch:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        total_written += len(batch)
        log(f"   💾 Записано: {total_written:,}")

    if workers == 1:
        # Последовательный перевод (безопаснее для rate limit)
        for i, item in enumerate(records, 1):
            result = translate_record(item, tr, delay)
            if result:
                batch.append(result)
            if len(batch) >= 50:
                flush(batch)
                batch = []
            if i % 100 == 0:
                log(f"   [{i:,}/{len(records):,}] — {total_written:,} переведено")
    else:
        # Параллельный с thread pool
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(translate_record, item, tr, delay): i
                       for i, item in enumerate(records)}
            completed = 0
            for fut in as_completed(futures):
                completed += 1
                result = fut.result()
                if result:
                    batch.append(result)
                if len(batch) >= 50:
                    flush(batch)
                    batch = []
                if completed % 100 == 0:
                    log(f"   [{completed:,}/{len(records):,}]")

    if batch:
        flush(batch)

    return total_written

# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Translate Alpaca 52k → Kazakh")
    p.add_argument("--limit",    type=int,   default=10000, help="Макс примеров (default 10000)")
    p.add_argument("--delay",    type=float, default=1.5,   help="Задержка между запросами (сек)")
    p.add_argument("--workers",  type=int,   default=1,     help="Параллельные потоки")
    p.add_argument("--out",      default=str(OUT_FILE),     help="Выходной файл")
    p.add_argument("--fast",     action="store_true",        help="Тест: лимит 200 примеров")
    return p.parse_args()

def main():
    args = parse_args()
    limit = 200 if args.fast else args.limit
    out   = Path(args.out)

    log("=" * 55)
    log("  KazGPT — Alpaca EN→KK Translator")
    log("=" * 55)
    log(f"  Лимит   : {limit:,}")
    log(f"  Задержка: {args.delay:.1f}с")
    log(f"  Workers : {args.workers}")
    log(f"  Выход   : {out}")
    log("=" * 55)

    # 1. Загрузить Alpaca
    records = load_alpaca_hf(limit)
    if not records:
        log("❌ Не удалось загрузить данные")
        sys.exit(1)

    # 2. Перемешать для разнообразия
    random.shuffle(records)

    # 3. Перевести
    total = run_translation(records, args.delay, args.workers, out)

    log(f"\n{'='*55}")
    log(f"✅ ГОТОВО: {out}")
    log(f"   Переведено примеров: {total:,}")
    log(f"{'='*55}")
    log("\n➕ Добавь alpaca_kk.jsonl в collect_all.py или merge вручную:")
    log(f"   cat ml/data/kazgpt_full_raw.jsonl ml/data/alpaca_kk.jsonl > ml/data/merged.jsonl")
    log("   Затем запусти filter_dataset.py на merged.jsonl")


if __name__ == "__main__":
    main()
