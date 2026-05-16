#!/usr/bin/env python3
"""
KazGPT — Финальный фильтр и сплиттер датасета

Вход:  ml/data/kazgpt_full_raw.jsonl   (собрал collect_all.py)
Выход: ml/data/train.jsonl
       ml/data/valid.jsonl
       ml/data/test.jsonl              (опционально)

Запуск:
    python filter_dataset.py
    python filter_dataset.py --input data/kazgpt_full_raw.jsonl --valid-ratio 0.02
    python filter_dataset.py --stats-only   # только статистика
"""

import json, re, hashlib, random, argparse, sys
from pathlib import Path
from datetime import datetime
from collections import Counter

random.seed(42)
ML_DIR = Path(__file__).parent
DATA_DIR = ML_DIR / "data"

# ── Казахские буквы (маркер языка) ───────────────────────────────
KK_CHARS = set('әғқңөұүһіӘҒҚҢӨҰҮҺІ')

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ══════════════════════════════════════════════════════════════════
# КАЧЕСТВО
# ══════════════════════════════════════════════════════════════════

def kk_ratio(text: str) -> float:
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return 0.0
    return sum(1 for c in alpha if c in KK_CHARS) / len(alpha)

def word_count(text: str) -> int:
    return len(text.split())

def has_repetition(text: str, ngram=5, max_repeat=3) -> bool:
    """Обнаруживает повторяющиеся n-граммы (признак вырожденных ответов)."""
    words = text.split()
    if len(words) < ngram * 2:
        return False
    ngrams = [' '.join(words[i:i+ngram]) for i in range(len(words)-ngram+1)]
    counts = Counter(ngrams)
    return counts.most_common(1)[0][1] > max_repeat

def is_good(record: dict, cfg: dict) -> tuple[bool, str]:
    """Возвращает (True, '') или (False, 'причина')."""
    prompt = (record.get("prompt") or "").strip()
    compl  = (record.get("completion") or "").strip()

    if not prompt or not compl:
        return False, "empty"

    # длина completion
    if len(compl) < cfg["min_len"]:
        return False, "too_short"
    if len(compl) > cfg["max_len"]:
        compl = compl[:cfg["max_len"]]  # обрезаем — не отбрасываем

    # слова
    if word_count(compl) < cfg["min_words"]:
        return False, "few_words"

    # казахский язык
    combined = prompt + " " + compl
    if kk_ratio(combined) < cfg["min_kk"]:
        return False, "not_kazakh"

    # HTML мусор
    if re.search(r'<[a-z]{1,10}[\s/>]', compl, re.I):
        return False, "html"

    # повторы
    if has_repetition(compl):
        return False, "repetition"

    # слишком много пунктуации (спам)
    punct = sum(1 for c in compl if c in '!?.,;:')
    if len(compl) > 0 and punct / len(compl) > 0.25:
        return False, "punct_spam"

    return True, ""

# ══════════════════════════════════════════════════════════════════
# ДЕДУПЛИКАЦИЯ
# ══════════════════════════════════════════════════════════════════

def fingerprint(record: dict) -> str:
    text = (record.get("prompt","") + record.get("completion","")).lower()
    text = re.sub(r'\s+', ' ', text)
    return hashlib.md5(text.encode()).hexdigest()

def deduplicate(records: list) -> list:
    seen = set()
    out = []
    for r in records:
        fp = fingerprint(r)
        if fp not in seen:
            seen.add(fp)
            out.append(r)
    return out

# ══════════════════════════════════════════════════════════════════
# НОРМАЛИЗАЦИЯ
# ══════════════════════════════════════════════════════════════════

def normalize(record: dict) -> dict:
    """Приводим к единому формату MLX LoRA: prompt + completion."""
    p = re.sub(r'\s+', ' ', (record.get("prompt") or "").strip())
    c = re.sub(r'\s+', ' ', (record.get("completion") or "").strip())

    # MLX LoRA ожидает completion начинающийся с пробела
    if c and not c.startswith(' '):
        c = ' ' + c

    return {
        "prompt":     p,
        "completion": c,
        "source":     record.get("source", "unknown")
    }

# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="KazGPT dataset filter + split")
    p.add_argument("--input",       default=str(DATA_DIR / "kazgpt_full_raw.jsonl"))
    p.add_argument("--out-dir",     default=str(DATA_DIR))
    p.add_argument("--valid-ratio", type=float, default=0.02,  help="Доля valid (default 2%)")
    p.add_argument("--test-ratio",  type=float, default=0.01,  help="Доля test (default 1%)")
    p.add_argument("--min-len",     type=int,   default=30,    help="Мин символов completion")
    p.add_argument("--max-len",     type=int,   default=6000,  help="Макс символов completion")
    p.add_argument("--min-words",   type=int,   default=8,     help="Мин слов completion")
    p.add_argument("--min-kk",      type=float, default=0.03,  help="Мин доля казахских букв")
    p.add_argument("--stats-only",  action="store_true",        help="Только статистика, без записи")
    p.add_argument("--no-dedup",    action="store_true",        help="Пропустить дедупликацию")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = {
        "min_len":   args.min_len,
        "max_len":   args.max_len,
        "min_words": args.min_words,
        "min_kk":    args.min_kk,
    }

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        log(f"❌ Файл не найден: {in_path}")
        log("   Сначала запусти: python collect_all.py")
        sys.exit(1)

    # ── Загрузка ─────────────────────────────────────────────────
    log(f"📂 Загрузка: {in_path}")
    raw = []
    with open(in_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    log(f"   Загружено: {len(raw):,} записей")

    # ── Статистика по источникам ──────────────────────────────────
    sources = Counter(r.get("source","?") for r in raw)
    log("\n📊 Источники (до фильтрации):")
    for src, cnt in sources.most_common():
        bar = '█' * min(40, cnt // max(1, max(sources.values()) // 40))
        log(f"  {src:<35} {cnt:>8,}  {bar}")

    # ── Фильтрация ────────────────────────────────────────────────
    log("\n🔍 Фильтрация...")
    reject_reasons: Counter = Counter()
    good = []
    for r in raw:
        ok, reason = is_good(r, cfg)
        if ok:
            good.append(normalize(r))
        else:
            reject_reasons[reason] += 1

    log(f"   Прошли фильтр: {len(good):,} / {len(raw):,} ({100*len(good)/max(1,len(raw)):.1f}%)")
    if reject_reasons:
        log("   Отклонено:")
        for reason, cnt in reject_reasons.most_common():
            log(f"     {reason:<20} {cnt:>7,}")

    if args.stats_only:
        log("\n--stats-only: запись пропущена.")
        return

    # ── Дедупликация ──────────────────────────────────────────────
    if not args.no_dedup:
        before = len(good)
        good = deduplicate(good)
        log(f"\n🗑️  Дедупликация: {before:,} → {len(good):,} (удалено {before-len(good):,})")

    # ── Перемешать ────────────────────────────────────────────────
    random.shuffle(good)
    total = len(good)

    # ── Split ─────────────────────────────────────────────────────
    n_valid = max(100, int(total * args.valid_ratio))
    n_test  = max(50,  int(total * args.test_ratio))
    n_train = total - n_valid - n_test

    train = good[:n_train]
    valid = good[n_train:n_train + n_valid]
    test  = good[n_train + n_valid:]

    # ── Запись ───────────────────────────────────────────────────
    splits = {"train": train, "valid": valid, "test": test}
    log("\n💾 Запись сплитов:")
    for name, records in splits.items():
        path = out_dir / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        log(f"   {name:<8} {len(records):>8,}  → {path}")

    # ── Финальная статистика ──────────────────────────────────────
    log(f"\n{'='*55}")
    log(f"✅ ГОТОВО")
    log(f"{'='*55}")
    log(f"  Всего примеров : {total:,}")
    log(f"  train          : {len(train):,}")
    log(f"  valid          : {len(valid):,}")
    log(f"  test           : {len(test):,}")

    avg_prompt = sum(len(r["prompt"]) for r in good) / max(1, total)
    avg_compl  = sum(len(r["completion"]) for r in good) / max(1, total)
    log(f"  Ср. длина prompt     : {avg_prompt:.0f} символов")
    log(f"  Ср. длина completion : {avg_compl:.0f} символов")
    log(f"{'='*55}")

    log("\n🚀 Теперь можно тренировать:")
    log("   mlx_lm.lora --model <model> --train --data ml/data \\")
    log("               --iters 2000 --batch-size 4 --lora-layers 16")


if __name__ == "__main__":
    main()
