"""KazGPT V3 — Объединение всех KZ instruction datasets с агрессивной фильтрацией.

Источники (в порядке приоритета качества):
1. AmanMussa/kazakh-instruction-v2 (52k, чистый Alpaca)
2. saillab/alpaca-kazakh-cleaned (52k, ~93% чистый)
3. Наш Wiki+synthetic из ml/data/train.jsonl (4-5k, уже filtered)

Фильтрация (для val_loss 0.3-0.4):
- Длина prompt: 3-2000 токенов (слов)
- Длина output: 5-1500 токенов
- KZ purity >= 0.85 (минимум 85% казахских слов в output)
- Без English noise ("Instruction in English", "Response in English")
- Без HTML/markdown мусора
- Dedup по hash(prompt + output)

Формат output: ChatML JSONL — {"messages": [{"role":"user", "content":...}, {"role":"assistant", "content":...}]}

Использование:
    python ml/merge_datasets.py --output ./ml/data_v3 --train-ratio 0.92 --valid-ratio 0.05
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
import random

random.seed(42)

# Re-use KZ purity logic from eval
sys.path.insert(0, str(Path(__file__).parent / "eval"))
from metrics.kz_purity import kz_purity  # noqa: E402


# Маркеры английского noise (saillab/taco artifacts)
EN_NOISE_PATTERNS = [
    re.compile(r"Instruction\s+in\s+English", re.I),
    re.compile(r"Response\s+in\s+English", re.I),
    re.compile(r"Translation\s*:", re.I),
    re.compile(r"<\|begin_of_text\|>"),  # leftover chat template
    re.compile(r"<\|eot_id\|>"),
    re.compile(r"<\|start_header_id\|>"),
]


def has_english_noise(text: str) -> bool:
    if not text:
        return False
    for pat in EN_NOISE_PATTERNS:
        if pat.search(text):
            return True
    return False


def is_quality_example(prompt: str, output: str, min_kz_purity: float = 0.85) -> tuple:
    """Возвращает (is_good, reason). Reason='' если good."""
    if not prompt or not output:
        return False, "empty"
    if len(prompt.strip()) < 3:
        return False, "prompt_too_short"
    if len(output.strip()) < 10:
        return False, "output_too_short"
    if has_english_noise(prompt) or has_english_noise(output):
        return False, "english_noise"

    p_words = prompt.split()
    o_words = output.split()
    if len(p_words) < 2 or len(o_words) < 3:
        return False, "too_few_words"
    if len(p_words) > 2000 or len(o_words) > 1500:
        return False, "too_long"

    # KZ purity check — output должен быть преимущественно казахским
    purity = kz_purity(output)["purity"]
    if purity < min_kz_purity:
        return False, f"low_kz_purity_{purity:.2f}"

    return True, ""


def to_chatml(prompt: str, output: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": prompt.strip()},
            {"role": "assistant", "content": output.strip()},
        ]
    }


def load_amanmussa():
    """AmanMussa/kazakh-instruction-v2 — лучшее качество, Alpaca format."""
    from datasets import load_dataset
    print("=> Loading AmanMussa/kazakh-instruction-v2 ...", flush=True)
    ds = load_dataset("AmanMussa/kazakh-instruction-v2", split="train")
    records = []
    for ex in ds:
        # Format: {input, output, instruction}
        instr = (ex.get("instruction") or "").strip()
        inp = (ex.get("input") or "").strip()
        out = (ex.get("output") or "").strip()
        if inp and inp not in ("nan", "None"):
            prompt = f"{instr}\n\n{inp}"
        else:
            prompt = instr
        records.append((prompt, out))
    print(f"   raw: {len(records)}", flush=True)
    return records


def load_saillab_cleaned():
    """saillab/alpaca-kazakh-cleaned — ~93% чистый Alpaca KZ."""
    from datasets import load_dataset
    print("=> Loading saillab/alpaca-kazakh-cleaned ...", flush=True)
    ds_train = load_dataset("saillab/alpaca-kazakh-cleaned", split="train")
    ds_test = load_dataset("saillab/alpaca-kazakh-cleaned", split="test")
    records = []
    for ds in (ds_train, ds_test):
        for ex in ds:
            instr = (ex.get("instruction") or "").strip()
            inp = (ex.get("input") or "").strip()
            out = (ex.get("output") or "").strip()
            if inp and inp not in ("nan", "None"):
                prompt = f"{instr}\n\n{inp}"
            else:
                prompt = instr
            records.append((prompt, out))
    print(f"   raw: {len(records)}", flush=True)
    return records


def load_existing(path: Path):
    """Наш Wiki+synthetic из ml/data/train.jsonl (если есть)."""
    if not path.exists():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
                msgs = d.get("messages", [])
                if len(msgs) == 2:
                    records.append((msgs[0]["content"], msgs[1]["content"]))
            except Exception:
                continue
    print(f"=> Existing ml/data/train.jsonl: {len(records)}", flush=True)
    return records


def filter_and_dedup(raw_records, min_kz_purity: float = 0.85):
    """Применяет all-in-one фильтрацию и dedup."""
    stats = {"total": len(raw_records), "passed": 0}
    reasons = {}
    seen = set()
    out = []

    for prompt, output in raw_records:
        ok, reason = is_quality_example(prompt, output, min_kz_purity)
        if not ok:
            reasons[reason] = reasons.get(reason, 0) + 1
            continue

        h = hashlib.md5((prompt.lower() + "||" + output.lower()).encode("utf-8")).hexdigest()
        if h in seen:
            reasons["duplicate"] = reasons.get("duplicate", 0) + 1
            continue
        seen.add(h)

        out.append(to_chatml(prompt, output))
        stats["passed"] += 1

    stats["reasons"] = reasons
    return out, stats


def split_and_save(records, output_dir: Path, train_ratio: float, valid_ratio: float):
    random.shuffle(records)
    n = len(records)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)
    splits = {
        "train": records[:n_train],
        "valid": records[n_train:n_train + n_valid],
        "test": records[n_train + n_valid:],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in splits.items():
        path = output_dir / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"   {path.name}: {len(data)}", flush=True)
    return splits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="./ml/data_v3")
    p.add_argument("--existing", default="./ml/data/train.jsonl",
                   help="существующий train.jsonl, добавится к новым датасетам")
    p.add_argument("--min-kz-purity", type=float, default=0.85)
    p.add_argument("--train-ratio", type=float, default=0.92)
    p.add_argument("--valid-ratio", type=float, default=0.05)
    p.add_argument("--skip-amanmussa", action="store_true")
    p.add_argument("--skip-saillab", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()

    print("KazGPT V3 — Production Dataset Merge")
    print("=" * 50)

    all_raw = []
    if not args.skip_amanmussa:
        all_raw.extend(load_amanmussa())
    if not args.skip_saillab:
        all_raw.extend(load_saillab_cleaned())
    if not args.skip_existing:
        all_raw.extend(load_existing(Path(args.existing)))

    if not all_raw:
        print("[FATAL] No sources loaded.")
        sys.exit(1)

    print(f"\n=> Total raw: {len(all_raw)}", flush=True)
    print(f"=> Filtering (min kz_purity={args.min_kz_purity}) + dedup...", flush=True)
    records, stats = filter_and_dedup(all_raw, args.min_kz_purity)

    print(f"\n=> Passed: {stats['passed']} / {stats['total']} ({100 * stats['passed'] / stats['total']:.1f}%)", flush=True)
    print(f"=> Rejected breakdown:", flush=True)
    for reason, count in sorted(stats["reasons"].items(), key=lambda x: -x[1]):
        print(f"     {reason}: {count}", flush=True)

    print(f"\n=> Saving to {args.output}...", flush=True)
    splits = split_and_save(records, Path(args.output), args.train_ratio, args.valid_ratio)

    print(f"\n=> Sample (3 from train):", flush=True)
    for r in random.sample(splits["train"], min(3, len(splits["train"]))):
        u = r["messages"][0]["content"][:80]
        a = r["messages"][1]["content"][:80]
        print(f"   USER: {u}{'...' if len(u) == 80 else ''}", flush=True)
        print(f"   ASST: {a}{'...' if len(a) == 80 else ''}", flush=True)
        print(flush=True)

    print(f"=> READY for training:", flush=True)
    print(f"   python ml/train_cuda.py --data {args.output} --output ./adapters_v3", flush=True)


if __name__ == "__main__":
    main()
