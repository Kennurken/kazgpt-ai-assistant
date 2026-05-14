"""Главный orchestrator KazGPT eval harness.

Использование:

    # Быстрый прогон против base модели (без BERTScore и LLM-judge)
    python run_eval.py --models base --fast

    # Полный прогон, сравнение двух моделей
    python run_eval.py --models base v2 --enable-bertscore --enable-llm-judge

    # Фильтрация по домену
    python run_eval.py --models base --domain kz_knowledge

Что делает:
1. Читает golden_set.jsonl
2. Для каждой пары (model, question) делает запрос к бэкенду через client.py
3. Считает: BLEU, ROUGE-L, BERTScore, loop_rate, kz_purity, latency, assertions, LLM-judge
4. Пишет JSON-отчёт в reports/{timestamp}.json и markdown в reports/{timestamp}.md
5. На stdout — компактная таблица для CI

Зависит от: client.py, metrics/*.py
"""

import argparse
import io
import json
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Dict, List

# Принудительный UTF-8 stdout для Windows (где default cp1251 рушится на казахских буквах).
# На macOS/Linux это no-op.
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", line_buffering=True)

EVAL_DIR = Path(__file__).parent
sys.path.insert(0, str(EVAL_DIR))

from client import chat, ChatResponse  # noqa: E402
from metrics import automatic, loop_detector, kz_purity, latency, llm_judge  # noqa: E402


def load_golden_set(path: Path) -> List[Dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[warn] golden_set.jsonl:{line_no} invalid JSON: {e}")
    return items


def filter_items(items: List[Dict], domain: str = "", tag: str = "") -> List[Dict]:
    if domain:
        items = [it for it in items if it.get("domain") == domain]
    if tag:
        items = [it for it in items if tag in it.get("tags", [])]
    return items


def eval_one(
    item: Dict,
    response: ChatResponse,
    args,
) -> Dict:
    """Считает все метрики для одного (item, response) пары."""
    pred = response.text
    refs = item.get("reference_answers", [])

    auto = automatic.compute_all(
        pred,
        refs,
        must_contain_any=item.get("must_contain_any"),
        must_not_contain=item.get("must_not_contain"),
        skip_bertscore=not args.enable_bertscore,
    )
    loop = loop_detector.detect_loop(pred, n=args.loop_ngram, threshold=args.loop_threshold)
    purity = kz_purity.kz_purity(pred)

    judge_result = None
    if args.enable_llm_judge:
        judge_result = llm_judge.judge(
            item["question"],
            pred,
            refs,
            provider=args.judge_provider,
            model=args.judge_model,
        )

    return {
        "id": item["id"],
        "domain": item.get("domain", "unknown"),
        "question": item["question"],
        "prediction": pred,
        "error": response.error,
        "latency": {
            "ttft_ms": response.ttft_ms,
            "total_ms": response.total_ms,
            "tokens": response.token_count,
        },
        "automatic": auto,
        "loop": loop,
        "purity": purity,
        "judge": judge_result,
    }


def aggregate_per_model(rows: List[Dict], responses: List[ChatResponse]) -> Dict:
    """Сворачивает построчные метрики в средние/процентили по модели."""
    if not rows:
        return {}

    def safe_mean(vals):
        vals = [v for v in vals if v is not None]
        return round(mean(vals), 3) if vals else None

    bleu = safe_mean([r["automatic"]["bleu"] for r in rows])
    rouge = safe_mean([r["automatic"]["rouge_l"] for r in rows])
    bs = safe_mean([r["automatic"]["bertscore"] for r in rows])
    loop_rate_avg = safe_mean([r["loop"]["repetition_rate"] for r in rows])
    loop_pct = round(100 * sum(1 for r in rows if r["loop"]["is_loop"]) / len(rows), 1)
    purity_avg = safe_mean([r["purity"]["purity"] for r in rows])
    assertions_pct = round(
        100 * sum(1 for r in rows if r["automatic"]["assertions"]["passed"]) / len(rows), 1
    )
    judge_scores = [r["judge"]["score"] for r in rows if r.get("judge")]
    judge_avg = round(mean(judge_scores), 2) if judge_scores else None

    lat = latency.aggregate(responses)

    return {
        "n": len(rows),
        "bleu_avg": bleu,
        "rouge_l_avg": rouge,
        "bertscore_avg": bs,
        "judge_avg": judge_avg,
        "loop_pct": loop_pct,
        "loop_rate_avg": loop_rate_avg,
        "kz_purity_avg": purity_avg,
        "assertions_pct": assertions_pct,
        **lat,
    }


def write_markdown_report(out_path: Path, summary: Dict, all_rows: Dict[str, List[Dict]]):
    """Удобочитаемый отчёт. Compare.py делал нечто похожее — мы расширяем."""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# KazGPT Eval Report — {datetime.now().isoformat(timespec='seconds')}\n\n")

        f.write("## Summary per model\n\n")
        f.write("| Model | N | BLEU | ROUGE-L | BERTScore | Judge | Loop% | KZ-purity | Assert% | TTFT p50 | tok/s |\n")
        f.write("|-------|---|------|---------|-----------|-------|-------|-----------|---------|----------|-------|\n")
        for model, s in summary.items():
            f.write(
                f"| **{model}** | {s['n']} | {s['bleu_avg']} | {s['rouge_l_avg']} | "
                f"{s['bertscore_avg']} | {s['judge_avg']} | {s['loop_pct']}% | "
                f"{s['kz_purity_avg']} | {s['assertions_pct']}% | "
                f"{s.get('ttft_p50_ms', '—')}ms | {s.get('tokens_per_sec_avg', '—')} |\n"
            )

        f.write("\n## Failures (loops or assertion failures)\n\n")
        for model, rows in all_rows.items():
            bad = [r for r in rows if r["loop"]["is_loop"] or not r["automatic"]["assertions"]["passed"]]
            if not bad:
                f.write(f"### {model} — clean\n\n")
                continue
            f.write(f"### {model} — {len(bad)} issue(s)\n\n")
            for r in bad:
                f.write(f"- **{r['id']}** ({r['domain']}): ")
                if r["loop"]["is_loop"]:
                    f.write(f"loop on '{r['loop']['most_common_ngram']}' ({r['loop']['max_count']}×); ")
                if not r["automatic"]["assertions"]["passed"]:
                    f.write(
                        f"missing keywords={not r['automatic']['assertions']['contains_required']} "
                        f"forbidden={r['automatic']['assertions']['forbidden_found']}; "
                    )
                f.write(f"\n  - Q: {r['question'][:80]}...\n  - A: {r['prediction'][:120]}...\n")
            f.write("\n")

        f.write("\n## Sample responses (first 3 per model)\n\n")
        for model, rows in all_rows.items():
            f.write(f"### {model}\n\n")
            for r in rows[:3]:
                f.write(f"**Q ({r['id']}):** {r['question']}\n\n")
                f.write(f"**A:** {r['prediction']}\n\n")
                if r.get("judge"):
                    f.write(f"_judge_score: {r['judge']['score']}, reasoning: {r['judge']['reasoning'][:100]}_\n\n")
                f.write("---\n\n")


def main():
    parser = argparse.ArgumentParser(description="KazGPT eval harness")
    parser.add_argument("--backend", default="http://localhost:8080/api/chat/stream")
    parser.add_argument("--models", nargs="+", default=["base"], help="модели для прогона")
    parser.add_argument("--golden", default=str(EVAL_DIR / "golden_set.jsonl"))
    parser.add_argument("--reports", default=str(EVAL_DIR / "reports"))
    parser.add_argument("--domain", default="", help="фильтр по полю domain")
    parser.add_argument("--tag", default="", help="фильтр по полю tags")
    parser.add_argument("--fast", action="store_true", help="отключает дорогие метрики")
    parser.add_argument("--enable-bertscore", action="store_true")
    parser.add_argument("--enable-llm-judge", action="store_true")
    parser.add_argument("--judge-provider", default="openai", choices=["openai", "anthropic"])
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--loop-ngram", type=int, default=4)
    parser.add_argument("--loop-threshold", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="ограничить число примеров (для дебага)")
    args = parser.parse_args()

    if args.fast:
        args.enable_bertscore = False
        args.enable_llm_judge = False

    items = load_golden_set(Path(args.golden))
    items = filter_items(items, args.domain, args.tag)
    if args.limit > 0:
        items = items[: args.limit]
    if not items:
        print(f"[error] no items loaded from {args.golden} (after filters)")
        sys.exit(1)
    print(f"=> loaded {len(items)} items")

    summary = {}
    all_rows = {}

    for model in args.models:
        print(f"\n=== {model} ===")
        rows = []
        responses = []
        for i, item in enumerate(items, 1):
            r = chat(args.backend, item["question"], model=model)
            responses.append(r)
            row = eval_one(item, r, args)
            rows.append(row)
            marker = "✗" if (r.error or row["loop"]["is_loop"]) else "✓"
            print(
                f"  [{i:>3}/{len(items)}] {marker} {item['id']:<20} "
                f"ttft={r.ttft_ms or 0:.0f}ms loop={row['loop']['repetition_rate']} "
                f"purity={row['purity']['purity']}"
            )
        summary[model] = aggregate_per_model(rows, responses)
        all_rows[model] = rows

    # Save artifacts
    Path(args.reports).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_out = Path(args.reports) / f"{ts}.json"
    md_out = Path(args.reports) / f"{ts}.md"

    with open(json_out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": all_rows, "args": vars(args)}, f, ensure_ascii=False, indent=2)
    write_markdown_report(md_out, summary, all_rows)

    print(f"\n=> JSON: {json_out}")
    print(f"=> MD:   {md_out}")
    print("\n=== Summary ===")
    for model, s in summary.items():
        print(
            f"  {model:<12} bleu={s['bleu_avg']} rouge={s['rouge_l_avg']} "
            f"loop%={s['loop_pct']} purity={s['kz_purity_avg']} "
            f"assert%={s['assertions_pct']} ttft_p50={s.get('ttft_p50_ms', '—')}ms"
        )


if __name__ == "__main__":
    main()
