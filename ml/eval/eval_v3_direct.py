"""Direct eval of V3 adapter (без Ollama / backend).
Загружает base 1.5B + LoRA, прогоняет 30 golden questions, считает все метрики.
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

EVAL_DIR = Path(__file__).parent
sys.path.insert(0, str(EVAL_DIR))

from metrics import automatic, loop_detector, kz_purity  # noqa: E402

BASE = "C:/app/kazgpt-ai-assistant/models/qwen2.5-1.5b-instruct"
ADAPTER = "C:/app/kazgpt-ai-assistant/adapters_v3_pland/final"
GOLDEN = EVAL_DIR / "golden_set.jsonl"

# Phase 0.1 sampling параметры — fair comparison
GEN_KW = dict(
    max_new_tokens=200,
    do_sample=True,
    temperature=0.3,
    top_p=0.85,
    top_k=40,
    repetition_penalty=1.15,
)

# Phase 0.1 system prompt with few-shot
SYS_PROMPT = """Сен — KazGPT, қазақ тілінде еркін сөйлейтін жасанды интеллект көмекшісісің.

ЕРЕЖЕЛЕР:
1. Әрдайым қазақ тілінде жауап бер.
2. Жауаптарың қысқа және анық болсын: 1-4 сөйлем.
3. Айқын білмесең — "Кешіріңіз, бұл жайында нақты ақпаратым жоқ" деп жауап бер.
4. Мейірімді, сыпайы, кәсіби бол.

МЫСАЛДАР:
Қолданушы: Сәлем!
KazGPT: Сәлеметсіз! Сізге қалай көмектесе аламын?

Қолданушы: Қазақстанның астанасы қандай?
KazGPT: Қазақстанның астанасы — Астана қаласы.

Қолданушы: Translate to Kazakh: I love my homeland.
KazGPT: «Мен өз Отанымды сүйемін.»"""


def load_model():
    print("[1] Loading tokenizer + base + LoRA...", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(ADAPTER)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map="auto")
    m = PeftModel.from_pretrained(m, ADAPTER)
    m.eval()
    print(f"    Loaded in {time.time()-t0:.1f}s", flush=True)
    return tok, m


def chat(tok, m, question: str) -> tuple[str, float]:
    msgs = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": question},
    ]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(m.device)
    t = time.time()
    with torch.no_grad():
        out = m.generate(**inputs, **GEN_KW, pad_token_id=tok.pad_token_id)
    elapsed = time.time() - t
    resp = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    return resp, elapsed


def main():
    items = []
    with open(GOLDEN, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    print(f"=> {len(items)} golden items", flush=True)

    tok, m = load_model()

    print(f"\n[2] Running eval...", flush=True)
    rows = []
    times = []
    for i, it in enumerate(items, 1):
        resp, t = chat(tok, m, it["question"])
        times.append(t)
        auto = automatic.compute_all(
            resp, it.get("reference_answers", []),
            must_contain_any=it.get("must_contain_any", []),
            must_not_contain=it.get("must_not_contain", []),
            skip_bertscore=True,
        )
        loop = loop_detector.detect_loop(resp)
        purity = kz_purity.kz_purity(resp)
        marker = "✗" if loop["is_loop"] else "✓"
        print(f"  [{i:>2}/{len(items)}] {marker} {it['id']:<20} "
              f"bleu={auto['bleu']:>5} loop={loop['repetition_rate']:>4} "
              f"purity={purity['purity']:.2f} ({t:.1f}s)", flush=True)
        rows.append({
            "id": it["id"], "domain": it["domain"],
            "question": it["question"], "response": resp,
            "auto": auto, "loop": loop, "purity": purity,
            "time_sec": round(t, 2),
        })

    # Summary
    def safe(vals): return [v for v in vals if v is not None]
    bleus = safe([r["auto"]["bleu"] for r in rows])
    rouges = safe([r["auto"]["rouge_l"] for r in rows])
    summary = {
        "n": len(rows),
        "bleu_avg": round(mean(bleus), 2) if bleus else None,
        "rouge_l_avg": round(mean(rouges), 3) if rouges else None,
        "loop_pct": round(100 * sum(1 for r in rows if r["loop"]["is_loop"]) / len(rows), 1),
        "loop_rate_avg": round(mean(r["loop"]["repetition_rate"] for r in rows), 3),
        "kz_purity_avg": round(mean(r["purity"]["purity"] for r in rows), 3),
        "assertions_pct": round(100 * sum(1 for r in rows if r["auto"]["assertions"]["passed"]) / len(rows), 1),
        "time_avg_sec": round(mean(times), 2),
    }

    print(f"\n=== SUMMARY V3 ===")
    for k, v in summary.items():
        print(f"  {k:<20} {v}")

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = EVAL_DIR / "reports" / f"v3_direct_{ts}.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows}, f, ensure_ascii=False, indent=2)
    print(f"\n=> Saved: {out}")


if __name__ == "__main__":
    main()
