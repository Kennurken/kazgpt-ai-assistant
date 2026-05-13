"""Анализ результатов обучения: график loss + before/after примеры."""

import re
import sys
from pathlib import Path

ML_DIR = Path(__file__).parent
DOCS_DIR = ML_DIR.parent / "docs"
DOCS_DIR.mkdir(exist_ok=True)

LOG_FILE = ML_DIR / "train.log"
GRAPH_FILE = DOCS_DIR / "training_curve.png"
RESULTS_FILE = DOCS_DIR / "before_after.md"


def parse_log(path):
    train_iters, train_losses = [], []
    val_iters, val_losses = [], []
    pat_train = re.compile(r"Iter (\d+): Train loss ([\d.]+)")
    pat_val = re.compile(r"Iter (\d+): Val loss ([\d.]+)")
    with open(path) as f:
        for line in f:
            m = pat_train.search(line)
            if m:
                train_iters.append(int(m.group(1)))
                train_losses.append(float(m.group(2)))
            m = pat_val.search(line)
            if m:
                val_iters.append(int(m.group(1)))
                val_losses.append(float(m.group(2)))
    return train_iters, train_losses, val_iters, val_losses


def plot(ti, tl, vi, vl):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(ti, tl, label="Train loss", color="#00d4ff", linewidth=2)
    if vi:
        plt.plot(vi, vl, label="Val loss", color="#7b2ff7", linewidth=2, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("KazGPT — Training Curve (Qwen2.5-1.5B + LoRA on KazQAD)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(GRAPH_FILE, dpi=150)
    print(f"=> График сохранён: {GRAPH_FILE}")


def main():
    if not LOG_FILE.exists():
        print(f"ERROR: {LOG_FILE} not found. Запустите train.sh сначала.")
        sys.exit(1)

    ti, tl, vi, vl = parse_log(LOG_FILE)
    if not tl:
        print("WARN: Не удалось распарсить loss из лога.")
        sys.exit(1)

    print(f"=> Train iters parsed: {len(tl)}, val iters: {len(vl)}")
    print(f"   Train loss: {tl[0]:.3f} → {tl[-1]:.3f}")
    if vl:
        print(f"   Val loss: {vl[0]:.3f} → {vl[-1]:.3f}")

    plot(ti, tl, vi, vl)

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        f.write("# KazGPT — Результаты обучения\n\n")
        f.write(f"## Метрики\n\n")
        f.write(f"- Итераций обучения: {ti[-1] if ti else 0}\n")
        f.write(f"- Train loss: {tl[0]:.3f} → {tl[-1]:.3f}\n")
        if vl:
            f.write(f"- Val loss: {vl[0]:.3f} → {vl[-1]:.3f}\n")
        f.write(f"- Модель: Qwen2.5-1.5B-Instruct (4-bit)\n")
        f.write(f"- Метод: LoRA, 4 слоя, batch_size=1\n")
        f.write(f"- Датасет: KazQAD (ISSAI Nazarbayev University)\n")
        f.write(f"- Hardware: Apple M2, 16GB unified memory\n\n")
        f.write(f"## Before/After примеры\n\n")
        f.write(f"_Заполнить после ручного прогона test.jsonl через base и v2._\n")
    print(f"=> Сводка: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
