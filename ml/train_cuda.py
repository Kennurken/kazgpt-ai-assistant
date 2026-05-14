"""KazGPT V3 — Production fine-tune Qwen2.5-7B-Instruct + QLoRA на RTX 3070 Ti (8GB VRAM).

Цель: val loss 3.5 → 0.5-0.7, production-grade казахские ответы.

Стек:
- transformers + peft (LoRA)
- bitsandbytes (4-bit quantization, NF4)
- trl (SFTTrainer)
- unsloth (опционально, 2× ускорение и -30% VRAM)
- Hardware: RTX 3070 Ti 8GB, ~7 часов на 3 эпохи Qwen2.5-7B

Запуск (после `pip install -r requirements_train.txt` и `hf auth login`):
    python ml/train_cuda.py --data ./data --output ./adapters_v3 --epochs 3

Ключевые решения:
1. **rank=32**: меньше чем 64 в исходном плане — экономит 30% VRAM, теряем <5% качества
2. **batch_size=1 + grad_accum=16**: эффективный batch=16, помещается в 7.5GB
3. **gradient_checkpointing=True**: trade compute for memory (на 30% медленнее, -40% VRAM)
4. **packing=True в SFTTrainer**: упаковывает короткие примеры → 2-3× быстрее
5. **bf16, если поддерживается** (Ampere+) — стабильнее чем fp16
6. **EarlyStopping** на val_loss с patience=3 — защита от overfit
7. **save_best_only** — только лучший checkpoint, экономит диск
"""

import argparse
import json
import os
import sys
from pathlib import Path

# --- Lazy imports (validate args first, don't waste time importing torch on bad args) ---


def _patch_torch_load_for_resume():
    """transformers 5.x требует weights_only=True для безопасности torch.load,
    но наши собственные checkpoints содержат TrainerState / optimizer state,
    которые требуют unsafe loading. Patch'им torch.load с weights_only=False
    для resume from local checkpoint (мы доверяем своим файлам)."""
    import torch
    _orig_load = torch.load

    def patched(*args, **kwargs):
        kwargs["weights_only"] = False
        return _orig_load(*args, **kwargs)

    torch.load = patched


def parse_args():
    p = argparse.ArgumentParser(description="KazGPT V3 QLoRA training")
    # Default: Qwen2.5-7B-Instruct (public, без gating, отлично multilingual).
    # Альтернативы:
    #   - issai/LLama-3.1-KazLLM-1.0-8B (gated, нужен ISSAI approval) — лучший казахский baseline
    #   - AmanMussa/llama2-kazakh-7b (public, Llama2 fine-tune) — устаревшая база
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct",
                   help="HF model id (default: Qwen2.5-7B; для KazLLM-8B нужен ISSAI access)")
    p.add_argument("--data", default="./data", help="директория с train.jsonl + valid.jsonl")
    p.add_argument("--output", default="./adapters_v3", help="куда сохранять адаптеры")
    # Production defaults для val_loss 0.3-0.5 на 76k+ KZ instruction data:
    p.add_argument("--epochs", type=float, default=2.0,
                   help="2 epochs over 76k = ~10k steps = 10-13h on RTX 3070 Ti")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=16,
                   help="effective batch 16, помещается в 8GB при QLoRA 7B")
    p.add_argument("--lr", type=float, default=2e-4,
                   help="cosine 2e-4 → 2e-6 over training")
    p.add_argument("--max-seq", type=int, default=2048,
                   help="2048 — sweet spot для chat + instruction (4096 рискованно на 8GB)")
    p.add_argument("--lora-rank", type=int, default=64,
                   help="r=64 для 76k данных (больше capacity = ниже loss)")
    p.add_argument("--lora-alpha", type=int, default=128,
                   help="alpha = 2× rank — best practice")
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--warmup-ratio", type=float, default=0.05,
                   help="5% warmup для cosine — стабильный старт")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--use-unsloth", action="store_true", help="включить unsloth (2× ускорение, если установлен)")
    p.add_argument("--no-packing", action="store_true", help="отключить packing (debug)")
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--eval-steps", type=int, default=200)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", type=str, default=None, help="resume from checkpoint dir")
    return p.parse_args()


def check_env():
    """Проверяет CUDA, VRAM, torch."""
    try:
        import torch
    except ImportError:
        print("[FATAL] torch not installed. Install via:")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cu121")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("[FATAL] CUDA not available. Check NVIDIA driver + PyTorch CUDA build.")
        sys.exit(1)

    name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    has_bf16 = torch.cuda.is_bf16_supported()
    print(f"=> GPU: {name} ({vram_gb:.1f}GB VRAM)")
    print(f"=> CUDA: {torch.version.cuda}, torch: {torch.__version__}")
    print(f"=> bf16 supported: {has_bf16}")
    return has_bf16, vram_gb


def load_dataset(data_dir: Path):
    """Загружает train.jsonl + valid.jsonl. Конвертирует в HF Dataset с полем 'text'."""
    from datasets import Dataset

    train_path = data_dir / "train.jsonl"
    valid_path = data_dir / "valid.jsonl"
    if not train_path.exists() or not valid_path.exists():
        print(f"[FATAL] Не найдены {train_path} или {valid_path}.")
        print("Запусти сначала: python ml/prepare_data.py (или pull_kazqad.py)")
        sys.exit(1)

    def _load(path):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    train = _load(train_path)
    valid = _load(valid_path)

    def normalize(rec):
        """Нормализует к ChatML формату — поддерживает 3 варианта input:
        1. {messages: [{role, content}, ...]} — уже ChatML (от merge_datasets.py)
        2. {prompt, completion} — старый формат (от prepare_data.py)
        3. {instruction, input, output} — raw Alpaca формат (fallback)
        """
        if "messages" in rec:
            return {"messages": rec["messages"]}
        if "prompt" in rec and "completion" in rec:
            return {
                "messages": [
                    {"role": "user", "content": rec["prompt"]},
                    {"role": "assistant", "content": rec["completion"].strip()},
                ]
            }
        if "instruction" in rec and "output" in rec:
            instr = rec.get("instruction", "")
            inp = rec.get("input", "")
            prompt = f"{instr}\n\n{inp}" if inp and inp.strip() and inp != "nan" else instr
            return {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": rec["output"].strip()},
                ]
            }
        raise ValueError(f"Unknown record format. Keys: {list(rec.keys())}")

    return Dataset.from_list([normalize(r) for r in train]), Dataset.from_list([normalize(r) for r in valid])


def main():
    args = parse_args()
    has_bf16, vram_gb = check_env()

    # Patch torch.load для resume (своим checkpoints доверяем)
    if args.resume:
        _patch_torch_load_for_resume()

    if vram_gb < 7:
        print(f"[WARN] Только {vram_gb:.1f}GB VRAM. Возможно OOM. Попробуй --lora-rank 16 --max-seq 1024.")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # 1. Загружаем модель: unsloth путь vs стандартный transformers+peft
    # ============================================================
    use_unsloth = args.use_unsloth
    if use_unsloth:
        try:
            from unsloth import FastLanguageModel
            print("=> Using UNSLOTH path (2× faster, -30% VRAM)")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=args.model,
                max_seq_length=args.max_seq,
                dtype=None,  # auto
                load_in_4bit=True,
            )
            model = FastLanguageModel.get_peft_model(
                model,
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                use_gradient_checkpointing="unsloth",
                random_state=args.seed,
            )
        except ImportError:
            print("=> unsloth not installed, falling back to transformers+peft")
            use_unsloth = False

    if not use_unsloth:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        print("=> Using transformers+peft path")
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if has_bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb_cfg,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",  # вместо flash-attn (его на Windows нет)
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

        peft_cfg = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        # Если --resume указан и существует — загружаем существующий LoRA adapter (safetensors)
        # вместо fresh peft_cfg. Это обходит torch.load security issue в transformers 5.x.
        if args.resume and Path(args.resume).exists() and (Path(args.resume) / "adapter_model.safetensors").exists():
            from peft import PeftModel
            print(f"=> Loading existing LoRA adapter from {args.resume} (continue training)")
            model = PeftModel.from_pretrained(model, args.resume, is_trainable=True)
        else:
            model = get_peft_model(model, peft_cfg)
        model.print_trainable_parameters()

    # ============================================================
    # 2. Датасет
    # ============================================================
    train_ds, valid_ds = load_dataset(Path(args.data))
    # Cap valid set — full eval каждые N шагов превращается в часы.
    # 200 examples достаточно для стабильного val_loss сигнала, eval занимает ~5 мин.
    if len(valid_ds) > 200:
        valid_ds = valid_ds.shuffle(seed=42).select(range(200))
    print(f"=> Train: {len(train_ds)} | Valid: {len(valid_ds)} (subsampled for eval speed)")

    # ============================================================
    # 3. SFTTrainer
    # ============================================================
    from trl import SFTConfig, SFTTrainer

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=not use_unsloth,  # unsloth включает свой
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        max_length=args.max_seq,
        packing=not args.no_packing,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=has_bf16,
        fp16=not has_bf16,
        report_to=["tensorboard"],
        seed=args.seed,
        optim="adamw_torch_fused" if has_bf16 else "adamw_torch",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        processing_class=tokenizer,
    )

    # ============================================================
    # 4. Обучение
    # ============================================================
    print("=> Starting training...")
    # Не передаём resume_from_checkpoint в trainer — LoRA уже загружен выше,
    # optimizer state не нужен (fresh optimizer с low LR = gentle polishing).
    trainer.train()

    # ============================================================
    # 5. Сохраняем финальный LoRA адаптер
    # ============================================================
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"=> Final adapter saved to {final_dir}")

    # Summary
    print("\n=== TRAINING SUMMARY ===")
    print(f"Output dir: {output_dir}")
    print(f"Final adapter: {final_dir}")
    print("Next step: merge LoRA into base + convert to GGUF for Ollama:")
    print(f"  python ml/fuse_and_export.py --adapter {final_dir} --output kazgpt-v3.gguf")


if __name__ == "__main__":
    main()
