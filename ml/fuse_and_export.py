"""KazGPT V3 — Слияние LoRA адаптера с базовой моделью + экспорт в GGUF для Ollama.

После train_cuda.py получаем `adapters_v3/final/` — LoRA адаптер.
Этот скрипт:
1. Загружает base Qwen2.5-7B-Instruct (fp16).
2. Применяет LoRA адаптер (merge_and_unload).
3. Сохраняет merged-модель в HF формате.
4. Конвертирует в GGUF (через llama.cpp) для Ollama.
5. Создаёт Modelfile для `ollama create kazgpt`.

Использование:
    python ml/fuse_and_export.py \\
        --adapter ./adapters_v3/final \\
        --output ./kazgpt-v3-merged \\
        --gguf-out ./kazgpt-v3-Q4_K_M.gguf \\
        --quantize Q4_K_M

Затем:
    ollama create kazgpt -f ./Modelfile
    ollama run kazgpt
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct",
                   help="base модель (та же, что использовалась в train_cuda.py)")
    p.add_argument("--adapter", required=True, help="LoRA адаптер директория (adapters_v3/final)")
    p.add_argument("--output", default="./kazgpt-v3-merged", help="merged-модель (HF формат)")
    p.add_argument("--gguf-out", default="./kazgpt-v3-Q4_K_M.gguf")
    p.add_argument("--quantize", default="Q4_K_M",
                   choices=["F16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q4_0", "Q3_K_M"],
                   help="GGUF quantization. Q4_K_M — лучший баланс качества/размера для 7B")
    p.add_argument("--llama-cpp", default="./llama.cpp", help="путь к локальной llama.cpp")
    p.add_argument("--skip-merge", action="store_true", help="merged-модель уже есть, сразу GGUF")
    p.add_argument("--skip-gguf", action="store_true", help="не конвертировать в GGUF (только HF merge)")
    return p.parse_args()


def merge_lora(base: str, adapter: str, output: Path):
    """Шаг 1: применяем LoRA к base модели и сохраняем в HF формате."""
    print(f"=> Loading base model {base} (fp16)")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"=> Applying LoRA adapter from {adapter}")
    model = PeftModel.from_pretrained(model, adapter)

    print("=> Merging adapter into base weights (merge_and_unload)")
    model = model.merge_and_unload()

    output.mkdir(parents=True, exist_ok=True)
    print(f"=> Saving merged model to {output}")
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    print(f"=> Merged model size: ~{sum(f.stat().st_size for f in output.rglob('*') if f.is_file()) / 1e9:.1f}GB")


def convert_to_gguf(merged_dir: Path, gguf_out: Path, quantize: str, llama_cpp_dir: Path):
    """Шаг 2: конвертация HF → GGUF через llama.cpp."""
    if not llama_cpp_dir.exists():
        print(f"[FATAL] llama.cpp не найден в {llama_cpp_dir}.")
        print("Установка:")
        print("  git clone https://github.com/ggerganov/llama.cpp")
        print("  cd llama.cpp && cmake -B build && cmake --build build --config Release")
        sys.exit(1)

    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        print(f"[FATAL] {convert_script} не найден. Обнови llama.cpp до свежей версии.")
        sys.exit(1)

    fp16_gguf = gguf_out.with_name(gguf_out.stem + "-F16.gguf")

    print(f"=> Converting HF → GGUF F16: {fp16_gguf}")
    subprocess.run(
        [sys.executable, str(convert_script),
         str(merged_dir),
         "--outfile", str(fp16_gguf),
         "--outtype", "f16"],
        check=True,
    )

    if quantize == "F16":
        if fp16_gguf != gguf_out:
            shutil.move(str(fp16_gguf), str(gguf_out))
        return

    # Поиск quantize-бинарника (llama.cpp >= 2024 переименовал)
    quantize_bin = None
    for candidate in ["llama-quantize", "quantize"]:
        for ext in ["", ".exe"]:
            for subdir in ["build/bin/Release", "build/bin", ""]:
                p = llama_cpp_dir / subdir / f"{candidate}{ext}"
                if p.exists():
                    quantize_bin = p
                    break
            if quantize_bin:
                break
        if quantize_bin:
            break
    if not quantize_bin:
        print("[FATAL] Не нашёл llama-quantize. Собери llama.cpp: cmake --build build --config Release")
        sys.exit(1)

    print(f"=> Quantizing {fp16_gguf} → {gguf_out} ({quantize})")
    subprocess.run([str(quantize_bin), str(fp16_gguf), str(gguf_out), quantize], check=True)

    # Чистим промежуточный F16 (он ~14GB)
    fp16_gguf.unlink(missing_ok=True)


def write_modelfile(gguf_path: Path, modelfile_path: Path):
    """Шаг 3: Modelfile для Ollama. Те же параметры, что в application.yml."""
    content = f"""# KazGPT V3 — production казахский AI ассистент
# Создано: python ml/fuse_and_export.py
FROM {gguf_path.absolute().as_posix()}

PARAMETER temperature 0.3
PARAMETER top_p 0.85
PARAMETER top_k 40
PARAMETER min_p 0.05
PARAMETER repeat_penalty 1.3
PARAMETER repeat_last_n 256
PARAMETER num_ctx 4096
PARAMETER num_predict 512
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
PARAMETER stop "Қолданушы:"

TEMPLATE \"\"\"{{{{ if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}{{{{ if .Prompt }}}}<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
{{{{ end }}}}<|im_start|>assistant
{{{{ .Response }}}}<|im_end|>
\"\"\"

SYSTEM \"\"\"Сен — KazGPT, қазақ тілінде сөйлейтін жасанды интеллект көмекшісісің.
Әрдайым қазақ тілінде жауап бер. Қысқа, анық, пайдалы жауап бер.
\"\"\"
"""
    modelfile_path.write_text(content, encoding="utf-8")
    print(f"=> Modelfile written to {modelfile_path}")


def main():
    args = parse_args()
    merged_dir = Path(args.output)
    gguf_out = Path(args.gguf_out)

    if not args.skip_merge:
        merge_lora(args.base, args.adapter, merged_dir)

    if not args.skip_gguf:
        convert_to_gguf(merged_dir, gguf_out, args.quantize, Path(args.llama_cpp))

    modelfile = Path("./Modelfile")
    write_modelfile(gguf_out, modelfile)

    print("\n=== READY TO IMPORT INTO OLLAMA ===")
    print(f"  ollama create kazgpt -f {modelfile}")
    print(f"  ollama run kazgpt 'Сәлем! Өзің туралы айтып бер.'")
    print(f"\nThen update backend application.yml: models.base.name=kazgpt (instead of qwen2.5:7b)")


if __name__ == "__main__":
    main()
