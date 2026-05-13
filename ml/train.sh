#!/usr/bin/env bash
# KazGPT — Запуск MLX LoRA fine-tune
# Использование: ./train.sh
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

VENV_PATH="${VENV_PATH:-$HOME/kazgpt-venv}"

if [[ ! -d "$VENV_PATH" ]]; then
    echo "Создаём venv: $VENV_PATH"
    python3 -m venv "$VENV_PATH"
fi

# shellcheck disable=SC1090
source "$VENV_PATH/bin/activate"

echo "=> Установка/обновление зависимостей..."
pip install --quiet --upgrade pip
pip install --quiet mlx-lm datasets huggingface_hub matplotlib

if [[ ! -f "data/train.jsonl" ]]; then
    echo "=> Данных нет, запускаем prepare_data.py"
    python prepare_data.py
fi

echo "=> Старт fine-tune (config.yaml)"
echo "   Лог: $SCRIPT_DIR/train.log"

mkdir -p adapters
python -m mlx_lm.lora --config config.yaml 2>&1 | tee train.log

echo
echo "=> Готово. Адаптер: $SCRIPT_DIR/adapters/"
echo "=> Запуск mlx-lm server для интеграции с бэкэндом:"
echo "   python -m mlx_lm.server --model mlx-community/Qwen2.5-1.5B-Instruct-4bit --adapter-path ./adapters --port 11435"
