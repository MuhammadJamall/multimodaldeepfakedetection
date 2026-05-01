#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/default.yaml}"
OUT_DIR="${2:-runs/ablation}"

python training/train.py --config "$CONFIG"
python training/train.py --config "$CONFIG"
python training/train.py --config "$CONFIG"
