#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/default.yaml}"
OUT_DIR="${2:-runs/ablation}"

python -m training.train --config "$CONFIG" --output-dir "$OUT_DIR/base"
python -m training.train --config "$CONFIG" --output-dir "$OUT_DIR/no-face-crop" --override data.use_face_crop=false
python -m training.train --config "$CONFIG" --output-dir "$OUT_DIR/low-attn" --override model.attention.num_heads=4 --override model.attention.embed_dim=128
