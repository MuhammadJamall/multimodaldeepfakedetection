#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-data/raw}"
mkdir -p "$DATA_DIR"

echo "Place FakeAVCeleb and DFDC data under $DATA_DIR."
echo "Example (requires access):"
echo "  kaggle datasets download -d <dataset> -p $DATA_DIR"
