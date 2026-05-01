---
phase: ad-hoc-all-directory-review
reviewed: 2026-05-01T00:00:00Z
depth: standard
files_reviewed: 27
files_reviewed_list:
  - training/train.py
  - training/scheduler.py
  - training/losses.py
  - models/audio_encoder.py
  - models/cross_attention.py
  - models/detector.py
  - models/visual_encoder.py
  - data/preprocessing.py
  - evaluation/evaluate.py
  - evaluation/interpretability.py
  - scripts/preprocess_to_hdf5.py
  - scripts/run_evaluation.py
  - scripts/smoke_test.py
  - scripts/download_data.sh
  - scripts/run_ablation.sh
  - configs/default.yaml
  - requirements.txt
  - Dockerfile
  - tests/test1.py
  - notebooks/colab_training.py
  - README.md
  - CONTEXT.md
  - models/__init__.py
  - training/__init__.py
  - evaluation/__init__.py
  - tests/__init__.py
  - configs/__init__.py
findings:
  critical: 0
  warning: 2
  info: 1
  total: 3
status: issues_found
---

# Phase Ad-hoc: Code Review Report

**Reviewed:** 2026-05-01T00:00:00Z
**Depth:** standard
**Files Reviewed:** 27
**Status:** issues_found

## Summary

Reviewed the full pipeline scripts, training loop, evaluation utilities, configs, and supporting files. The core pipeline is coherent, but there are a few correctness and usability issues that will surface when running ablations or when validation data is degenerate.

## Warnings

### WR-01: Ablation script passes unsupported CLI args

**File:** `scripts/run_ablation.sh:7-9`
**Issue:** `training/train.py` only accepts `--config`, so `--output-dir` and `--override` will cause an argparse error and abort the ablation runs.
**Fix:** Either add these flags to `training/train.py`, or remove them from the script. Minimal safe change:
```bash
python training/train.py --config "$CONFIG"
```
If overrides are required, implement an override parser in `training/train.py` and apply to the loaded YAML.

### WR-02: AUROC computation can crash on single-class validation sets

**File:** `training/train.py:177`
**Issue:** `roc_auc_score` raises `ValueError` if the validation split contains only one class, which will crash training.
**Fix:** Guard AUROC computation with `try/except` and fall back to `nan` (or 0.5) plus a warning.
```python
try:
    auroc = roc_auc_score(all_labels, all_probs)
except ValueError:
    auroc = float("nan")
```

## Info

### IN-01: `learning_rate_audio` is defined but unused

**File:** `configs/default.yaml:23`
**Issue:** The config exposes `learning_rate_audio`, but `training/train.py` only uses `learning_rate_vit` and `learning_rate_fusion` (audio backbone inherits `learning_rate_vit`). This can mislead when tuning.
**Fix:** Either remove the unused config key or extend `get_param_groups()` to accept an explicit audio LR and pass it from `train.py`.

---

_Reviewed: 2026-05-01T00:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
