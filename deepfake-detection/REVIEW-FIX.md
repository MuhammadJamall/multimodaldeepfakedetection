---
phase: ad-hoc-all-directory-review
fixed_at: 2026-05-01T00:00:00Z
review_path: deepfake-detection/REVIEW.md
iteration: 1
findings_in_scope: 3
fixed: 3
skipped: 0
status: all_fixed
---

# Phase Ad-hoc: Code Review Fix Report

**Fixed at:** 2026-05-01T00:00:00Z
**Source review:** deepfake-detection/REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 3
- Fixed: 3
- Skipped: 0

## Fixed Issues

### WR-01: Ablation script passes unsupported CLI args

**Files modified:** `scripts/run_ablation.sh`
**Commit:** e42dc3f
**Applied fix:** Removed unsupported CLI flags and invoke the training script with `--config` only.

### WR-02: AUROC computation can crash on single-class validation sets

**Files modified:** `training/train.py`
**Commit:** 19d5312
**Applied fix:** Guarded AUROC computation with `try/except` and fall back to `nan` when the validation set is single-class.

### IN-01: `learning_rate_audio` is defined but unused

**Files modified:** `models/detector.py`, `training/train.py`
**Commit:** ff29a69
**Applied fix:** Added an optional audio backbone LR in `get_param_groups()` and passed `learning_rate_audio` from config.

---

_Fixed: 2026-05-01T00:00:00Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
