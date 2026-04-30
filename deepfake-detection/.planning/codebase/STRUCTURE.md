# Codebase Structure

**Analysis Date:** 2026-04-29

## Directory Layout

```
[project-root]/
├── configs/          # YAML configuration
├── data/             # Dataset and preprocessing
├── models/           # Model components
├── training/         # Training loop and loss/scheduler
├── evaluation/       # Metrics and interpretability
├── notebooks/        # Analysis notebooks
├── scripts/          # Utility scripts
├── requirements.txt  # Python dependencies
└── Dockerfile        # CUDA runtime container
```

## Directory Purposes

**configs/:**
- Purpose: Configuration for training and evaluation
- Contains: YAML files
- Key files: `configs/default.yaml`

**data/:**
- Purpose: Input preprocessing and dataset loading
- Contains: `Dataset` class and augmentation helpers
- Key files: `data/dataset.py`, `data/preprocessing.py`, `data/augmentation.py`

**models/:**
- Purpose: Model components for multi-modal detection
- Contains: Encoders, cross-attention, detector
- Key files: `models/visual_encoder.py`, `models/audio_encoder.py`, `models/cross_attention.py`, `models/detector.py`

**training/:**
- Purpose: Training loop, losses, scheduler
- Contains: Training scripts and utilities
- Key files: `training/train.py`, `training/losses.py`, `training/scheduler.py`

**evaluation/:**
- Purpose: Metrics computation and interpretability helpers
- Contains: Evaluation logic and attention utilities
- Key files: `evaluation/evaluate.py`, `evaluation/interpretability.py`

**notebooks/:**
- Purpose: Exploratory analysis
- Contains: Jupyter notebooks
- Key files: `notebooks/exploratory_analysis.ipynb`

**scripts/:**
- Purpose: Automation helpers
- Contains: Shell scripts for data and ablations
- Key files: `scripts/download_data.sh`, `scripts/run_ablation.sh`

## Key File Locations

**Entry Points:**
- `training/train.py`: Main training loop
- `evaluation/evaluate.py`: Metrics helpers
- `scripts/download_data.sh`: Data acquisition guidance
- `scripts/run_ablation.sh`: Ablation runner

**Configuration:**
- `configs/default.yaml`: Hyperparameters and paths

**Core Logic:**
- `models/detector.py`: Model assembly
- `data/dataset.py`: Data loading and feature extraction

**Testing:**
- Not detected

## Naming Conventions

**Files:**
- snake_case Python modules (example: `data/preprocessing.py`)
- kebab-case shell scripts (example: `scripts/run_ablation.sh`)

**Directories:**
- lowercase directories (example: `models/`)

## Where to Add New Code

**New Feature:**
- Primary code: `models/` or `data/`
- Tests: Not detected (no test directory)

**New Component/Module:**
- Implementation: `models/`

**Utilities:**
- Shared helpers: `training/` or `data/`

## Special Directories

**.planning/codebase/:**
- Purpose: Codebase mapping documents
- Generated: Yes
- Committed: Yes

---

*Structure analysis: 2026-04-29*
