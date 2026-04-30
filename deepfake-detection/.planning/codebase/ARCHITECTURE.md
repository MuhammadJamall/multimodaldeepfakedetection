# Architecture

**Analysis Date:** 2026-04-29

## Pattern Overview

**Overall:** Modular multi-modal training pipeline

**Key Characteristics:**
- Config-driven training via `configs/default.yaml`
- Separate data, model, training, and evaluation layers under `data/`, `models/`, `training/`, `evaluation/`
- Fusion model composed of visual/audio encoders and cross-attention in `models/detector.py`

## Layers

**Data Layer:**
- Purpose: Clip loading, face cropping, and audio feature extraction
- Location: `data/`
- Contains: Dataset class, preprocessing, augmentation helpers
- Depends on: `data/preprocessing.py`, `data/augmentation.py`
- Used by: `training/train.py`

**Model Layer:**
- Purpose: Encoders, cross-attention, and detector assembly
- Location: `models/`
- Contains: `VisualEncoder`, `AudioEncoder`, attention blocks, detector
- Depends on: `models/visual_encoder.py`, `models/audio_encoder.py`, `models/cross_attention.py`
- Used by: `models/detector.py`, `training/train.py`

**Training Layer:**
- Purpose: Optimization, loss computation, scheduling
- Location: `training/`
- Contains: Training loop, loss composition, scheduler
- Depends on: `training/losses.py`, `training/scheduler.py`
- Used by: `training/train.py`

**Evaluation Layer:**
- Purpose: Metrics and interpretability utilities
- Location: `evaluation/`
- Contains: Metrics computation and attention summarization
- Depends on: `evaluation/evaluate.py`, `evaluation/interpretability.py`
- Used by: `training/train.py`, `notebooks/exploratory_analysis.ipynb`

## Data Flow

**Training Run:**

1. `training/train.py` loads hyperparameters from `configs/default.yaml`
2. `data/dataset.py` reads clips and audio, calling `data/preprocessing.py`
3. `models/detector.py` encodes modalities and fuses via `models/cross_attention.py`
4. `training/losses.py` computes combined BCE + LSE-D loss
5. `training/scheduler.py` advances cosine schedule after each epoch
6. `evaluation/evaluate.py` computes AUROC/EER/Accuracy on validation split

**State Management:**
- Training state lives in local variables inside `training/train.py`
- Model parameters are managed by PyTorch modules in `models/`

## Key Abstractions

**Dataset:**
- Purpose: Load synchronized video/audio and labels
- Examples: `data/dataset.py`
- Pattern: `torch.utils.data.Dataset` wrapper

**Encoders:**
- Purpose: Map inputs to embeddings
- Examples: `models/visual_encoder.py`, `models/audio_encoder.py`
- Pattern: `nn.Module` with pooled feature output

**Fusion:**
- Purpose: Cross-modal attention and classification
- Examples: `models/cross_attention.py`, `models/detector.py`
- Pattern: Bidirectional attention + MLP head

## Entry Points

**Training CLI:**
- Location: `training/train.py`
- Triggers: `python -m training.train`
- Responsibilities: Load config, train model, run evaluation, write checkpoints

**Evaluation Helpers:**
- Location: `evaluation/evaluate.py`
- Triggers: Imported by training or notebooks
- Responsibilities: Compute AUROC, EER, accuracy

## Error Handling

**Strategy:** Rely on exceptions from I/O and library calls

**Patterns:**
- File access errors raised in `data/dataset.py`
- Metric edge cases handled in `evaluation/evaluate.py`

## Cross-Cutting Concerns

**Logging:** Console output in `training/train.py`
**Validation:** Dataset validation implicitly handled by read failures in `data/dataset.py`
**Authentication:** Not applicable

---

*Architecture analysis: 2026-04-29*
