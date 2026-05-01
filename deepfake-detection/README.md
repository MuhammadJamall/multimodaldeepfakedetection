# Multimodal Deepfake Detection Using Cross-Attention Audio-Visual Fusion

Binary authenticity classification (real/fake) of talking-face video clips using joint audio-visual modeling with bidirectional cross-attention fusion.

> **Core Innovation**: Audio and video in deepfakes are generated independently, leaving cross-modal desynchronization. This model exploits these inconsistencies using a dual-stream architecture (ViT-B/16 + CNN-6) with bidirectional cross-attention fusion.

## Architecture

```
Video Frames (B, 16, 6, 224, 224)     Mel Spectrogram (B, 16, 80, F)
         │                                       │
    VisualEncoder                           AudioEncoder
    (ViT-B/16, 6-ch)                       (CNN-6)
         │                                       │
    v ∈ (B, 16, 512)                       a ∈ (B, 16, 512)
         │                                       │
         └──────── CrossAttentionFusion ─────────┘
                   (Bidirectional, 8 heads)
                          │
                    (B, 1024) fused
                          │
                   ClassifierHead
                   (1024→256→1, Sigmoid)
                          │
                   Probability ∈ [0, 1]
```

## Quick Start

### 1. Environment Setup

```bash
# Option A: Docker (recommended)
docker build -t deepfake-detector .
docker run --gpus all -v /path/to/data:/data -it deepfake-detector

# Option B: Local
pip install -r requirements.txt
```

### 2. Verify Installation (Smoke Test)

```bash
python scripts/smoke_test.py
```

This runs 2 epochs with dummy data to verify the entire pipeline:
data → model → loss → optimizer → checkpoint → evaluation.

### 3. Preprocess FakeAVCeleb Dataset

```bash
python scripts/preprocess_to_hdf5.py \
    --data-dir /path/to/FakeAVCeleb \
    --output ./data/preprocessed/fakeavceleb.h5 \
    --num-frames 16
```

Expected directory structure for FakeAVCeleb:
```
FakeAVCeleb/
├── RealVideo/
│   └── *.mp4
└── FakeVideo/
    ├── FaceSwap/
    │   └── *.mp4
    ├── Wav2Lip/
    │   └── *.mp4
    └── .../
```

### 4. Configure for Real Data

Edit `configs/default.yaml`:

```yaml
data:
  use_dummy_data: false
  hdf5_path: "./data/preprocessed/fakeavceleb.h5"
```

### 5. Train

```bash
python training/train.py --config configs/default.yaml
```

Training runs for 30 epochs with:
- **Phase 1 (Epochs 1–5)**: Backbones frozen, only fusion + classifier train
- **Phase 2 (Epochs 6–30)**: All layers fine-tuned with differential learning rates
- Checkpoints saved to `checkpoints/` (best AUROC + latest)
- Metrics logged to Weights & Biases

### 6. Evaluate

```bash
# In-distribution evaluation
python scripts/run_evaluation.py \
    --checkpoint checkpoints/best_auroc.pt \
    --per-method

# Cross-dataset (DFDC zero-shot)
python scripts/run_evaluation.py \
    --checkpoint checkpoints/best_auroc.pt \
    --hdf5-path data/preprocessed/dfdc.h5 \
    --dataset-name DFDC
```

## Project Structure

```
deepfake-detection/
├── configs/
│   └── default.yaml              # All hyperparameters
├── data/
│   ├── preprocessing.py          # MTCNN face extraction, Mel spectrograms
│   ├── augmentation.py           # JPEG, blur, frame-drop, audio noise
│   ├── dataset.py                # PyTorch Dataset (dummy + HDF5)
│   └── dummy_dataset.py          # Synthetic data for testing
├── models/
│   ├── visual_encoder.py         # ViT-B/16 (6-channel input)
│   ├── audio_encoder.py          # CNN-6 for Mel spectrograms
│   ├── cross_attention.py        # Bidirectional cross-attention fusion
│   └── detector.py               # Top-level model
├── training/
│   ├── train.py                  # Training loop (phased warmup + finetune)
│   ├── losses.py                 # BCE + LSE-D combined loss
│   └── scheduler.py              # Linear warmup → cosine annealing
├── evaluation/
│   ├── evaluate.py               # AUROC, EER, accuracy computation
│   └── interpretability.py       # Cross-attention heatmap extraction
├── scripts/
│   ├── smoke_test.py             # End-to-end pipeline verification
│   ├── preprocess_to_hdf5.py     # FakeAVCeleb → HDF5 offline processing
│   ├── run_evaluation.py         # Standalone evaluation with reporting
│   ├── run_ablation.sh           # Ablation experiments
│   └── download_data.sh          # Data download helper
├── tests/
│   └── test1.py                  # Phase 1 validation suite (6 tests)
├── notebooks/
│   └── exploratory_analysis.ipynb
├── requirements.txt
├── Dockerfile
├── CONTEXT.md                    # Full technical specification
└── README.md                     # This file
```

## Training Configuration

All hyperparameters in `configs/default.yaml`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Epochs | 30 | Total training epochs |
| Batch size | 32 | Per-GPU batch size |
| LR (backbone) | 1e-4 | ViT + CNN-6 learning rate |
| LR (fusion) | 1e-3 | Fusion + classifier learning rate |
| Warmup | 5 epochs | Linear LR warmup |
| Weight decay | 1e-2 | AdamW regularization |
| Gradient clip | 1.0 | Max gradient norm |
| λ (LSE-D) | 0.3 | LSE-D loss weight |
| Margin | 1.0 | LSE-D margin for fake videos |
| Augmentation | 30% | Per-video augmentation probability |

## Loss Function

```
L_total = L_BCE + λ · L_LSE-D

L_BCE = -[y·log(ŷ) + (1-y)·log(1-ŷ)]

L_LSE-D:
  Real (y=0): ||v - a||₂           (minimize sync distance)
  Fake (y=1): max(0, m - ||v - a||₂)  (push modalities apart)
```

## Target Metrics

| Metric | Target | Protocol |
|--------|--------|----------|
| AUROC (in-distribution) | ≥ 0.93 | FakeAVCeleb test (3,000 videos) |
| Accuracy | ≥ 88% | Threshold at 0.5 |
| EER | < 8% | Equal Error Rate |
| AUROC (cross-dataset) | < 5% drop | DFDC zero-shot (10,000 videos) |

## Preprocessing Pipeline

Each video is processed offline into HDF5:

1. **Frame extraction**: 16 uniformly sampled frames from video
2. **Face detection**: MTCNN with fallback (last bbox → center crop)
3. **Dual crops**: Full-face (224×224) + mouth region (96×96 → 224×224)
4. **Channel stacking**: 6-channel tensor (face RGB + mouth RGB)
5. **Audio extraction**: Resampled to 16kHz, mono
6. **Mel spectrogram**: 80-band, windowed to T=16 matching visual frames

## Augmentation (Training Only)

Applied with 30% probability per video:
- JPEG recompression (quality 40–80)
- Gaussian blur (σ 0.5–2.0)
- Temporal frame-dropping (20% per frame)
- Audio Gaussian noise (σ = 0.01)
- H.264 re-encoding (CRF 23–35)

## Hardware Requirements

- **GPU**: 24GB VRAM minimum (NVIDIA A100 or RTX 4090)
- **CPU**: 16+ cores recommended
- **Storage**: 500GB+ for preprocessed HDF5 datasets

## Running Tests

```bash
# Phase 1 data pipeline tests
python tests/test1.py

# Preprocessing unit tests
python data/preprocessing.py

# Augmentation unit tests
python data/augmentation.py

# Full end-to-end smoke test
python scripts/smoke_test.py
```

## Switching from Dummy to Real Data

The project ships with dummy data generators for development. To switch:

1. Place FakeAVCeleb videos in a directory
2. Run: `python scripts/preprocess_to_hdf5.py --data-dir /path/to/FakeAVCeleb`
3. Edit `configs/default.yaml`:
   ```yaml
   data:
     use_dummy_data: false
     hdf5_path: "./data/preprocessed/fakeavceleb.h5"
   ```
4. Train: `python training/train.py`

## Team

- Institution: IBIT, University of the Punjab, Lahore
- Team size: 2 developers
- Completion: April 2026
