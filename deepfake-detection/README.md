# DeepDetect — Multimodal Deepfake Detection

Binary authenticity classification (real/fake) of talking-face video clips using joint audio-visual modeling with bidirectional cross-attention fusion.

> **Core Innovation**: Audio and video in deepfakes are generated independently, leaving cross-modal desynchronization. This model exploits these inconsistencies using a dual-stream architecture (ViT-B/16 + CNN-6) with bidirectional cross-attention fusion and LSE-D loss.

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

## Results

| Dataset | AUROC | Val Accuracy | Training |
|---------|-------|-------------|----------|
| DFDC (2,202 videos) | **0.8149** | 58.97% | 20 epochs on Colab T4 |

> **Note**: Accuracy appears low because AUROC is the primary metric for imbalanced datasets (82% fake / 18% real). The model discriminates well — it just needs threshold tuning per deployment.

## Quick Start

### 1. Environment Setup

```bash
pip install -r requirements.txt
```

### 2. Preprocess Dataset

**DFDC Dataset:**
```bash
python scripts/preprocess_dfdc.py
```

**FakeAVCeleb Dataset:**
```bash
python scripts/preprocess_to_hdf5.py \
    --data-dir /path/to/FakeAVCeleb \
    --output ./data/preprocessed/fakeavceleb.h5 \
    --num-frames 16
```

### 3. Train

**Option A — Google Colab (recommended for free GPU):**

Open `notebooks/ModelTraining.ipynb` on [Google Colab](https://colab.research.google.com/).

Features:
- Auto-resume from checkpoints on Google Drive
- Mixed precision (AMP) for T4 GPU
- Time guard for free-tier session limits
- Gradient accumulation (effective batch size = 8)

**Option B — Local GPU:**
```bash
python training/train.py --config configs/default.yaml
```

Training runs with:
- **Phase 1 (Epochs 1–5)**: Backbones frozen, only fusion + classifier train
- **Phase 2 (Epochs 6+)**: All layers fine-tuned with differential learning rates
- Checkpoints saved to `checkpoints/` (best AUROC + latest)

### 4. Evaluate

```bash
python scripts/run_evaluation.py \
    --checkpoint checkpoints/best_auroc.pt \
    --hdf5-path data/preprocessed/dfdc.h5 \
    --split test \
    --dataset-name DFDC \
    --per-method
```

### 5. Web Interface

```bash
python web/server.py --checkpoint checkpoints/best_auroc.pt
```

Open `http://localhost:5000` — upload any video for real-time deepfake detection with forensic analysis.

## Project Structure

```
deepfake-detection/
├── configs/
│   └── default.yaml              # All hyperparameters
├── data/
│   ├── preprocessing.py          # MTCNN face extraction, Mel spectrograms
│   ├── augmentation.py           # JPEG, blur, frame-drop, audio noise
│   ├── dataset.py                # PyTorch Dataset + WeightedRandomSampler
│   └── dummy_dataset.py          # Synthetic data for testing
├── models/
│   ├── visual_encoder.py         # ViT-B/16 (6-channel dual-stream input)
│   ├── audio_encoder.py          # CNN-6 for Mel spectrograms
│   ├── cross_attention.py        # Bidirectional cross-attention fusion
│   └── detector.py               # Top-level model orchestrator
├── training/
│   ├── train.py                  # Training loop (phased warmup + finetune)
│   ├── losses.py                 # BCE + LSE-D combined loss
│   └── scheduler.py              # Linear warmup → cosine annealing
├── evaluation/
│   ├── evaluate.py               # AUROC, EER, accuracy computation
│   └── interpretability.py       # Cross-attention heatmap extraction
├── scripts/
│   ├── preprocess_to_hdf5.py     # FakeAVCeleb → HDF5 offline processing
│   ├── preprocess_dfdc.py        # DFDC → HDF5 offline processing
│   └── run_evaluation.py         # Standalone evaluation with reporting
├── notebooks/
│   └── ModelTraining.ipynb       # Google Colab training notebook
├── web/
│   ├── server.py                 # Flask backend for inference
│   ├── index.html                # Frontend UI
│   ├── styles.css                # Styling
│   └── script.js                 # Frontend logic
├── requirements.txt
└── README.md
```

## Training Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Epochs | 20 | Total training epochs |
| Batch size | 4 | Per-GPU batch size (×2 accumulation = effective 8) |
| LR (backbone) | 1e-4 | ViT + CNN-6 learning rate |
| LR (fusion) | 1e-3 | Fusion + classifier learning rate |
| Warmup | 5 epochs | Backbone frozen phase |
| Weight decay | 1e-2 | AdamW regularization |
| Gradient clip | 1.0 | Max gradient norm |
| λ (LSE-D) | 0.3 | LSE-D loss weight |
| Margin | 1.0 | LSE-D margin for fake videos |
| Augmentation | 30% | Per-video augmentation probability |

## Loss Function

```
L_total = L_BCE + λ · L_LSE-D

L_BCE = -[y·log(ŷ) + (1-y)·log(1-ŷ)]

L_LSE-D (Lip-Sync Error Distance):
  Real (y=0): ||v - a||₂             → minimize sync distance
  Fake (y=1): max(0, m - ||v - a||₂) → push modalities apart
```

## Class Imbalance Handling

The DFDC dataset has a 4.7:1 fake-to-real ratio. Handled via:
- **WeightedRandomSampler**: Ensures each batch sees ~50/50 real/fake samples
- **Stratified splitting**: Train/val/test maintain the same class ratio

## Preprocessing Pipeline

Each video is processed offline into HDF5:

1. **Frame extraction** — 16 uniformly sampled frames from video
2. **Face detection** — MTCNN with fallback (last bbox → center crop)
3. **Dual crops** — Full-face (224×224) + mouth region (96×96 → 224×224)
4. **Channel stacking** — 6-channel tensor (face RGB + mouth RGB)
5. **Audio extraction** — Resampled to 16kHz, mono
6. **Mel spectrogram** — 80-band, windowed to T=16 matching visual frames

## Augmentation (Training Only)

Applied with 30% probability per video:
- JPEG recompression (quality 40–80)
- Gaussian blur (σ 0.5–2.0)
- Temporal frame-dropping (20% per frame)
- Audio Gaussian noise (σ = 0.01)

## Hardware

| Environment | GPU | Training Time |
|-------------|-----|---------------|
| Google Colab (free) | Tesla T4 16GB | ~107 min / 20 epochs |
| Local | Any CUDA GPU 8GB+ | Varies |

## Team

- **Institution**: IBIT, University of the Punjab, Lahore
- **Team size**: 2 developers
- **Completion**: May 2026
