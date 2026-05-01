# Multimodal Deepfake Detection Using Cross-Attention Audio-Visual Fusion
## Complete Project Context and Technical Specifications

## 1. Project Overview

Institution: Institute of Business and IT (IBIT), University of the Punjab, Lahore
Completion Date: April 2026
Team Size: 2 developers
Objective: Binary authenticity classification (real/fake) of talking-face video clips using joint audio-visual modeling.

Core Innovation: Audio and video in deepfakes are generated independently, leaving cross-modal desynchronization. The model exploits these inconsistencies using a dual-stream architecture with bidirectional cross-attention fusion.

## 2. System Architecture

### 2.1 Visual Stream (Spatiotemporal ViT Encoder)

Backbone: ViT-B/16 pre-trained on ImageNet-21k (Hugging Face transformers library)

Preprocessing:
- MTCNN extracts face bounding boxes
- Generates two crops per frame:
  - Full-face: 224x224 pixels
  - Mouth region: upscaled from 96x96 to 224x224
- Fallback logic: If MTCNN fails on a frame, copy bbox from last valid frame. If first frame fails, use static center crop.
- Input sequence: 16 uniformly sampled frames (T=16)

Key Constraint: Must stack full-face and mouth crops channel-wise (resulting in 6-channel input) to pass through a single ViT backbone, rather than duplicating the heavy ViT model.

Temporal Aggregation: Mean-pooling over [CLS] tokens to yield embedding $v \in R^{T\times 512}$

### 2.2 Audio Stream (Mel CNN Encoder)

Backbone: Custom CNN-6 architecture
- 6 convolutional layers with Batch Normalization, ReLU, and MaxPool

Preprocessing:
- Extract 80-band Mel Spectrograms from raw waveforms
- Sample rate: 16,000 Hz
- Window size: 25 ms
- Hop length: 10 ms
- Critical: Audio segment must perfectly match video duration (frame-by-frame alignment). No sliding window. Strict synchronization required for LSE-D loss computation.

Output: Linear projection to yield audio embedding $a \in R^{T\times 512}$ (time-windowed to match visual frames)

### 2.3 Cross-Attention Fusion Module

Bidirectional Cross-Attention Mechanism:

- Visual-attends-to-Audio: $Attn_{V\to A}(v,a) = softmax((Q_V K_A^T) / \sqrt{d_k}) V_A$
- Audio-attends-to-Visual: $Attn_{A\to V}(a,v) = softmax((Q_A K_V^T) / \sqrt{d_k}) V_V$

Parameters:
- 8 attention heads
- $d_k = 64$ (head dimension)
- Layer Normalization + residual connections around attention blocks
- Single stacked layer (no additional layers to avoid overfitting)

Integration: Concatenate visual-to-audio and audio-to-visual outputs before passing to classifier head.

### 2.4 Classifier Head

Architecture:
- Input: Concatenated embeddings (1024 dims)
- Hidden layer: 256 dims
- Output: Single neuron with Sigmoid activation
- Dropout: 0.3 before final linear layer

Output: Deepfake probability $y\_hat \in [0, 1]$

## 3. Training Strategy and Loss Function

### 3.1 Combined Loss Function

$$
\mathcal{L}_{total} = \mathcal{L}_{BCE} + \lambda \mathcal{L}_{LSE-D}
$$

Binary Cross-Entropy (BCE):
$$
\mathcal{L}_{BCE} = -[y \log(\hat{y}) + (1-y) \log(1-\hat{y})]
$$

Lip-Synchronization Error Distance (LSE-D):
- Computation: L2 distance between final visual and audio embeddings: $||v - a||_2$
- For Real Videos ($y=0$): Minimize the distance
  $$
  \mathcal{L}_{LSE-D} = ||v - a||_2
  $$
- For Fake Videos ($y=1$): Penalize perfect sync
  $$
  \mathcal{L}_{LSE-D} = \max(0, m - ||v - a||_2)
  $$
  where margin $m = 1.0$

LSE-D Weighting: Initial $\lambda = 0.3$

### 3.2 Optimizer and Learning Rate Schedule

Optimizer: AdamW
- $\beta_1 = 0.9$
- $\beta_2 = 0.999$
- Weight decay: $1 \times 10^{-2}$
- Gradient clipping: Max norm = 1.0

Differential Learning Rates:
- ViT backbone: $1 \times 10^{-4}$
- CNN-6 backbone: $1 \times 10^{-4}$
- Fusion module and classifier: $1 \times 10^{-3}$

Learning Rate Schedule:
- Linear warmup: 5 epochs
- Cosine annealing: Epochs 6-30

### 3.3 Training Phases

Phase 1 - Warm-up (Epochs 1-5):
- Freeze ViT and CNN-6 backbones
- Train only fusion module and classifier
- Critical: Keep BatchNorm and Dropout in eval() mode on frozen backbones to preserve pre-trained representations

Phase 2 - Fine-Tuning (Epochs 6-30):
- Unfreeze all layers simultaneously at Epoch 6
- Rely on lower backbone learning rate ($1 \times 10^{-4}$) to prevent catastrophic forgetting

## 4. Data Pipeline and Augmentation

### 4.1 Datasets

FakeAVCeleb (In-Distribution):
- Total: 20,000 videos (1:1 real/fake balance)
- Split:
  - Training: 14,000 videos
  - Validation: 3,000 videos
  - Test: 3,000 videos
- Stratification: By forgery method (FaceSwap, Wav2Lip, etc.)

DFDC (Out-of-Distribution Testing):
- 10,000 video subset
- Used for zero-shot cross-dataset evaluation (no fine-tuning)

### 4.2 Data Preprocessing and Serialization

Preprocessing Pipeline:
1. Extract 16 uniformly sampled frames from each video
2. Apply MTCNN for face detection and bounding box extraction
3. Generate full-face (224x224) and mouth (upscaled to 224x224) crops
4. Extract audio segment (aligned to video duration)
5. Compute 80-band Mel Spectrogram
6. Serialize to HDF5 for offline storage

Critical: Preprocess offline into HDF5 files. On-the-fly MTCNN and Mel extraction will severely bottleneck GPU training.

### 4.3 Compression Augmentation

Application: Training only (30% probability per video)
- Validation and test sets remain clean

Augmentation Operations:
- JPEG recompression (quality 40-80)
- H.264 re-encoding (CRF 23-35)
- Gaussian blur
- Temporal frame-dropping
- Audio Gaussian noise

Scope: Per-video augmentation (applied at clip level, not individual frames)

### 4.4 Data Loader and Batching

Batch Size: 32 clips per GPU (fits 24GB VRAM constraint)

Temporal Consistency: Frames from the same clip must remain together in tensor shape: (Batch, Frames, Channels, Height, Width) = (32, 16, 6, 224, 224)

Class Balancing: Use WeightedRandomSampler to guarantee 1:1 real/fake ratio in every batch

## 5. Evaluation Metrics and Targets

### 5.1 Primary Metrics

1. Area Under Receiver Operating Characteristic (AUROC)
   - In-distribution: >= 0.93
   - Cross-dataset: < 5% drop on DFDC

2. Accuracy
   - In-distribution: >= 88%

3. Equal Error Rate (EER)
   - < 8%

### 5.2 Evaluation Protocol

- In-Distribution Metrics: Computed on held-out FakeAVCeleb Test split (3,000 videos)
- Cross-Dataset Testing: DFDC subset evaluated zero-shot (no fine-tuning)
- Per-Method Analysis: Report accuracy by deepfake method (FaceSwap, Wav2Lip, etc.) for generalization proof

### 5.3 Metrics Computation

- Threshold-based accuracy at 0.5 probability
- ROC curve (AUROC) via scikit-learn
- EER via calibration curves

## 6. Implementation Requirements

### 6.1 Technology Stack

| Component | Version |
|-----------|---------|
| Python | 3.11 |
| PyTorch | >= 2.0 |
| Hugging Face Transformers | 4.x |
| OpenCV | Latest |
| MTCNN | Latest |
| librosa | Latest (Mel spectrogram extraction) |
| Weights and Biases (WandB) | Latest |
| Docker | Latest (NVIDIA CUDA 12.1+) |

### 6.2 Hardware Requirements

- Minimum GPU: 24GB VRAM (NVIDIA A100 or RTX 4090)
- CPU: 16+ cores recommended
- Storage: 500GB+ for preprocessed HDF5 datasets

### 6.3 Configuration Management

YAML Configuration File (configs/default.yaml):

```yaml
model:
  vit_model: "google/vit-base-patch16-224-in21k"
  vit_hidden_dim: 512
  audio_hidden_dim: 512
  num_frames: 16
  num_heads: 8
  head_dim: 64
  ffn_hidden_dim: 256
  dropout: 0.3

training:
  epochs: 30
  batch_size: 32
  learning_rate_vit: 1e-4
  learning_rate_audio: 1e-4
  learning_rate_fusion: 1e-3
  warmup_epochs: 5
  weight_decay: 1e-2
  gradient_clip: 1.0
  lse_d_lambda: 0.3
  lse_d_margin: 1.0

data:
  mel_bins: 80
  sample_rate: 16000
  window_ms: 25
  hop_ms: 10
  compression_augmentation_prob: 0.3
  train_split: 14000
  val_split: 3000
  test_split: 3000
```

## 7. Repository Structure

```
deepfake-detection/
├── configs/
│   └── default.yaml            # Hyperparameters
├── data/
│   ├── preprocessing.py        # MTCNN, Mel spectrogram extraction
│   ├── augmentation.py         # JPEG, H.264, blur, frame-dropping
│   └── dataset.py              # PyTorch Dataset class (HDF5 loader)
├── models/
│   ├── visual_encoder.py       # ViT-B/16 (6-channel input)
│   ├── audio_encoder.py        # CNN-6 architecture
│   ├── cross_attention.py      # Bidirectional attention layers
│   └── detector.py             # Main class assembling all components
├── training/
│   ├── train.py                # Main training loop script
│   ├── losses.py               # BCE + LSE-D loss functions
│   └── scheduler.py            # Cosine annealing + warmup
├── evaluation/
│   ├── evaluate.py             # AUROC, EER, Accuracy computation
│   └── interpretability.py     # Attention heatmap generation
├── notebooks/
│   └── exploratory_analysis.ipynb
├── scripts/
│   ├── download_data.sh         # FakeAVCeleb/DFDC download
│   └── run_ablation.sh         # Automated model variant testing
├── requirements.txt            # Dependencies (pinned versions)
├── Dockerfile                  # NVIDIA CUDA 12.1 environment
└── README.md                   # Setup and usage instructions
```

## 8. Development Phases and Milestones

### Phase 1: Data Pipeline (Blocker)
1. HDF5 dataset structure design
2. MTCNN face extraction + fallback logic
3. Mel spectrogram extraction (librosa)
4. Offline preprocessing pipeline
5. Unit tests for data loader

Success Criterion: Successfully serialize 20,000 FakeAVCeleb videos to HDF5

### Phase 2: Model Architecture
1. ViT encoder (6-channel input adaptation)
2. CNN-6 audio encoder
3. Cross-attention fusion module
4. Classifier head (FFN + Sigmoid)
5. Forward pass test with dummy data

Success Criterion: Single batch of dummy data flows through entire architecture without shape errors; loss computes correctly

### Phase 3: Training Loop
1. LSE-D loss function implementation
2. Differential learning rates + gradient clipping
3. Training loop with warm-up and fine-tuning phases
4. Validation checkpoint saving (based on AUROC)
5. WandB integration (loss, AUROC, accuracy logging)

Success Criterion: 30 epochs train without errors; validation metrics logged; best model checkpoint saved

### Phase 4: Evaluation and Deployment
1. Metrics computation (AUROC, EER, Accuracy)
2. Cross-dataset evaluation (DFDC zero-shot)
3. Attention visualization and interpretability
4. Dockerfile + environment reproducibility
5. Documentation (README, setup guide, ablation scripts)

Success Criterion: Report in-distribution AUROC >= 0.93, accuracy >= 88%, EER < 8%; cross-dataset AUROC drop < 5%

## 9. Critical Gotchas and Constraints

Data Loading:
- Gotcha: Dropping frames due to MTCNN failure breaks 16-frame tensor shape
- Solution: Use fallback (last bbox -> center crop) to maintain exact 16 frames

Preprocessing:
- Gotcha: On-the-fly MTCNN/Mel extraction during training causes severe GPU bottlenecking
- Solution: Preprocess offline to HDF5 before training

Audio Synchronization:
- Gotcha: Sliding window audio extraction misaligns with video frames, corrupting LSE-D signal
- Solution: Extract audio segment strictly matching video duration (frame-aligned)

Backbone Freezing:
- Gotcha: Leaving BatchNorm in train() mode while weights frozen corrupts running statistics
- Solution: Keep frozen backbones in eval() mode during warm-up

Channel Dimension:
- Gotcha: Processing full-face and mouth crops separately requires 2x ViT parameters
- Solution: Stack as 6-channel input to single ViT model

Cross-Attention Combination:
- Gotcha: Using gates or other learnable fusion adds unnecessary complexity and overfitting risk
- Solution: Simple concatenation of attention outputs

Augmentation Timing:
- Gotcha: Augmenting validation/test sets introduces artificial degradation, invalidating clean baseline
- Solution: Augmentation applied to training only

## 10. Success Criteria (Final)

| Metric | Target | Protocol |
|--------|--------|----------|
| In-Distribution AUROC | >= 0.93 | FakeAVCeleb Test (3,000 videos) |
| In-Distribution Accuracy | >= 88% | FakeAVCeleb Test (3,000 videos) |
| Equal Error Rate (EER) | < 8% | Calibration curve computation |
| Cross-Dataset AUROC | < 5% drop | DFDC zero-shot (10,000 videos) |
| Training Convergence | 30 epochs | No divergence, stable loss curve |

## 11. Team Collaboration Guidelines

Team Size: 2 developers

Recommended Task Split:
- Developer 1: Data pipeline (preprocessing, augmentation, HDF5 serialization) + evaluation metrics
- Developer 2: Model architecture (encoders, fusion, classifier) + training loop

Communication:
- GitHub Issues for task tracking
- Pull requests with code reviews before merging
- WandB for shared experiment tracking
- Weekly sync on metrics and blockers

## 12. Reproducibility and Environment

Docker Containerization:

```dockerfile
FROM nvidia/cuda:12.1-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y python3.11 python3.11-venv
COPY requirements.txt /workspace/
RUN pip install -r /workspace/requirements.txt
WORKDIR /workspace
```

Data Volume Mounting:

```bash
docker run --gpus all -v /path/to/datasets:/data deepfake-detector
```

Reproducibility Checklist:
- Pin all dependencies in requirements.txt
- Set random seeds (PyTorch, NumPy, random)
- WandB run IDs for experiment tracking
- Commit code before training runs
- Save config YAML with each checkpoint

## 13. References and Related Work

- ViT-B/16: Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
- Deepfake Detection: Li et al., "In Ictu Oculi: Exposing AI Created Fake Videos"
- Cross-Modal Learning: Tsai et al., "Multimodal Transformer for Unaligned Multimodal Language Sequences"
- Lip Sync Detection: Chung et al., "Lip Reading Sentences in the Wild"

End of Complete Project Context
