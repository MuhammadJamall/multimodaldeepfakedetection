# Multimodal Deepfake Detection Using Cross-Attention Audio-Visual Fusion

**Institution:** Institute of Business & IT (IBIT), University of the Punjab, Lahore  
**Completion Date:** April 2026  
**Team:** Solo Developer (Muhammad Jamal)

## 📋 Project Overview

This project implements a binary authenticity classification system for talking-face video clips using joint audio-visual modeling. The core innovation exploits cross-modal desynchronization in deepfakes through a dual-stream architecture with bidirectional cross-attention fusion.

**Key Insight:** Audio and video in deepfakes are generated independently, leaving detectable desynchronization patterns. Our model learns to exploit these inconsistencies.

## 🏗️ System Architecture

### Dual-Stream Encoder
- **Visual Stream:** ViT-B/16 (ImageNet-21k pretrained)
  - Input: 16 frames, full-face (224×224) + mouth crop (224×224) stacked as 6 channels
  - Output: Temporal embeddings (T×512)

- **Audio Stream:** Custom CNN-6 Architecture
  - Input: 80-band Mel Spectrograms (16kHz, synchronized to video)
  - Output: Temporal embeddings (T×512)

### Cross-Attention Fusion Module
- Bidirectional attention (Visual ↔ Audio)
- 8 attention heads, head dimension = 64
- Layer normalization + residual connections

### Classifier Head
- FFN: 1024 → 256 → 1
- Sigmoid activation
- Dropout: 0.3

## 🎯 Target Metrics

| Metric | Target |
|--------|--------|
| In-Distribution AUROC | ≥ 0.93 |
| In-Distribution Accuracy | ≥ 88% |
| Equal Error Rate (EER) | < 8% |
| Cross-Dataset AUROC Drop | < 5% |

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- CUDA 12.1 (for GPU training)
- 24GB+ VRAM (recommended)

### Local Setup

1. **Clone Repository:**
```bash
git clone https://github.com/MuhammadJamall/multimodaldeepfakedetection.git
cd multimodaldeepfakedetection
```

2. **Install Dependencies:**
```bash
pip install -r requirements.txt
```

3. **Verify Installation:**
```bash
python tests/test_phase1.py
```

### Docker Setup

1. **Build Docker Image:**
```bash
docker build -t deepfake-detector .
```

2. **Run Container with GPU:**
```bash
docker run --gpus all -v /path/to/datasets:/data -it deepfake-detector
```

3. **Run Container without GPU:**
```bash
docker run -v /path/to/datasets:/data -it deepfake-detector
```

## 📁 Project Structure

```
deepfake-detection/
├── configs/
│   ├── __init__.py
│   └── default.yaml              # All hyperparameters
├── data/
│   ├── __init__.py
│   ├── dummy_dataset.py          # Synthetic data generator
│   ├── dataset.py                # PyTorch Dataset class
│   ├── preprocessing.py          # MTCNN + Mel extraction (TODO)
│   └── augmentation.py           # Compression augmentation (TODO)
├── models/
│   ├── __init__.py
│   ├── visual_encoder.py         # ViT-B/16 (TODO)
│   ├── audio_encoder.py          # CNN-6 (TODO)
│   ├── cross_attention.py        # Fusion module (TODO)
│   └── detector.py               # Main model class (TODO)
├── training/
│   ├── __init__.py
│   ├── train.py                  # Training loop (TODO)
│   ├── losses.py                 # BCE + LSE-D (TODO)
│   └── scheduler.py              # LR scheduling (TODO)
├── evaluation/
│   ├── __init__.py
│   ├── evaluate.py               # Metrics computation (TODO)
│   └── interpretability.py       # Attention visualization (TODO)
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   └── data_preprocessing_colab.ipynb  # For Colab preprocessing
├── scripts/
│   ├── download_data.sh
│   └── run_training.sh
├── tests/
│   └── test_phase1.py            # Phase 1 validation
├── requirements.txt              # Dependencies
├── Dockerfile                    # Containerization
├── README.md                     # This file
└── .gitignore
```

## 📊 Data Pipeline

### FakeAVCeleb Dataset

The project uses the **FakeAVCeleb** dataset (20,000 videos, 1:1 real/fake balance):

**Split:**
- Training: 14,000 videos
- Validation: 3,000 videos
- Test: 3,000 videos

**Request Access:**
1. Visit: [FakeAVCeleb Google Form](https://sites.google.com/view/fakeavcelebdash-lab/)
2. Fill form with affiliation
3. Wait for approval (24-48 hours)
4. Download using provided script

### Preprocessing Steps

1. **Face Extraction:** MTCNN for face detection
   - Full-face crop: 224×224
   - Mouth crop: upscaled to 224×224
   - Fallback: Use last valid bbox or center crop

2. **Audio Processing:** Mel Spectrogram Extraction
   - Sample rate: 16,000 Hz
   - Mel bins: 80
   - Window: 25ms, Hop: 10ms
   - Frame-aligned with video

3. **Augmentation (Training Only):**
   - JPEG recompression (quality 40-80)
   - H.264 re-encoding (CRF 23-35)
   - Gaussian blur
   - Frame dropping
   - Audio Gaussian noise

4. **Serialization:** HDF5 format for efficient I/O

### Colab Preprocessing Notebook

Heavy preprocessing tasks (MTCNN, Mel extraction) run on **Google Colab**:
- See: `notebooks/data_preprocessing_colab.ipynb`
- Output: HDF5 files in Google Drive
- Then download to local/Colab for training

## 🏃 Training Instructions

### Phase 1: Dummy Data Testing (LOCAL)

Test the pipeline with synthetic data:
```bash
python tests/test_phase1.py
```

### Phase 2-4: Implementation (LOCAL)

Code development and integration testing with dummy data.

### Phase 5: Real Data Training (COLAB)

Once dataset is ready, run full training on Colab:
1. Upload HDF5 files to Google Drive
2. Mount Drive in Colab
3. Run training notebook

```bash
python training/train.py \
  --config configs/default.yaml \
  --data_dir /data/fakeavceleb_hdf5 \
  --output_dir ./checkpoints
```

## 📈 Evaluation

### Compute Metrics:
```bash
python evaluation/evaluate.py \
  --checkpoint ./checkpoints/best_model.pt \
  --test_data /data/fakeavceleb_test_hdf5
```

### Cross-Dataset Testing (DFDC):
```bash
python evaluation/evaluate.py \
  --checkpoint ./checkpoints/best_model.pt \
  --test_data /data/dfdc_test_hdf5 \
  --zero_shot true
```

## 💻 System Requirements

| Component | Requirement |
|-----------|-------------|
| GPU | 24GB VRAM (NVIDIA A100 or RTX 4090) |
| CPU | 16+ cores recommended |
| RAM | 64GB+ |
| Storage | 500GB+ for preprocessed data |
| OS | Ubuntu 20.04+ or Windows with WSL2 |

## 📚 References

- **ViT:** [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
- **FakeAVCeleb:** [DASH-Lab/FakeAVCeleb](https://github.com/DASH-Lab/FakeAVCeleb)
- **Cross-Modal Learning:** [Multimodal Transformer](https://arxiv.org/abs/1906.00295)

## 🐛 Troubleshooting

### Issue: "CUDA out of memory"
- Solution: Reduce batch_size in `configs/default.yaml`
- Or use Colab with T4/A100 GPU

### Issue: "MTCNN failed to detect face"
- Solution: Fallback logic uses last valid bbox or center crop
- See: `data/preprocessing.py`

### Issue: "ImportError: No module named 'transformers'"
- Solution: `pip install -r requirements.txt`

### Issue: "Data loading is slow"
- Solution: Use HDF5 preprocessing (offload to Colab)
- See: `notebooks/data_preprocessing_colab.ipynb`

## 📝 Development Phases

| Phase | Status | Tasks |
|-------|--------|-------|
| **Phase 1** | ✅ IN PROGRESS | Scaffolding, dummy data, README |
| **Phase 2** | ⏳ NEXT | Data pipeline (MTCNN, Mel, HDF5) |
| **Phase 3** | ⏳ NEXT | Model architecture (encoders, fusion) |
| **Phase 4** | ⏳ NEXT | Training loop, losses, W&B logging |
| **Phase 5** | ⏳ NEXT | Real data training, evaluation |

## 🔗 Related Issues

- [#1] Phase 1: Project Scaffolding
- [#2] Phase 2: Data Pipeline
- [#3] Phase 3: Model Architecture
- [#4] Phase 4: Training & Logging
- [#5] Phase 5: Evaluation & Deployment

## 📄 License

This project is licensed under the MIT License.

## 👤 Author

**Muhammad Jamal**  
Institute of Business & IT, University of the Punjab

---

**Last Updated:** April 30, 2026