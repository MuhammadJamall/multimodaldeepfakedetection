# ============================================================================
# Multimodal Deepfake Detection — Google Colab Pro Training Notebook
# ============================================================================
#
# WORKFLOW:
#   1. Preprocess FakeAVCeleb locally → HDF5 file
#   2. Upload HDF5 to Google Drive
#   3. Open this notebook in Colab Pro (A100 GPU)
#   4. Run all cells → trained model
#   5. Download checkpoint
#
# SETUP: Runtime → Change runtime type → A100 GPU (or T4 for budget)
# ============================================================================

# %% [markdown]
# # 🔍 Multimodal Deepfake Detection — Training
# **ViT-B/16 + CNN-6 + Cross-Attention Fusion**

# %% [markdown]
# ## Step 1: Setup Environment

# %%
# Install dependencies
# !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# !pip install transformers wandb scikit-learn pyyaml tqdm h5py

# %%
# Check GPU
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# %% [markdown]
# ## Step 2: Mount Google Drive & Clone Repo

# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
# Clone your repo (or upload your project)
# !git clone https://github.com/YOUR_USERNAME/deepfake-detection.git /content/deepfake-detection

# OR: Copy from Google Drive
# !cp -r "/content/drive/MyDrive/deepfake-detection" /content/deepfake-detection

import os
os.chdir('/content/deepfake-detection')

import sys
sys.path.insert(0, '/content/deepfake-detection')

# %% [markdown]
# ## Step 3: Verify Installation (Quick Smoke Test)

# %%
# Run Phase 1 tests
# !python tests/test1.py

# %% [markdown]
# ## Step 4: Configure Training

# %%
import yaml

# Load default config
with open('configs/default.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

# ── Point to your HDF5 file on Google Drive ──
# CHANGE THIS PATH to your actual HDF5 location:
HDF5_PATH = "/content/drive/MyDrive/deepfake-data/fakeavceleb.h5"

cfg['data']['use_dummy_data'] = False
cfg['data']['hdf5_path'] = HDF5_PATH
cfg['data']['num_workers'] = 2  # Colab works best with 2

# ── Training settings (adjust based on your GPU) ──
# A100 (40GB): batch_size=32 (default)
# T4 (16GB):   batch_size=16
# V100 (16GB): batch_size=16
GPU_TYPE = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
if "T4" in GPU_TYPE or "V100" in GPU_TYPE:
    cfg['training']['batch_size'] = 16
    print(f"[Config] Reduced batch_size to 16 for {GPU_TYPE}")
elif "A100" in GPU_TYPE:
    cfg['training']['batch_size'] = 32
    print(f"[Config] Using default batch_size=32 for {GPU_TYPE}")
else:
    cfg['training']['batch_size'] = 8
    print(f"[Config] Using safe batch_size=8 for {GPU_TYPE}")

# ── Checkpoint directory (save to Drive so you don't lose them!) ──
CHECKPOINT_DIR = "/content/drive/MyDrive/deepfake-checkpoints"
cfg['checkpoint_dir'] = CHECKPOINT_DIR
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ── WandB (optional — set your API key) ──
# import wandb
# wandb.login(key="YOUR_WANDB_API_KEY")

print("\n[Config] Final training config:")
print(f"  Dataset: {HDF5_PATH}")
print(f"  Batch size: {cfg['training']['batch_size']}")
print(f"  Epochs: {cfg['training']['epochs']}")
print(f"  Checkpoints: {CHECKPOINT_DIR}")
print(f"  GPU: {GPU_TYPE}")

# %% [markdown]
# ## Step 5: Run Training

# %%
# ── Option A: Use the training script directly ──

# Save modified config
with open('/content/deepfake-detection/configs/colab_config.yaml', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)

# !python training/train.py --config configs/colab_config.yaml

# %%
# ── Option B: Run training inline (better for Colab — see progress) ──

from models.detector import DeepfakeDetector
from training.losses import DeepfakeLoss
from training.scheduler import build_scheduler
from training.train import train_epoch, val_epoch, save_checkpoint
from data.dataset import build_dataloaders

import torch.optim as optim
import time
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Build components
print("\n[1/4] Building model...")
mcfg = cfg['model']
model = DeepfakeDetector(
    vit_model=mcfg['vit_model'],
    vit_hidden_dim=mcfg['vit_hidden_dim'],
    audio_hidden_dim=mcfg['audio_hidden_dim'],
    num_heads=mcfg['num_heads'],
    ffn_hidden_dim=mcfg['ffn_hidden_dim'],
    dropout=mcfg['dropout'],
).to(device)
model.set_warmup_mode()

total_params = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total params: {total_params:,}, Trainable: {trainable:,}")

print("\n[2/4] Building data loaders...")
train_loader, val_loader = build_dataloaders(cfg)

print("\n[3/4] Building optimizer + loss...")
tcfg = cfg['training']
lcfg = cfg.get('loss', {})
param_groups = model.get_param_groups(
    lr_backbone=tcfg['learning_rate_vit'],
    lr_fusion=tcfg['learning_rate_fusion'],
)
optimizer = optim.AdamW(param_groups, betas=(0.9, 0.999), weight_decay=tcfg['weight_decay'])
scheduler = build_scheduler(optimizer, warmup_epochs=tcfg['warmup_epochs'], total_epochs=tcfg['epochs'])
criterion = DeepfakeLoss(
    lse_d_lambda=lcfg.get('lse_d_lambda', 0.3),
    lse_d_margin=lcfg.get('lse_d_margin', 1.0),
)

print("\n[4/4] Starting training...")
best_auroc = 0.0

for epoch in range(1, tcfg['epochs'] + 1):
    t0 = time.time()

    # Phase transition
    if epoch == tcfg['warmup_epochs'] + 1:
        model.set_finetune_mode()
        print(f"\n{'='*60}")
        print(f"  PHASE TRANSITION: Warmup → Fine-tuning (epoch {epoch})")
        print(f"{'='*60}\n")

    # Train
    train_metrics = train_epoch(
        model, train_loader, optimizer, criterion,
        device, tcfg['gradient_clip'], epoch,
    )

    # Validate
    val_metrics = val_epoch(model, val_loader, criterion, device)
    scheduler.step()

    elapsed = time.time() - t0
    auroc = val_metrics['val/auroc']
    acc = val_metrics['val/accuracy']

    print(
        f"Epoch {epoch:3d}/{tcfg['epochs']} │ "
        f"loss {train_metrics['train/loss_total']:.4f} │ "
        f"val_auroc {auroc:.4f} │ "
        f"val_acc {acc:.4f} │ "
        f"{elapsed:.1f}s"
    )

    # Save best
    if auroc > best_auroc:
        best_auroc = auroc
        save_checkpoint(
            model, optimizer, scheduler, epoch,
            val_metrics, cfg,
            path=os.path.join(CHECKPOINT_DIR, "best_auroc.pt"),
        )
        print(f"  ↑ New best AUROC: {best_auroc:.4f}")

    # Save latest (for resume)
    save_checkpoint(
        model, optimizer, scheduler, epoch,
        val_metrics, cfg,
        path=os.path.join(CHECKPOINT_DIR, "latest.pt"),
    )

print(f"\n✅ Training complete! Best AUROC: {best_auroc:.4f}")
print(f"   Checkpoints saved to: {CHECKPOINT_DIR}")

# %% [markdown]
# ## Step 6: Evaluate

# %%
from evaluation.evaluate import compute_metrics
from sklearn.metrics import roc_auc_score

# Load best model
ckpt = torch.load(os.path.join(CHECKPOINT_DIR, "best_auroc.pt"), map_location=device, weights_only=False)
model.load_state_dict(ckpt['model'])
model.eval()

# Evaluate on test split
test_dataset = __import__('data.dataset', fromlist=['BasicDataset']).BasicDataset(
    use_dummy_data=False,
    hdf5_path=HDF5_PATH,
    split="test",
)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

all_probs, all_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        prob = model(batch['frames'].to(device), batch['mel'].to(device))
        all_probs.extend(prob.cpu().numpy().flatten().tolist())
        all_labels.extend(batch['label'].cpu().numpy().flatten().tolist())

labels = np.array(all_labels)
probs = np.array(all_probs)
metrics = compute_metrics(labels, probs)

print(f"\n{'='*50}")
print(f"  TEST SET RESULTS")
print(f"{'='*50}")
print(f"  AUROC:    {metrics['auroc']:.4f}  (target: >= 0.93)")
print(f"  Accuracy: {metrics['accuracy']:.4f}  (target: >= 0.88)")
print(f"  EER:      {metrics['eer']:.4f}  (target: < 0.08)")
print(f"{'='*50}")

# %% [markdown]
# ## Step 7: Download Best Checkpoint

# %%
# Your checkpoint is already on Google Drive at:
print(f"Checkpoint location: {CHECKPOINT_DIR}/best_auroc.pt")
print("You can access it from Google Drive directly.")

# Or download to local machine:
# from google.colab import files
# files.download(os.path.join(CHECKPOINT_DIR, "best_auroc.pt"))
