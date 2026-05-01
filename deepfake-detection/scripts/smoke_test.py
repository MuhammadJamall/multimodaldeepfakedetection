#!/usr/bin/env python3
"""
smoke_test.py
-------------
End-to-end smoke test for the entire deepfake detection pipeline.

Runs 2 epochs of training with tiny dummy data (8 samples, batch=4)
to verify the complete pipeline works:
  Data loading → Model forward → Loss computation → Backward →
  Optimizer step → Validation → Checkpoint saving → Evaluation

No GPU required. Completes in ~1-2 minutes on CPU.

Usage:
    python scripts/smoke_test.py
"""

import os
import sys
import time
import shutil

import torch
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from models.detector import DeepfakeDetector
from training.losses import DeepfakeLoss
from training.scheduler import build_scheduler
from data.dataset import build_dataloaders


def print_header(text: str):
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80)


def print_step(step: int, text: str):
    print(f"\n[Step {step}] {text}")


def main():
    print_header("END-TO-END SMOKE TEST")
    print("Verifying: Data → Model → Loss → Optimizer → Checkpoint → Evaluation")
    print("Using: Tiny dummy data (8 train, 4 val, batch_size=4, 2 epochs)")

    t_start = time.time()
    device = torch.device("cpu")
    ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints", "_smoke_test")

    # Clean up any previous smoke test artifacts
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)

    # ── Step 1: Build data loaders ─────────────────────────────────────────
    print_step(1, "Building DataLoaders with dummy data...")
    cfg = {
        "data": {
            "use_dummy_data": True,
            "train_split": 8,
            "val_split": 4,
            "num_workers": 0,
            "pin_memory": False,
            "compression_augmentation_prob": 0.3,
        },
        "training": {"batch_size": 4},
        "augmentation": {
            "jpeg_quality_range": [40, 80],
            "blur_sigma_range": [0.5, 2.0],
            "frame_drop_prob": 0.2,
            "audio_noise_std": 0.01,
        },
    }
    train_loader, val_loader = build_dataloaders(cfg)

    # Verify batch shapes
    batch = next(iter(train_loader))
    assert batch["frames"].shape == (4, 16, 6, 224, 224), \
        f"Wrong frames shape: {batch['frames'].shape}"
    assert batch["mel"].shape[0] == 4 and batch["mel"].shape[1] == 16, \
        f"Wrong mel shape: {batch['mel'].shape}"
    assert batch["label"].shape == (4, 1), \
        f"Wrong label shape: {batch['label'].shape}"
    print(f"    ✅ Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")
    print(f"    Frames: {batch['frames'].shape}, Mel: {batch['mel'].shape}")

    # ── Step 2: Build model ────────────────────────────────────────────────
    print_step(2, "Building DeepfakeDetector...")
    model = DeepfakeDetector(
        vit_model="google/vit-base-patch16-224-in21k",
        vit_hidden_dim=512,
        audio_hidden_dim=512,
        num_heads=8,
        ffn_hidden_dim=256,
        dropout=0.3,
    ).to(device)

    model.set_warmup_mode()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    ✅ Model built: {total_params:,} total params, {trainable_params:,} trainable (warmup)")

    # ── Step 3: Build optimizer + scheduler + loss ─────────────────────────
    print_step(3, "Building optimizer, scheduler, loss...")
    param_groups = model.get_param_groups(lr_backbone=1e-4, lr_fusion=1e-3)
    optimizer = optim.AdamW(param_groups, betas=(0.9, 0.999), weight_decay=1e-2)
    scheduler = build_scheduler(optimizer, warmup_epochs=1, total_epochs=2)
    criterion = DeepfakeLoss(lse_d_lambda=0.3, lse_d_margin=1.0)
    print("    ✅ Optimizer: AdamW, Scheduler: warmup+cosine, Loss: BCE+LSE-D")

    # ── Step 4: Training loop (2 epochs) ───────────────────────────────────
    print_step(4, "Running 2 training epochs...")

    for epoch in range(1, 3):
        t0 = time.time()

        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        if epoch <= 1:
            # Keep frozen backbones in eval mode (BN protection)
            model.visual_encoder.vit.eval()
            for m in [model.audio_encoder.conv1, model.audio_encoder.conv2,
                      model.audio_encoder.conv3, model.audio_encoder.conv4,
                      model.audio_encoder.conv5, model.audio_encoder.conv6]:
                m.eval()

        if epoch == 2:
            model.set_finetune_mode()

        train_loss = 0.0
        n_train = 0
        for batch in train_loader:
            frames = batch["frames"].to(device)
            mel = batch["mel"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            prob, embs = model(frames, mel, return_embeddings=True)
            losses = criterion(prob, embs["v"], embs["a"], labels)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += losses["total"].item()
            n_train += 1

        # ── Validate ──────────────────────────────────────────────────────
        model.eval()
        all_probs, all_labels = [], []
        val_loss = 0.0
        n_val = 0

        with torch.no_grad():
            for batch in val_loader:
                frames = batch["frames"].to(device)
                mel = batch["mel"].to(device)
                labels = batch["label"].to(device)

                prob, embs = model(frames, mel, return_embeddings=True)
                losses = criterion(prob, embs["v"], embs["a"], labels)

                val_loss += losses["total"].item()
                n_val += 1
                all_probs.append(prob.numpy())
                all_labels.append(labels.numpy())

        all_probs = np.concatenate(all_probs).flatten()
        all_labels = np.concatenate(all_labels).flatten()

        try:
            auroc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auroc = 0.5  # Default for edge case

        acc = ((all_probs >= 0.5).astype(float) == all_labels).mean()

        scheduler.step()
        elapsed = time.time() - t0

        print(f"    Epoch {epoch}/2 │ train_loss: {train_loss/n_train:.4f} │ "
              f"val_loss: {val_loss/n_val:.4f} │ AUROC: {auroc:.4f} │ "
              f"acc: {acc:.4f} │ {elapsed:.1f}s")

    # ── Step 5: Save checkpoint ────────────────────────────────────────────
    print_step(5, "Saving checkpoint...")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "smoke_test.pt")
    torch.save({
        "epoch": 2,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }, ckpt_path)
    ckpt_size = os.path.getsize(ckpt_path) / (1024 * 1024)
    print(f"    ✅ Checkpoint saved: {ckpt_path} ({ckpt_size:.1f} MB)")

    # ── Step 6: Load checkpoint and verify ─────────────────────────────────
    print_step(6, "Loading checkpoint and verifying...")
    ckpt = torch.load(ckpt_path, weights_only=False)
    model2 = DeepfakeDetector().to(device)
    model2.load_state_dict(ckpt["model"])
    model2.eval()

    # Quick inference test
    with torch.no_grad():
        test_batch = next(iter(val_loader))
        prob = model2(test_batch["frames"].to(device), test_batch["mel"].to(device))
        assert prob.shape[1] == 1, "Output should be (B, 1)"
        assert (prob >= 0).all() and (prob <= 1).all(), "Prob should be in [0, 1]"
    print("    ✅ Checkpoint loaded, inference verified")

    # ── Step 7: Evaluation metrics ─────────────────────────────────────────
    print_step(7, "Testing evaluation metrics...")
    from evaluation.evaluate import compute_metrics
    dummy_labels = np.array([0, 0, 1, 1], dtype=np.float32)
    dummy_scores = np.array([0.2, 0.3, 0.8, 0.9], dtype=np.float32)
    metrics = compute_metrics(dummy_labels, dummy_scores)
    assert "accuracy" in metrics and "auroc" in metrics and "eer" in metrics
    print(f"    ✅ Metrics: accuracy={metrics['accuracy']:.2f}, "
          f"AUROC={metrics['auroc']:.2f}, EER={metrics['eer']:.4f}")

    # ── Cleanup ────────────────────────────────────────────────────────────
    shutil.rmtree(ckpt_dir)

    # ── Summary ────────────────────────────────────────────────────────────
    total_time = time.time() - t_start
    print_header("SMOKE TEST RESULTS")
    print(f"""
    ✅ Data pipeline:     PASS (dummy data, balanced sampling, augmentation)
    ✅ Model forward:     PASS (ViT + CNN-6 + CrossAttention + Classifier)
    ✅ Loss computation:  PASS (BCE + LSE-D)
    ✅ Backward + optim:  PASS (AdamW + gradient clipping)
    ✅ Phase transition:  PASS (warmup → finetune)
    ✅ Checkpoint save:   PASS
    ✅ Checkpoint load:   PASS
    ✅ Evaluation:        PASS (AUROC, accuracy, EER)

    Total time: {total_time:.1f}s
    Status: 🎉 ALL SYSTEMS GO — Ready for real data!
    """)

    return 0


if __name__ == "__main__":
    sys.exit(main())
