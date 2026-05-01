"""
train.py
--------
Main training script for DeepfakeDetector.

Implements:
  - Phase 1 (Epochs 1–5)  : backbone frozen, fusion+head trained
  - Phase 2 (Epochs 6–30) : full end-to-end fine-tuning
  - AdamW with differential LR + gradient clipping
  - Linear warmup → cosine annealing
  - BCE + LSE-D combined loss
  - Validation after every epoch (AUROC-based best checkpoint saving)
  - Weights & Biases logging

Usage:
    python training/train.py --config configs/default.yaml
"""

import os
import sys
import argparse
import yaml
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    import wandb  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    class _WandbStub:
        def init(self, *args, **kwargs):
            return None

        def log(self, *args, **kwargs):
            return None

        def finish(self, *args, **kwargs):
            return None

    wandb = _WandbStub()
from sklearn.metrics import roc_auc_score
import numpy as np

# Make sure project root is on path when running as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.detector   import DeepfakeDetector
from training.losses   import DeepfakeLoss
from training.scheduler import build_scheduler


# ── Helpers ───────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        print("[Device] CPU (no GPU found — training will be slow)")
    return dev


# ── Epoch runners ─────────────────────────────────────────────────────────────

def train_epoch(
    model:     DeepfakeDetector,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: DeepfakeLoss,
    device:    torch.device,
    grad_clip: float,
    epoch:     int,
) -> dict:
    """
    Run one training epoch.

    Returns dict of averaged losses for W&B logging.
    """
    model.train()
    # CRITICAL: if backbones are frozen, keep them in eval() mode
    # to avoid corrupting BatchNorm running statistics.
    if epoch <= 5:
        model.visual_encoder.vit.eval()
        for m in [model.audio_encoder.conv1, model.audio_encoder.conv2,
                  model.audio_encoder.conv3, model.audio_encoder.conv4,
                  model.audio_encoder.conv5, model.audio_encoder.conv6]:
            m.eval()

    total_loss = bce_loss = lsed_loss = 0.0
    n_batches  = 0

    for batch in loader:
        frames = batch["frames"].to(device)   # (B, T, 6, 224, 224)
        mel    = batch["mel"].to(device)      # (B, T, 80, F)
        labels = batch["label"].to(device)    # (B, 1) float

        optimizer.zero_grad()

        prob, embeddings = model(frames, mel, return_embeddings=True)
        losses = criterion(prob, embeddings["v"], embeddings["a"], labels)

        losses["total"].backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()

        total_loss += losses["total"].item()
        bce_loss   += losses["bce"].item()
        lsed_loss  += losses["lsed"].item()
        n_batches  += 1

    return {
        "train/loss_total": total_loss / n_batches,
        "train/loss_bce":   bce_loss   / n_batches,
        "train/loss_lsed":  lsed_loss  / n_batches,
    }


@torch.no_grad()
def val_epoch(
    model:     DeepfakeDetector,
    loader:    DataLoader,
    criterion: DeepfakeLoss,
    device:    torch.device,
) -> dict:
    """
    Run one validation epoch.

    Returns dict of losses + AUROC for checkpoint logic and W&B logging.
    """
    model.eval()

    total_loss = bce_loss = lsed_loss = 0.0
    n_batches  = 0

    all_probs  = []
    all_labels = []

    for batch in loader:
        frames = batch["frames"].to(device)
        mel    = batch["mel"].to(device)
        labels = batch["label"].to(device)

        prob, embeddings = model(frames, mel, return_embeddings=True)
        losses = criterion(prob, embeddings["v"], embeddings["a"], labels)

        total_loss += losses["total"].item()
        bce_loss   += losses["bce"].item()
        lsed_loss  += losses["lsed"].item()
        n_batches  += 1

        all_probs.append(prob.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_probs  = np.concatenate(all_probs,  axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    auroc = roc_auc_score(all_labels, all_probs)
    acc   = ((all_probs >= 0.5).astype(float) == all_labels).mean()

    return {
        "val/loss_total": total_loss / n_batches,
        "val/loss_bce":   bce_loss   / n_batches,
        "val/loss_lsed":  lsed_loss  / n_batches,
        "val/auroc":      auroc,
        "val/accuracy":   acc,
    }


# ── Checkpoint utils ──────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, scheduler, epoch, metrics, cfg, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch":     epoch,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "metrics":   metrics,
        "config":    cfg,
    }, path)
    print(f"  [Checkpoint] Saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def train(cfg: dict):
    set_seed(cfg.get("seed", 42))
    device = get_device()

    # ── W&B init ───────────────────────────────────────────────────────────
    log_cfg = cfg.get("logging", {})
    wandb.init(
        project=log_cfg.get("wandb_project", "deepfake-detection"),
        config=cfg,
        name=cfg.get("run_name", f"run_{int(time.time())}"),
    )

    # ── Model ──────────────────────────────────────────────────────────────
    mcfg = cfg["model"]
    model = DeepfakeDetector(
        vit_model        = mcfg["vit_model"],
        vit_hidden_dim   = mcfg["vit_hidden_dim"],
        audio_hidden_dim = mcfg["audio_hidden_dim"],
        num_heads        = mcfg["num_heads"],
        ffn_hidden_dim   = mcfg["ffn_hidden_dim"],
        dropout          = mcfg["dropout"],
    ).to(device)

    model.set_warmup_mode()   # Phase 1: freeze backbones

    # ── Optimizer ──────────────────────────────────────────────────────────
    tcfg = cfg["training"]
    param_groups = model.get_param_groups(
        lr_backbone = tcfg["learning_rate_vit"],
        lr_fusion   = tcfg["learning_rate_fusion"],
    )
    optimizer = optim.AdamW(
        param_groups,
        betas        = (0.9, 0.999),
        weight_decay = tcfg["weight_decay"],
    )

    # ── Scheduler ──────────────────────────────────────────────────────────
    scheduler = build_scheduler(
        optimizer,
        warmup_epochs = tcfg["warmup_epochs"],
        total_epochs  = tcfg["epochs"],
    )

    # ── Loss ───────────────────────────────────────────────────────────────
    lcfg = cfg.get("loss", {})
    criterion = DeepfakeLoss(
        lse_d_lambda = lcfg.get("lse_d_lambda", 0.3),
        lse_d_margin = lcfg.get("lse_d_margin", 1.0),
    )

    # ── Data loaders ───────────────────────────────────────────────────────
    # NOTE: Replace these with real Dataset classes once data is available.
    # For now we use a stub that returns random tensors of the correct shape,
    # so the training loop can be smoke-tested end-to-end.
    from data.dataset import build_dataloaders   # noqa — imported lazily

    train_loader, val_loader = build_dataloaders(cfg)

    # ── Training loop ──────────────────────────────────────────────────────
    best_auroc   = 0.0
    ckpt_dir     = cfg.get("checkpoint_dir", "checkpoints")

    for epoch in range(1, tcfg["epochs"] + 1):
        t0 = time.time()

        # ── Phase transition ───────────────────────────────────────────────
        if epoch == tcfg["warmup_epochs"] + 1:
            model.set_finetune_mode()
            print(f"[Epoch {epoch}] Switched to fine-tune mode.")

        # ── Train + validate ───────────────────────────────────────────────
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion,
            device, tcfg["gradient_clip"], epoch,
        )
        val_metrics = val_epoch(model, val_loader, criterion, device)

        scheduler.step()

        # ── Logging ────────────────────────────────────────────────────────
        epoch_time = time.time() - t0
        log_dict   = {**train_metrics, **val_metrics,
                      "epoch": epoch,
                      "epoch_time_s": epoch_time,
                      "lr/backbone": optimizer.param_groups[0]["lr"],
                      "lr/fusion":   optimizer.param_groups[-1]["lr"]}
        wandb.log(log_dict)

        auroc = val_metrics["val/auroc"]
        acc   = val_metrics["val/accuracy"]
        print(
            f"Epoch {epoch:3d}/{tcfg['epochs']} │ "
            f"loss {train_metrics['train/loss_total']:.4f} │ "
            f"val_auroc {auroc:.4f} │ "
            f"val_acc {acc:.4f} │ "
            f"{epoch_time:.1f}s"
        )

        # ── Checkpoint: save best AUROC model ──────────────────────────────
        if auroc > best_auroc:
            best_auroc = auroc
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                val_metrics, cfg,
                path=os.path.join(ckpt_dir, "best_auroc.pt"),
            )

        # ── Checkpoint: save latest (for resuming) ─────────────────────────
        save_checkpoint(
            model, optimizer, scheduler, epoch,
            val_metrics, cfg,
            path=os.path.join(ckpt_dir, "latest.pt"),
        )

    print(f"\nTraining complete. Best val AUROC: {best_auroc:.4f}")
    wandb.finish()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DeepfakeDetector")
    parser.add_argument("--config", type=str,
                        default="configs/default.yaml",
                        help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg)