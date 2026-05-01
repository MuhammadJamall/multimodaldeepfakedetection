"""
scheduler.py
------------
Learning rate schedule: linear warm-up → cosine annealing.

Epochs 1–5  : LR rises linearly from 0 → base_lr   (warm-up)
Epochs 6–30 : LR decays following a cosine curve    (annealing)
              reaching a minimum of eta_min (default 1e-6)

Works correctly with differential learning rate groups because
PyTorch's LambdaLR applies the same multiplicative factor to
every param group's *own* base lr.
"""

import math
import torch
from torch.optim.lr_scheduler import LambdaLR


def build_scheduler(
    optimizer:     torch.optim.Optimizer,
    warmup_epochs: int   = 5,
    total_epochs:  int   = 30,
    eta_min_ratio: float = 1e-6,   # eta_min as a fraction of base lr
) -> LambdaLR:
    """
    Returns a LambdaLR scheduler implementing:

        epoch ∈ [1, warmup_epochs]          → factor = epoch / warmup_epochs
        epoch ∈ (warmup_epochs, total_epochs] → cosine decay from 1.0 → eta_min_ratio

    Args:
        optimizer     : AdamW optimizer (any param groups).
        warmup_epochs : Number of linear warm-up epochs (default 5).
        total_epochs  : Total training epochs (default 30).
        eta_min_ratio : Minimum LR expressed as a ratio of base LR (default 1e-6).

    Usage:
        scheduler = build_scheduler(optimizer)
        for epoch in range(1, total_epochs + 1):
            train(...)
            scheduler.step()   # call AFTER the epoch, not per batch
    """

    cosine_epochs = total_epochs - warmup_epochs   # 25 epochs of cosine

    def lr_lambda(epoch: int) -> float:
        """
        epoch is 0-indexed internally by LambdaLR (starts at 0).
        We shift to 1-indexed for readability.
        """
        e = epoch + 1   # 1-indexed epoch number

        # ── Warm-up phase ─────────────────────────────────────────────────
        if e <= warmup_epochs:
            return float(e) / float(warmup_epochs)

        # ── Cosine annealing phase ─────────────────────────────────────────
        progress = float(e - warmup_epochs) / float(cosine_epochs)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))

        # Scale between eta_min_ratio and 1.0
        return eta_min_ratio + (1.0 - eta_min_ratio) * cosine_factor

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


# ── Smoke test + visualisation ────────────────────────────────────────────────

if __name__ == "__main__":
    import torch.optim as optim

    # Dummy model for param groups
    dummy_params = [
        {"params": [torch.nn.Parameter(torch.randn(1))], "lr": 1e-4, "name": "backbone"},
        {"params": [torch.nn.Parameter(torch.randn(1))], "lr": 1e-3, "name": "fusion"},
    ]
    optimizer = optim.AdamW(dummy_params, weight_decay=1e-2)
    scheduler = build_scheduler(optimizer, warmup_epochs=5, total_epochs=30)

    print("Epoch │ backbone LR      │ fusion LR")
    print("──────┼──────────────────┼──────────────────")

    backbone_lrs, fusion_lrs = [], []

    for epoch in range(1, 31):
        backbone_lr = optimizer.param_groups[0]["lr"]
        fusion_lr   = optimizer.param_groups[1]["lr"]
        backbone_lrs.append(backbone_lr)
        fusion_lrs.append(fusion_lr)
        print(f"  {epoch:2d}  │ {backbone_lr:.8f}   │ {fusion_lr:.8f}")
        scheduler.step()

    # Verify warm-up end values
    assert abs(backbone_lrs[4] - 1e-4) < 1e-9, "Warm-up end LR mismatch for backbone"
    assert abs(fusion_lrs[4]   - 1e-3) < 1e-8, "Warm-up end LR mismatch for fusion"

    # Verify decay direction
    assert backbone_lrs[5] < backbone_lrs[4], "LR should drop after warm-up"
    assert backbone_lrs[-1] < backbone_lrs[5], "LR should continue to decay"

    print("\n✅  Scheduler smoke test passed.")

    # ── Optional: ASCII plot ───────────────────────────────────────────────
    print("\n── backbone LR curve (scaled) ──")
    max_lr = max(backbone_lrs)
    width  = 40
    for i, lr in enumerate(backbone_lrs):
        bar_len = int((lr / max_lr) * width)
        bar     = "█" * bar_len
        marker  = " ← warmup ends" if i == 4 else ""
        print(f"  E{i+1:2d} {bar:<40} {lr:.2e}{marker}")