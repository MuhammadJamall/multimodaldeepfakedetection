"""
losses.py
---------
Loss functions for DeepfakeDetector training.

Combined loss per CONTEXT.md §3.1:

    L_total = L_BCE + λ · L_LSE-D

BCE Loss:
    Standard binary cross-entropy on the model's sigmoid output.

LSE-D (Lip-Synchronization Error Distance):
    L2 distance between mean-pooled visual and audio embeddings.
    - For Real videos (y=0): minimize ||v - a||₂  (encourage sync)
    - For Fake videos (y=1): max(0, margin - ||v - a||₂)  (push apart)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DeepfakeLoss(nn.Module):
    """
    Combined BCE + LSE-D loss for multimodal deepfake detection.

    Usage in training loop:
        criterion = DeepfakeLoss(lse_d_lambda=0.3, lse_d_margin=1.0)
        losses = criterion(prob, v_emb, a_emb, labels)
        losses["total"].backward()
    """

    def __init__(self, lse_d_lambda: float = 0.3, lse_d_margin: float = 1.0):
        """
        Args:
            lse_d_lambda : Weight for LSE-D loss (λ in the formula).
            lse_d_margin : Margin m for fake video penalty. Default 1.0.
        """
        super().__init__()
        self.lse_d_lambda = lse_d_lambda
        self.lse_d_margin = lse_d_margin
        self.bce = nn.BCELoss()  # prob is already sigmoided

    def forward(
        self,
        prob: torch.Tensor,
        v_emb: torch.Tensor,
        a_emb: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Compute combined loss.

        Args:
            prob   : (B, 1) deepfake probability from classifier (sigmoid output)
            v_emb  : (B, 512) mean-pooled visual embeddings
            a_emb  : (B, 512) mean-pooled audio embeddings
            labels : (B, 1) ground-truth labels (0=real, 1=fake) as float

        Returns:
            Dict with keys "total", "bce", "lsed" — all scalar tensors.
        """
        # ── BCE loss ──────────────────────────────────────────────────────────
        bce_loss = self.bce(prob, labels)

        # ── LSE-D loss (L2 distance between modality embeddings) ──────────────
        # Per-sample L2 distance: ||v - a||₂
        l2_dist = torch.norm(v_emb - a_emb, p=2, dim=-1)  # (B,)

        labels_flat = labels.view(-1)  # (B,)

        # Real samples (y=0): minimize distance → loss = ||v - a||₂
        real_mask = (labels_flat < 0.5)
        # Fake samples (y=1): penalise small distance → loss = max(0, m - ||v - a||₂)
        fake_mask = (labels_flat >= 0.5)

        lsed = prob.new_tensor(0.0)
        count = 0

        if real_mask.any():
            lsed = lsed + l2_dist[real_mask].mean()
            count += 1

        if fake_mask.any():
            fake_lsed = torch.clamp(self.lse_d_margin - l2_dist[fake_mask], min=0.0)
            lsed = lsed + fake_lsed.mean()
            count += 1

        if count > 0:
            lsed = lsed / count

        # ── Combined ──────────────────────────────────────────────────────────
        total = bce_loss + self.lse_d_lambda * lsed

        return {
            "total": total,
            "bce":   bce_loss,
            "lsed":  lsed,
        }


# ── Legacy / utility functions (kept for backward compatibility) ──────────────

def lse_d_loss(v_emb: torch.Tensor, a_emb: torch.Tensor,
               labels: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """Standalone LSE-D loss computation."""
    l2_dist = torch.norm(v_emb - a_emb, p=2, dim=-1)
    labels_flat = labels.view(-1)

    real_loss = l2_dist[labels_flat < 0.5]
    fake_loss = torch.clamp(margin - l2_dist[labels_flat >= 0.5], min=0.0)

    parts = []
    if real_loss.numel() > 0:
        parts.append(real_loss.mean())
    if fake_loss.numel() > 0:
        parts.append(fake_loss.mean())

    if len(parts) == 0:
        return v_emb.new_tensor(0.0)
    return torch.tensor(sum(parts) / len(parts))


def combined_loss(
    prob: torch.Tensor,
    v_emb: torch.Tensor,
    a_emb: torch.Tensor,
    labels: torch.Tensor,
    bce_weight: float = 1.0,
    lse_weight: float = 0.3,
    lse_margin: float = 1.0,
) -> torch.Tensor:
    """Standalone combined loss (scalar output)."""
    bce = nn.BCELoss()(prob, labels)
    lse = lse_d_loss(v_emb, a_emb, labels, margin=lse_margin)
    return bce_weight * bce + lse_weight * lse