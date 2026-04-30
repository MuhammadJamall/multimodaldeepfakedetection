from __future__ import annotations

import torch
from torch import nn


def lse_d_loss(logits: torch.Tensor, labels: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    pos = logits[labels > 0.5]
    neg = logits[labels <= 0.5]
    if pos.numel() == 0 or neg.numel() == 0:
        return logits.new_tensor(0.0)
    loss = torch.logsumexp(neg, dim=0) + torch.logsumexp(-pos, dim=0) + margin
    return loss


def combined_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    bce_weight: float = 1.0,
    lse_weight: float = 0.5,
    lse_margin: float = 1.0,
) -> torch.Tensor:
    bce = nn.BCEWithLogitsLoss()(logits, labels)
    lse = lse_d_loss(logits, labels, margin=lse_margin)
    return bce_weight * bce + lse_weight * lse
