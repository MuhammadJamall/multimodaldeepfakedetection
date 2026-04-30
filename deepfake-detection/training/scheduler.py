from __future__ import annotations

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR


def build_scheduler(optimizer: Optimizer, epochs: int) -> CosineAnnealingLR:
    return CosineAnnealingLR(optimizer, T_max=epochs)
