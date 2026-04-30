from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ViT_B_16_Weights, vit_b_16


class VisualEncoder(nn.Module):
    def __init__(self, name: str = "vit_b_16", pretrained: bool = True, frozen: bool = False):
        super().__init__()
        if name != "vit_b_16":
            raise ValueError(f"Unsupported visual encoder: {name}")

        weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        self.model = vit_b_16(weights=weights)
        self.model.heads = nn.Identity()
        self.out_dim = self.model.hidden_dim

        if frozen:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 5:
            batch, frames, channels, height, width = x.shape
            x = x.view(batch * frames, channels, height, width)
            feats = self.model(x)
            feats = feats.view(batch, frames, -1).mean(dim=1)
            return feats
        return self.model(x)
