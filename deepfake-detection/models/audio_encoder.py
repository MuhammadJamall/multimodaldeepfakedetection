"""
audio_encoder.py
----------------
Mel-CNN Audio Encoder (CNN-6 architecture).

Input tensor shape : (B, T, 80, F)
  B  = batch size
  T  = number of time windows (must match visual frames, default 16)
  80 = Mel frequency bins
  F  = number of time frames per window

Output tensor shape: (B, T, 512)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ConvBlock(nn.Module):
    """Conv2d → BN → ReLU → MaxPool block."""

    def __init__(self, in_channels: int, out_channels: int,
                 pool_size: tuple = (2, 2)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1, bias=False)
        self.bn   = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(pool_size) if pool_size != (1, 1) else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(F.relu(self.bn(self.conv(x))))


class AudioEncoder(nn.Module):
    """
    CNN-6 encoder for Mel spectrograms.

    Architecture:
        Input   : (B*T,  1, 80, F)
        Conv1   : 1   →  64  + BN + ReLU + MaxPool(2,2)
        Conv2   : 64  → 128  + BN + ReLU + MaxPool(2,2)
        Conv3   : 128 → 256  + BN + ReLU + MaxPool(2,2)
        Conv4   : 256 → 512  + BN + ReLU + MaxPool(2,2)
        Conv5   : 512 → 512  + BN + ReLU + MaxPool(2,2)
        Conv6   : 512 → 512  + BN + ReLU  (no spatial reduction)
        GAP     : → (B*T, 512)
        Proj    : Linear 512 → out_dim
    """

    def __init__(self, mel_bins: int = 80, out_dim: int = 512,
                 freeze: bool = False):
        """
        Args:
            mel_bins : Number of Mel frequency bins (default 80).
            out_dim  : Output embedding dimension (default 512).
            freeze   : If True, freeze all CNN parameters immediately.
        """
        super().__init__()
        self._backbone_frozen: bool = False

        self.conv1 = ConvBlock(1,    64,  pool_size=(2, 2))
        self.conv2 = ConvBlock(64,  128,  pool_size=(2, 2))
        self.conv3 = ConvBlock(128, 256,  pool_size=(2, 2))
        self.conv4 = ConvBlock(256, 512,  pool_size=(2, 2))
        self.conv5 = ConvBlock(512, 512,  pool_size=(2, 2))
        self.conv6 = ConvBlock(512, 512,  pool_size=(1, 1))  # no spatial reduction

        self.gap  = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(512, out_dim)

        if freeze:
            self.freeze_backbone()

    # ── Freeze helpers ────────────────────────────────────────────────────────

    @property
    def _backbone_modules(self):
        return [self.conv1, self.conv2, self.conv3,
                self.conv4, self.conv5, self.conv6, self.gap]

    def freeze_backbone(self):
        """
        Freeze CNN weights and lock BN in eval mode.
        Idempotent — safe to call multiple times.
        """
        self._backbone_frozen = True
        for module in self._backbone_modules:
            for param in module.parameters():
                param.requires_grad = False
            module.eval()

    def unfreeze_backbone(self):
        """Unfreeze CNN for fine-tuning. Idempotent."""
        self._backbone_frozen = False
        for module in self._backbone_modules:
            for param in module.parameters():
                param.requires_grad = True
        self.train()

    def train(self, mode: bool = True):
        """
        Override nn.Module.train() to protect frozen BN running statistics.

        Problem: calling model.train() on the parent DeepfakeDetector
        recursively sets ALL submodules to train mode, which overrides
        the .eval() set by freeze_backbone() and corrupts BN stats.

        Fix: after super().train() runs, re-apply .eval() to frozen backbone
        modules so they are always in eval mode when frozen.
        """
        super().train(mode)
        if self._backbone_frozen:
            for module in self._backbone_modules:
                module.eval()
        return self

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, T, 80, F)

        Returns:
            embeddings : (B, T, out_dim)
        """
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input (B,T,mel,F), got {x.ndim}D")
        B, T, mel_bins, F = x.shape
        if mel_bins != 80:
            raise ValueError(f"Expected 80 Mel bins, got {mel_bins}")

        x_flat = x.view(B * T, 1, mel_bins, F)   # (B*T, 1, 80, F)

        h = self.conv1(x_flat)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)
        h = self.conv5(h)
        h = self.conv6(h)

        h = self.gap(h)                    # (B*T, 512, 1, 1)
        h = h.view(B * T, -1)             # (B*T, 512)
        projected = self.proj(h)           # (B*T, out_dim)

        return projected.view(B, T, -1)   # (B, T, out_dim)


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing AudioEncoder…")

    encoder = AudioEncoder(mel_bins=80, out_dim=512)
    encoder.eval()

    dummy = torch.randn(2, 16, 80, 32)
    with torch.no_grad():
        out = encoder(dummy)

    assert out.shape == (2, 16, 512), f"Shape mismatch: {out.shape}"
    print(f"Input  : {dummy.shape}")
    print(f"Output : {out.shape}")

    # Test freeze survives model.train()
    encoder.freeze_backbone()
    encoder.train()   # simulates training loop
    for m in encoder._backbone_modules:
        if hasattr(m, 'training'):
            assert not m.training, "Backbone should stay in eval after model.train()"
    print("Freeze survives model.train() ✅")

    print("✅  AudioEncoder smoke test passed.")