"""
audio_encoder.py
----------------
Mel-CNN Audio Encoder (CNN-6 architecture).

Input tensor shape : (B, T, 80, F)
  B  = batch size
  T  = number of time windows (must match visual frames, default 16)
  80 = Mel frequency bins
  F  = number of time frames per window (derived from audio duration / T)

Output tensor shape: (B, T, 512)
  Each time window produces a 512-dim embedding.
  Kept as (B, T, 512) for cross-attention with visual stream.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Single Conv → BN → ReLU → MaxPool block."""

    def __init__(self, in_channels: int, out_channels: int,
                 pool_size: tuple = (2, 2)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1, bias=False)
        self.bn   = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(pool_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(F.relu(self.bn(self.conv(x))))


class AudioEncoder(nn.Module):
    """
    CNN-6 encoder for Mel spectrograms.

    Architecture (6 convolutional layers):
        Input   : (B*T,  1, 80, F)
        Conv1   : 1   → 64    + BN + ReLU + MaxPool(2,2)
        Conv2   : 64  → 128   + BN + ReLU + MaxPool(2,2)
        Conv3   : 128 → 256   + BN + ReLU + MaxPool(2,2)
        Conv4   : 256 → 512   + BN + ReLU + MaxPool(2,2)
        Conv5   : 512 → 512   + BN + ReLU + MaxPool(2,2)
        Conv6   : 512 → 512   + BN + ReLU + MaxPool(1,1)  ← no spatial reduction
        GAP     : Global Average Pool → (B*T, 512)
        Proj    : Linear 512 → 512  (identity by default, kept for flexibility)

    The 6 MaxPool layers reduce 80×F spatial dims down to near 1×1,
    after which Global Average Pooling collapses any remainder.
    """

    def __init__(self, mel_bins: int = 80, out_dim: int = 512,
                 freeze: bool = False):
        """
        Args:
            mel_bins : Number of Mel frequency bins (default 80).
            out_dim  : Output embedding dimension (default 512).
            freeze   : If True, freeze all CNN parameters (warm-up phase).
        """
        super().__init__()

        # ── 6 Conv blocks ─────────────────────────────────────────────────────
        self.conv1 = ConvBlock(1,    64,  pool_size=(2, 2))
        self.conv2 = ConvBlock(64,  128,  pool_size=(2, 2))
        self.conv3 = ConvBlock(128, 256,  pool_size=(2, 2))
        self.conv4 = ConvBlock(256, 512,  pool_size=(2, 2))
        self.conv5 = ConvBlock(512, 512,  pool_size=(2, 2))
        self.conv6 = ConvBlock(512, 512,  pool_size=(1, 1))  # no spatial reduction

        # ── Global Average Pooling ────────────────────────────────────────────
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # ── Linear projection ─────────────────────────────────────────────────
        self.proj = nn.Linear(512, out_dim)

        if freeze:
            self.freeze_backbone()

    # ── Public helpers ────────────────────────────────────────────────────────

    def freeze_backbone(self):
        """Freeze CNN weights; keep projection trainable."""
        backbone_modules = [self.conv1, self.conv2, self.conv3,
                            self.conv4, self.conv5, self.conv6, self.gap]
        for module in backbone_modules:
            for param in module.parameters():
                param.requires_grad = False
        # Set eval mode to freeze BatchNorm running statistics
        for module in backbone_modules:
            module.eval()

    def unfreeze_backbone(self):
        """Unfreeze CNN for fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True
        self.train()

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, T, 80, F)  — batch of T Mel spectrogram windows

        Returns:
            embeddings : (B, T, 512)
        """
        B, T, mel_bins, F = x.shape
        assert mel_bins == 80, f"Expected 80 Mel bins, got {mel_bins}"

        # Merge batch and time dims; add channel dim (CNN expects 1 channel)
        x_flat = x.view(B * T, 1, mel_bins, F)   # (B*T, 1, 80, F)

        # Forward through 6 Conv blocks
        h = self.conv1(x_flat)    # (B*T, 64,  40, F//2)
        h = self.conv2(h)         # (B*T, 128, 20, F//4)
        h = self.conv3(h)         # (B*T, 256, 10, F//8)
        h = self.conv4(h)         # (B*T, 512,  5, F//16)
        h = self.conv5(h)         # (B*T, 512,  2, F//32)
        h = self.conv6(h)         # (B*T, 512,  2, F//32)  ← pool(1,1) no change

        h = self.gap(h)           # (B*T, 512, 1, 1)
        h = h.view(B * T, -1)     # (B*T, 512)

        projected = self.proj(h)  # (B*T, 512)
        embeddings = projected.view(B, T, -1)  # (B, T, 512)

        return embeddings


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing AudioEncoder…")
    encoder = AudioEncoder(mel_bins=80, out_dim=512)
    encoder.eval()

    # F=32 → roughly what you get for short audio windows at 16kHz/10ms hop
    dummy = torch.randn(2, 16, 80, 32)   # B=2, T=16, 80 bins, 32 time frames
    with torch.no_grad():
        out = encoder(dummy)

    print(f"Input  shape : {dummy.shape}")
    print(f"Output shape : {out.shape}")   # Expected: (2, 16, 512)
    assert out.shape == (2, 16, 512), "Shape mismatch!"
    print("✅  AudioEncoder smoke test passed.")