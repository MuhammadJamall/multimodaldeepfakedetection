"""
visual_encoder.py
-----------------
Spatiotemporal Visual Encoder based on ViT-B/16.

Input tensor shape : (B, T, 6, 224, 224)
  B  = batch size
  T  = number of frames (default 16)
  6  = channels (full-face RGB + mouth-crop RGB stacked channel-wise)
  224x224 = spatial resolution

Output tensor shape: (B, T, 512)
  Each frame produces a 512-dim embedding (projected from ViT hidden dim 768).
  Temporal aggregation (mean-pool) is done in detector.py to get (B, 512)
  OR kept as (B, T, 512) for cross-attention.
"""

import torch
import torch.nn as nn
from transformers import ViTModel
from torch.nn.modules.utils import _pair


class VisualEncoder(nn.Module):
    """
    ViT-B/16 encoder adapted for 6-channel input.

    The standard ViT-B/16 expects 3-channel (RGB) input.  We adapt it for
    6-channel input (full-face + mouth crop stacked) by re-initialising the
    patch-embedding projection weights:
      - The first 3 channels are initialised from the pre-trained weights.
      - The next  3 channels are initialised as a copy of those same weights
        (a common and stable strategy for channel duplication).

    Hidden dimension of ViT-B/16: 768
    Output projection: 768 → 512  (matches audio encoder output dim)
    """

    VIT_HIDDEN_DIM = 768
    OUT_DIM = 512

    def __init__(self, model_name: str = "google/vit-base-patch16-224-in21k",
                 out_dim: int = OUT_DIM, freeze: bool = False):
        """
        Args:
            model_name : HuggingFace model identifier for ViT-B/16.
            out_dim    : Output embedding dimension (default 512).
            freeze     : If True, freeze all ViT parameters (used in warm-up phase).
        """
        super().__init__()

        # ── 1. Load pre-trained ViT ──────────────────────────────────────────
        self.vit = ViTModel.from_pretrained(model_name)

        # ── 2. Adapt patch embedding for 6-channel input ─────────────────────
        old_proj = self.vit.embeddings.patch_embeddings.projection  # Conv2d(3, 768, 16, 16)

        kernel_size = _pair(old_proj.kernel_size)
        stride = _pair(old_proj.stride)
        padding = old_proj.padding
        if isinstance(padding, tuple):
            padding = _pair(padding)

        new_proj = nn.Conv2d(
            in_channels=6,
            out_channels=old_proj.out_channels,   # 768
            kernel_size=kernel_size,               # 16
            stride=stride,                         # 16
            padding=padding,                       # 0
            bias=(old_proj.bias is not None),
        )

        with torch.no_grad():
            # First 3 channels ← pre-trained weights
            new_proj.weight[:, :3, :, :] = old_proj.weight.clone()
            # Next  3 channels ← copy of pre-trained weights (stable init)
            new_proj.weight[:, 3:, :, :] = old_proj.weight.clone()
            if old_proj.bias is not None:
                if new_proj.bias is None:
                    raise RuntimeError("Expected new_proj to have a bias term.")
                new_proj.bias.copy_(old_proj.bias)

        self.vit.embeddings.patch_embeddings.projection = new_proj

        # ── 3. Linear projection 768 → 512 ───────────────────────────────────
        self.proj = nn.Linear(self.VIT_HIDDEN_DIM, out_dim)

        # ── 4. Optional backbone freeze (warm-up phase) ───────────────────────
        if freeze:
            self.freeze_backbone()

    # ── Public helpers ────────────────────────────────────────────────────────

    def freeze_backbone(self):
        """Freeze ViT weights; keep projection trainable."""
        for param in self.vit.parameters():
            param.requires_grad = False
        # IMPORTANT: also set eval mode to preserve BN running stats
        self.vit.eval()

    def unfreeze_backbone(self):
        """Unfreeze ViT for fine-tuning phase."""
        for param in self.vit.parameters():
            param.requires_grad = True
        self.vit.train()

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, T, 6, 224, 224)  — batch of T-frame clips

        Returns:
            embeddings : (B, T, 512)
        """
        B, T, C, H, W = x.shape
        assert C == 6,  f"Expected 6 input channels, got {C}"
        assert H == 224 and W == 224, f"Expected 224×224 frames, got {H}×{W}"

        # Merge batch and time dims so ViT processes each frame independently
        x_flat = x.view(B * T, C, H, W)          # (B*T, 6, 224, 224)

        outputs = self.vit(pixel_values=x_flat)   # last_hidden_state: (B*T, 197, 768)
        cls_tokens = outputs.last_hidden_state[:, 0, :]  # (B*T, 768)  ← [CLS] token

        projected = self.proj(cls_tokens)          # (B*T, 512)
        embeddings = projected.view(B, T, -1)      # (B, T, 512)

        return embeddings


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading VisualEncoder (this downloads ViT weights on first run)…")
    encoder = VisualEncoder(freeze=True)
    encoder.eval()

    dummy = torch.randn(2, 16, 6, 224, 224)   # B=2, T=16
    with torch.no_grad():
        out = encoder(dummy)

    print(f"Input  shape : {dummy.shape}")
    print(f"Output shape : {out.shape}")    # Expected: (2, 16, 512)
    assert out.shape == (2, 16, 512), "Shape mismatch!"
    print("✅  VisualEncoder smoke test passed.")