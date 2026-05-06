"""
visual_encoder.py
-----------------
Spatiotemporal Visual Encoder based on ViT-B/16.

Input tensor shape : (B, T, 6, 224, 224)
  B  = batch size
  T  = number of frames (default 16)
  6  = channels (full-face RGB + mouth-crop RGB stacked channel-wise)
  224×224 = spatial resolution

Output tensor shape: (B, T, 512)
"""

import torch
import torch.nn as nn
from typing import Optional
from transformers import ViTModel
from torch.nn.modules.utils import _pair


class VisualEncoder(nn.Module):
    """
    ViT-B/16 encoder adapted for 6-channel input.

    The standard ViT-B/16 expects 3-channel RGB input. We adapt it for
    6-channel input (full-face + mouth crop stacked) by re-initialising
    the patch-embedding projection:
      - First 3 channels ← pre-trained weights (face)
      - Next  3 channels ← copy of pre-trained weights (mouth, stable init)

    ViT-B/16 hidden dim : 768
    Output projection   : 768 → 512  (matches audio encoder dim)
    """

    VIT_HIDDEN_DIM = 768

    def __init__(
        self,
        model_name:  str = "google/vit-base-patch16-224-in21k",
        out_dim:     int = 512,
        freeze:      bool = False,
        chunk_size:  int = 32,
    ):
        """
        Args:
            model_name : HuggingFace ViT-B/16 identifier.
            out_dim    : Output embedding dimension (default 512).
            freeze     : Freeze ViT backbone immediately if True.
            chunk_size : Max frames processed through ViT per forward chunk.
                         Reduce if hitting OOM on Colab.
                         Default 32 is safe for T4 (16 GB) with B≤4.
                         Set to 8–16 for larger batches.
        """
        super().__init__()
        self._backbone_frozen: bool = False
        self.chunk_size = chunk_size

        # ── 1. Load pre-trained ViT ───────────────────────────────────────────
        self.vit = ViTModel.from_pretrained(model_name)

        # ── 2. Adapt patch embedding for 6-channel input ──────────────────────
        old_proj = self.vit.embeddings.patch_embeddings.projection

        # Extract padding safely — can be int or tuple depending on version
        raw_padding = old_proj.padding
        if isinstance(raw_padding, int):
            padding = raw_padding
        else:
            padding = _pair(raw_padding)

        new_proj = nn.Conv2d(
            in_channels=6,
            out_channels=old_proj.out_channels,
            kernel_size=_pair(old_proj.kernel_size),
            stride=_pair(old_proj.stride),
            padding=padding,
            bias=(old_proj.bias is not None),
        )

        with torch.no_grad():
            new_proj.weight[:, :3, :, :] = old_proj.weight.clone()  # face channels
            new_proj.weight[:, 3:, :, :] = old_proj.weight.clone()  # mouth channels
            if old_proj.bias is not None:
                assert new_proj.bias is not None
                new_proj.bias.copy_(old_proj.bias)

        self.vit.embeddings.patch_embeddings.projection = new_proj

        # Tell ViT internals to accept 6 channels
        self.vit.config.num_channels = 6
        self.vit.embeddings.patch_embeddings.num_channels = 6

        # ── 3. Projection 768 → out_dim ───────────────────────────────────────
        self.proj = nn.Linear(self.VIT_HIDDEN_DIM, out_dim)

        if freeze:
            self.freeze_backbone()

    # ── Freeze helpers ────────────────────────────────────────────────────────

    def freeze_backbone(self):
        """
        Freeze ViT weights and lock in eval mode.

        eval() is critical: ViT uses LayerNorm whose running stats must not
        update during the warm-up phase.
        Idempotent — safe to call multiple times.
        """
        self._backbone_frozen = True
        for param in self.vit.parameters():
            param.requires_grad = False
        self.vit.eval()

    def unfreeze_backbone(self):
        """Unfreeze ViT for fine-tuning. Idempotent."""
        self._backbone_frozen = False
        for param in self.vit.parameters():
            param.requires_grad = True
        self.vit.train()

    def train(self, mode: bool = True):
        """
        Override nn.Module.train() to protect frozen ViT stats.

        Problem: calling model.train() on the parent DeepfakeDetector
        recursively sets ALL submodules to train mode, overriding the
        .eval() set by freeze_backbone(). This corrupts LayerNorm/BN
        running statistics silently.

        Fix: after super().train(), re-apply .eval() to the ViT if frozen,
        so frozen state always wins regardless of external train() calls.
        """
        super().train(mode)
        if self._backbone_frozen:
            self.vit.eval()
        return self

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, T, 6, 224, 224)

        Returns:
            embeddings : (B, T, out_dim)
        """
        if x.ndim != 5:
            raise ValueError(f"Expected 5D input (B,T,C,H,W), got {x.ndim}D")

        B, T, C, H, W = x.shape

        if C != 6:
            raise ValueError(f"Expected 6 input channels, got {C}")
        if H != 224 or W != 224:
            raise ValueError(f"Expected 224×224 spatial dims, got {H}×{W}")

        # Merge batch and time dims → (B*T, 6, 224, 224)
        x_flat = x.view(B * T, C, H, W)

        # ── Chunked ViT forward to avoid OOM ─────────────────────────────────
        # Without chunking: B=32, T=16 → 512 images through ViT at once → OOM
        # With chunking: process self.chunk_size frames at a time → memory safe
        cls_list = []
        for chunk in x_flat.split(self.chunk_size, dim=0):
            out = self.vit(pixel_values=chunk)
            cls_list.append(out.last_hidden_state[:, 0, :])   # [CLS] token

        cls_tokens = torch.cat(cls_list, dim=0)   # (B*T, 768)
        projected  = self.proj(cls_tokens)         # (B*T, out_dim)

        return projected.view(B, T, -1)            # (B, T, out_dim)


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading VisualEncoder (downloads ViT weights on first run)…")

    encoder = VisualEncoder(freeze=True, chunk_size=8)
    encoder.eval()

    dummy = torch.randn(2, 16, 6, 224, 224)
    with torch.no_grad():
        out = encoder(dummy)

    assert out.shape == (2, 16, 512), f"Shape mismatch: {out.shape}"
    print(f"Input  : {dummy.shape}")
    print(f"Output : {out.shape}")

    # Test freeze survives model.train()
    encoder.freeze_backbone()
    encoder.train()   # simulates training loop call
    assert not encoder.vit.training, "ViT should stay in eval when frozen"
    print("Freeze survives model.train() ✅")

    # Test wrong channel input raises properly
    try:
        encoder(torch.randn(1, 4, 3, 224, 224))
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        print(f"ValueError raised correctly: {e} ✅")

    print("✅  VisualEncoder smoke test passed.")