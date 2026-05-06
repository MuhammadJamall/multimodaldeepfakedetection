"""
cross_attention.py
------------------
Bidirectional Cross-Attention Fusion Module.

Takes visual embeddings v ∈ ℝ^(B×T×512) and
      audio  embeddings a ∈ ℝ^(B×T×512).

Computes:
  v' = Attn(Q=v, K=a, V=a)   — visual attends to audio
  a' = Attn(Q=a, K=v, V=v)   — audio  attends to visual

"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# Check if PyTorch 2.0+ scaled_dot_product_attention is available
_SDPA_AVAILABLE = hasattr(F, "scaled_dot_product_attention")


class MultiHeadCrossAttention(nn.Module):
    """
    Multi-head cross-attention: Query from X, Key/Value from Y.

    Attn(X→Y) = softmax( QK^T / √d_k ) V
    Output    = LayerNorm( X + Proj(attention_output) )

    Uses F.scaled_dot_product_attention (FlashAttention) on PyTorch 2.0+
    for faster and more memory-efficient computation.
    """

    def __init__(self, embed_dim: int = 512, num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = math.sqrt(self.head_dim)
        self.dropout   = dropout

        self.q_proj   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.attn_drop  = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Query source  (B, T, embed_dim)
            y : Key/Value src (B, T, embed_dim)

        Returns:
            out : (B, T, embed_dim) — x enriched with info from y
        """
        B, T, D = x.shape

        Q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(y).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(y).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        if _SDPA_AVAILABLE:
            # PyTorch 2.0+ — uses FlashAttention kernel when on CUDA
            # faster and O(N) memory vs O(N²) for standard attention
            attn_output = F.scaled_dot_product_attention(
                Q, K, V,
                dropout_p=self.dropout if self.training else 0.0,
            )
        else:
            # Fallback: manual scaled dot-product attention
            attn_scores  = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.attn_drop(attn_weights)
            attn_output  = torch.matmul(attn_weights, V)

        # Merge heads: (B, H, T, d) → (B, T, D)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D)

        # Residual + LayerNorm
        return self.layer_norm(x + self.out_proj(attn_output))


class FeedForward(nn.Module):
    """
    Position-wise FFN: Linear → GELU → Dropout → Linear → residual + LN.
    Standard Transformer FFN, added to improve representation after attention.
    """

    def __init__(self, embed_dim: int = 512, ffn_dim: int = 2048,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_norm(x + self.net(x))


class CrossAttentionLayer(nn.Module):
    """One full cross-attention layer: cross-attn + FFN for both streams."""

    def __init__(self, embed_dim: int = 512, num_heads: int = 8,
                 ffn_dim: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.v_to_a = MultiHeadCrossAttention(embed_dim, num_heads, dropout)
        self.a_to_v = MultiHeadCrossAttention(embed_dim, num_heads, dropout)
        self.ffn_v  = FeedForward(embed_dim, ffn_dim, dropout)
        self.ffn_a  = FeedForward(embed_dim, ffn_dim, dropout)

    def forward(self, v: torch.Tensor,
                a: torch.Tensor):
        v_out = self.ffn_v(self.v_to_a(x=v, y=a))
        a_out = self.ffn_a(self.a_to_v(x=a, y=v))
        return v_out, a_out


class CrossAttentionFusion(nn.Module):
    """
    Full bidirectional cross-attention fusion with N stacked layers.

    v' = CrossAttn(Q=v, K=a, V=a) + FFN   [visual attends to audio]
    a' = CrossAttn(Q=a, K=v, V=v) + FFN   [audio  attends to visual]

    Both streams are mean-pooled over T, then concatenated → (B, 2*embed_dim)
    for the classifier head.

    num_layers=1 (default) matches original design and avoids overfitting
    on small datasets like FakeAVCeleb. Increase to 2 for larger datasets.
    """

    def __init__(self, embed_dim: int = 512, num_heads: int = 8,
                 dropout: float = 0.1, num_layers: int = 1,
                 ffn_dim: int = 2048):
        """
        Args:
            embed_dim  : Embedding dimension for both streams (512).
            num_heads  : Attention heads (8).
            dropout    : Dropout on attention weights and FFN (0.1).
                         FIX: v1 hardcoded 0.0 regardless of this argument.
            num_layers : Number of stacked cross-attention layers (default 1).
            ffn_dim    : FFN hidden dimension (default 2048 = 4×embed_dim).
        """
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttentionLayer(embed_dim, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        self.out_dim = embed_dim * 2   # for classifier head sizing

    def forward(self, v: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Args:
            v : Visual embeddings (B, T, embed_dim)
            a : Audio  embeddings (B, T, embed_dim)

        Returns:
            fused : (B, 2*embed_dim)
        """
        if v.shape != a.shape:
            raise ValueError(
                f"Visual and audio embeddings must have the same shape. "
                f"Got v={v.shape}, a={a.shape}"
            )

        for layer in self.layers:
            v, a = layer(v, a)

        v_pooled = v.mean(dim=1)               # (B, embed_dim)
        a_pooled = a.mean(dim=1)               # (B, embed_dim)
        return torch.cat([v_pooled, a_pooled], dim=-1)   # (B, 2*embed_dim)


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing CrossAttentionFusion…")
    print(f"  FlashAttention (SDPA) available: {_SDPA_AVAILABLE}")

    fusion = CrossAttentionFusion(
        embed_dim=512, num_heads=8, dropout=0.1, num_layers=1
    )
    fusion.eval()

    v_dummy = torch.randn(2, 16, 512)
    a_dummy = torch.randn(2, 16, 512)

    with torch.no_grad():
        out = fusion(v_dummy, a_dummy)

    assert out.shape == (2, 1024), f"Shape mismatch: {out.shape}"
    print(f"Visual input : {v_dummy.shape}")
    print(f"Audio  input : {a_dummy.shape}")
    print(f"Fused  output: {out.shape}")

    # Test shape mismatch raises properly
    try:
        fusion(torch.randn(2, 16, 512), torch.randn(2, 8, 512))
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        print(f"ValueError raised correctly ✅")

    # Test 2-layer version
    fusion2 = CrossAttentionFusion(num_layers=2)
    with torch.no_grad():
        out2 = fusion2(v_dummy, a_dummy)
    assert out2.shape == (2, 1024)
    print(f"2-layer fusion: {out2.shape} ✅")

    print("✅  CrossAttentionFusion smoke test passed.")