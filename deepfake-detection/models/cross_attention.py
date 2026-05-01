"""
cross_attention.py
------------------
Bidirectional Cross-Attention Fusion Module.

Takes visual embeddings v ∈ ℝ^(B×T×512) and
      audio  embeddings a ∈ ℝ^(B×T×512).

Computes:
  v' = Attn(Q=v, K=a, V=a)   — visual attends to audio
  a' = Attn(Q=a, K=v, V=v)   — audio  attends to visual

Outputs: concatenation of mean-pooled v' and a' → (B, 1024)
         which feeds directly into the classifier head.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadCrossAttention(nn.Module):
    """
    Standard scaled dot-product multi-head cross-attention.

    Query comes from stream X; Key and Value from stream Y.
    Attn(X→Y) = softmax( (Q_X · K_Y^T) / √d_k ) · V_Y

    Then: output = LayerNorm( X + linear_proj(attention_output) )
    """

    def __init__(self, embed_dim: int = 512, num_heads: int = 8,
                 dropout: float = 0.0):
        """
        Args:
            embed_dim : Dimensionality of input embeddings (512).
            num_heads : Number of attention heads (8).
            dropout   : Dropout on attention weights (0 by default).
        """
        super().__init__()
        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads   # 64
        self.scale     = math.sqrt(self.head_dim)

        # Projections for Query (from X), Key and Value (from Y)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.attn_drop = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Query source  (B, T, embed_dim)
            y : Key/Value src (B, T, embed_dim)

        Returns:
            out : (B, T, embed_dim)  — x enriched with information from y
        """
        B, T, D = x.shape

        # ── Linear projections ────────────────────────────────────────────────
        Q = self.q_proj(x)   # (B, T, D)
        K = self.k_proj(y)   # (B, T, D)
        V = self.v_proj(y)   # (B, T, D)

        # ── Split into heads: (B, T, D) → (B, num_heads, T, head_dim) ─────────
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # ── Scaled dot-product attention ──────────────────────────────────────
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H, T, T)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_drop(attn_weights)

        attn_output = torch.matmul(attn_weights, V)       # (B, H, T, head_dim)

        # ── Merge heads ───────────────────────────────────────────────────────
        attn_output = attn_output.transpose(1, 2).contiguous()  # (B, T, H, head_dim)
        attn_output = attn_output.view(B, T, D)                 # (B, T, D)

        # ── Output projection + residual + LayerNorm ──────────────────────────
        out = self.layer_norm(x + self.out_proj(attn_output))   # (B, T, D)

        return out


class CrossAttentionFusion(nn.Module):
    """
    Full bidirectional cross-attention fusion module.

    v' = CrossAttn(Q=v, K=a, V=a)   [visual attends to audio]
    a' = CrossAttn(Q=a, K=v, V=v)   [audio  attends to visual]

    Both streams are mean-pooled over the time dimension T,
    then concatenated → (B, 1024) for the classifier head.

    Design choice: single attention layer (no stacking) to avoid
    overfitting on the relatively small FakeAVCeleb dataset.
    """

    def __init__(self, embed_dim: int = 512, num_heads: int = 8,
                 dropout: float = 0.0):
        super().__init__()

        # Visual attends to audio
        self.v_to_a = MultiHeadCrossAttention(embed_dim, num_heads, dropout)
        # Audio attends to visual
        self.a_to_v = MultiHeadCrossAttention(embed_dim, num_heads, dropout)

    def forward(self, v: torch.Tensor,
                a: torch.Tensor) -> torch.Tensor:
        """
        Args:
            v : Visual embeddings (B, T, 512)
            a : Audio  embeddings (B, T, 512)

        Returns:
            fused : (B, 1024)  — concatenation of mean-pooled attended streams
        """
        # Bidirectional cross-attention
        v_prime = self.v_to_a(x=v, y=a)   # (B, T, 512)
        a_prime = self.a_to_v(x=a, y=v)   # (B, T, 512)

        # Temporal mean-pooling (collapses T dimension)
        v_pooled = v_prime.mean(dim=1)     # (B, 512)
        a_pooled = a_prime.mean(dim=1)     # (B, 512)

        # Concatenate both streams
        fused = torch.cat([v_pooled, a_pooled], dim=-1)  # (B, 1024)

        return fused


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing CrossAttentionFusion…")
    fusion = CrossAttentionFusion(embed_dim=512, num_heads=8)
    fusion.eval()

    v_dummy = torch.randn(2, 16, 512)   # B=2, T=16
    a_dummy = torch.randn(2, 16, 512)

    with torch.no_grad():
        out = fusion(v_dummy, a_dummy)

    print(f"Visual input shape : {v_dummy.shape}")
    print(f"Audio  input shape : {a_dummy.shape}")
    print(f"Fused  output shape: {out.shape}")    # Expected: (2, 1024)
    assert out.shape == (2, 1024), "Shape mismatch!"
    print("✅  CrossAttentionFusion smoke test passed.")