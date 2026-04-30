from __future__ import annotations

from typing import Dict

import torch
from torch import nn


class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor):
        attn_out, attn_weights = self.attn(
            query, key_value, key_value, need_weights=True, average_attn_weights=False
        )
        out = self.norm(query + attn_out)
        return out, attn_weights


class BidirectionalCrossAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.visual_to_audio = CrossAttentionBlock(embed_dim, num_heads, dropout)
        self.audio_to_visual = CrossAttentionBlock(embed_dim, num_heads, dropout)

    def forward(self, visual_tokens: torch.Tensor, audio_tokens: torch.Tensor):
        v2a, v2a_weights = self.visual_to_audio(visual_tokens, audio_tokens)
        a2v, a2v_weights = self.audio_to_visual(audio_tokens, visual_tokens)
        return v2a, a2v, {"visual_to_audio": v2a_weights, "audio_to_visual": a2v_weights}
