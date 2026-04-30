from __future__ import annotations

import torch
from torch import nn

from models.audio_encoder import AudioEncoder
from models.cross_attention import BidirectionalCrossAttention
from models.visual_encoder import VisualEncoder


class DeepfakeDetector(nn.Module):
    def __init__(
        self,
        visual_name: str = "vit_b_16",
        visual_pretrained: bool = True,
        visual_frozen: bool = False,
        audio_in_channels: int = 1,
        audio_out_dim: int = 256,
        attention_embed_dim: int = 256,
        attention_heads: int = 8,
        attention_dropout: float = 0.1,
        fusion_hidden_dim: int = 512,
        fusion_dropout: float = 0.2,
    ):
        super().__init__()
        self.visual_encoder = VisualEncoder(
            name=visual_name, pretrained=visual_pretrained, frozen=visual_frozen
        )
        self.audio_encoder = AudioEncoder(in_channels=audio_in_channels, out_dim=audio_out_dim)
        self.visual_proj = nn.Linear(self.visual_encoder.out_dim, attention_embed_dim)
        self.audio_proj = nn.Linear(self.audio_encoder.out_dim, attention_embed_dim)
        self.cross_attention = BidirectionalCrossAttention(
            embed_dim=attention_embed_dim,
            num_heads=attention_heads,
            dropout=attention_dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(attention_embed_dim * 2, fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_hidden_dim, 1),
        )

    def forward(self, video: torch.Tensor, audio: torch.Tensor, return_attention: bool = False):
        visual_feat = self.visual_encoder(video)
        audio_feat = self.audio_encoder(audio)

        visual_tok = self.visual_proj(visual_feat).unsqueeze(1)
        audio_tok = self.audio_proj(audio_feat).unsqueeze(1)

        v2a, a2v, attn = self.cross_attention(visual_tok, audio_tok)
        fused = torch.cat([v2a.squeeze(1), a2v.squeeze(1)], dim=-1)
        logits = self.classifier(fused).squeeze(-1)

        if return_attention:
            return logits, attn
        return logits
