"""
interpretability.py
-------------------
Attention heatmap extraction from the CrossAttentionFusion module.

Hooks into the MultiHeadCrossAttention layers to capture attention weights
during a forward pass, enabling visualisation of which audio frames attend
to which visual frames (and vice versa).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.detector import DeepfakeDetector


def get_attention_maps(
    model: DeepfakeDetector, batch: Dict[str, torch.Tensor], device: str
) -> Dict[str, torch.Tensor]:
    """
    Extract cross-attention weight maps from a forward pass.

    Uses forward hooks on the fusion module's MultiHeadCrossAttention layers
    to capture attention weight matrices without modifying the model's forward().

    Args:
        model: DeepfakeDetector instance
        batch: Dict with 'frames' (B,T,6,224,224) and 'mel' (B,T,80,F)
        device: Device string

    Returns:
        Dict with:
          'v_to_a': (B, num_heads, T, T) visual-attends-to-audio weights
          'a_to_v': (B, num_heads, T, T) audio-attends-to-visual weights
    """
    model.eval()
    frames = batch["frames"].to(device)
    mel    = batch["mel"].to(device)

    attention_maps: Dict[str, torch.Tensor] = {}
    hooks: List[torch.utils.hooks.RemovableHook] = []

    def _make_hook(name: str):
        """Create a hook that captures attention scores from MultiHeadCrossAttention."""
        def hook_fn(module, input_args, output):
            # MultiHeadCrossAttention.forward takes (x, y) and internally computes
            # attn_scores = (Q @ K^T) / sqrt(d_k).  We recalculate from Q, K
            # stored in the module during forward.  Alternatively, we can register
            # hooks on the softmax step.  For simplicity, we re-derive here.
            x, y = input_args
            B, T, D = x.shape
            Q = module.q_proj(x).view(B, T, module.num_heads, module.head_dim).transpose(1, 2)
            K = module.k_proj(y).view(B, T, module.num_heads, module.head_dim).transpose(1, 2)
            scores = torch.matmul(Q, K.transpose(-2, -1)) / module.scale
            weights = torch.softmax(scores, dim=-1)  # (B, H, T, T)
            attention_maps[name] = weights.detach().cpu()
        return hook_fn

    # Register hooks on the two cross-attention sub-modules
    hooks.append(model.fusion.v_to_a.register_forward_hook(_make_hook("v_to_a")))
    hooks.append(model.fusion.a_to_v.register_forward_hook(_make_hook("a_to_v")))

    with torch.no_grad():
        model(frames, mel)

    # Clean up hooks
    for h in hooks:
        h.remove()

    return attention_maps


def summarize_attention(attn: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Average attention weights across heads → (B, T, T) per direction."""
    return {name: weights.mean(dim=1) for name, weights in attn.items()}
