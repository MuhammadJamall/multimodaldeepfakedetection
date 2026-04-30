from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.detector import DeepfakeDetector


def get_attention_maps(
    model: DeepfakeDetector, batch: Dict[str, torch.Tensor], device: str
) -> Dict[str, torch.Tensor]:
    model.eval()
    video = batch["video"].to(device)
    audio = batch["audio"].to(device)

    with torch.no_grad():
        _, attn = model(video, audio, return_attention=True)

    return attn


def summarize_attention(attn: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {name: weights.mean(dim=1).detach().cpu() for name, weights in attn.items()}
