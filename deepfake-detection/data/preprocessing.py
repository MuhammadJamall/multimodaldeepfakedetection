from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import torch
from PIL import Image
import torchaudio
from torchvision import transforms

try:
    from facenet_pytorch import MTCNN
except ImportError:
    MTCNN = None


def build_face_extractor(image_size: int, device: Optional[str] = None):
    if MTCNN is None:
        return None
    return MTCNN(image_size=image_size, margin=0, post_process=True, device=device)


def crop_faces(
    frames: Iterable[np.ndarray],
    image_size: int,
    device: Optional[str] = None,
    extractor=None,
) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    if extractor is None and MTCNN is not None:
        extractor = build_face_extractor(image_size=image_size, device=device)

    if extractor is None:
        return torch.stack([transform(frame) for frame in frames], dim=0)

    pil_frames = [Image.fromarray(frame) for frame in frames]
    faces = extractor(pil_frames)

    if isinstance(faces, torch.Tensor):
        return faces

    return torch.stack([transform(frame) for frame in frames], dim=0)


def compute_mel_spectrogram(
    waveform: torch.Tensor,
    sample_rate: int,
    n_mels: int,
    hop_length: int,
    win_length: int,
) -> torch.Tensor:
    if waveform.dim() == 2 and waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_mels=n_mels,
        hop_length=hop_length,
        win_length=win_length,
    )(waveform)
    log_mel = torchaudio.transforms.AmplitudeToDB()(mel)
    return log_mel
