"""
augmentation.py
---------------
Compression and noise augmentation for deepfake detection training.

Per CONTEXT.md §4.3:
  - Training only (30% probability per video)
  - Validation/test sets remain clean
  - Per-video augmentation (applied at clip level, not individual frames)

Augmentation operations:
  - JPEG recompression (quality 40-80)
  - H.264 re-encoding (CRF 23-35)
  - Gaussian blur (sigma 0.5-2.0)
  - Temporal frame-dropping
  - Audio Gaussian noise
"""

from __future__ import annotations

import random
import subprocess
import tempfile
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch


# ── Visual Augmentations ─────────────────────────────────────────────────────

def jpeg_compress(image: np.ndarray, quality: int = 75) -> np.ndarray:
    """Apply JPEG recompression to a single frame."""
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, buffer = cv2.imencode(".jpg", image, encode_params)
    if not success:
        return image
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def gaussian_blur(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Apply Gaussian blur to a single frame."""
    # Kernel size must be odd, derived from sigma
    ksize = int(6 * sigma + 1) | 1  # Ensure odd
    ksize = max(ksize, 3)
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)


def add_gaussian_noise_image(image: np.ndarray, sigma: float = 5.0) -> np.ndarray:
    """Add Gaussian noise to a single frame."""
    noise = np.random.normal(0.0, sigma, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def h264_compress(input_path: str, output_path: str, crf: int = 23) -> None:
    """Apply H.264 re-encoding via ffmpeg subprocess."""
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", "fast",
        output_path,
    ]
    subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ── Temporal Augmentation ────────────────────────────────────────────────────

def temporal_frame_drop(
    frames: torch.Tensor,
    drop_prob: float = 0.2,
) -> torch.Tensor:
    """
    Randomly drop frames and duplicate neighbours to maintain T count.

    Args:
        frames: (T, C, H, W) visual tensor.
        drop_prob: Probability of dropping each frame (default 0.2).

    Returns:
        (T, C, H, W) tensor with some frames replaced by neighbours.
    """
    T = frames.shape[0]
    result = frames.clone()

    for i in range(T):
        if random.random() < drop_prob:
            # Replace with nearest valid neighbour
            if i > 0:
                result[i] = result[i - 1]
            elif i < T - 1:
                result[i] = frames[i + 1]

    return result


# ── Audio Augmentation ───────────────────────────────────────────────────────

def audio_gaussian_noise(
    mel: torch.Tensor,
    noise_std: float = 0.01,
) -> torch.Tensor:
    """
    Add Gaussian noise to Mel spectrogram.

    Args:
        mel: (T, mel_bins, F) Mel spectrogram tensor.
        noise_std: Standard deviation of noise (default 0.01).

    Returns:
        Noisy Mel spectrogram tensor.
    """
    noise = torch.randn_like(mel) * noise_std
    return mel + noise


# ── Master Augmentation Function ─────────────────────────────────────────────

def apply_augmentation(
    frames: torch.Tensor,
    mel: torch.Tensor,
    cfg: Optional[Dict] = None,
    is_training: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply augmentation pipeline to a single video clip.

    Per CONTEXT.md §4.3:
      - Training only (30% probability per video)
      - Validation/test sets remain clean

    Args:
        frames: (T, 6, H, W) visual tensor (values in [0, 1]).
        mel:    (T, mel_bins, F) audio tensor.
        cfg:    Augmentation config dict (from default.yaml augmentation section).
        is_training: If False, return inputs unchanged (val/test).

    Returns:
        (augmented_frames, augmented_mel) tuple.
    """
    if not is_training:
        return frames, mel

    if cfg is None:
        cfg = {}

    aug_prob = cfg.get("compression_augmentation_prob", 0.3)

    # Decide whether to augment this clip (30% probability)
    if random.random() > aug_prob:
        return frames, mel

    # ── Select and apply a random augmentation ─────────────────────────────
    aug_type = random.choice(["jpeg", "blur", "frame_drop", "audio_noise"])

    if aug_type == "jpeg":
        # JPEG recompression on visual frames
        quality_range = cfg.get("jpeg_quality_range", [40, 80])
        quality = random.randint(quality_range[0], quality_range[1])

        T, C, H, W = frames.shape
        augmented = []
        for t in range(T):
            # Convert tensor (6, H, W) → two BGR uint8 images for JPEG
            frame_np = (frames[t].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            # Apply JPEG to full 6-channel as two 3-channel images
            face_img = frame_np[:, :, :3]
            mouth_img = frame_np[:, :, 3:]
            face_compressed = jpeg_compress(face_img, quality)
            mouth_compressed = jpeg_compress(mouth_img, quality)
            # Reconstruct
            combined = np.concatenate([face_compressed, mouth_compressed], axis=-1)
            augmented.append(torch.from_numpy(combined).permute(2, 0, 1).float() / 255.0)

        frames = torch.stack(augmented, dim=0)

    elif aug_type == "blur":
        # Gaussian blur on visual frames
        sigma_range = cfg.get("blur_sigma_range", [0.5, 2.0])
        sigma = random.uniform(sigma_range[0], sigma_range[1])

        T, C, H, W = frames.shape
        augmented = []
        for t in range(T):
            frame_np = (frames[t].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            face_img = frame_np[:, :, :3]
            mouth_img = frame_np[:, :, 3:]
            face_blurred = gaussian_blur(face_img, sigma)
            mouth_blurred = gaussian_blur(mouth_img, sigma)
            combined = np.concatenate([face_blurred, mouth_blurred], axis=-1)
            augmented.append(torch.from_numpy(combined).permute(2, 0, 1).float() / 255.0)

        frames = torch.stack(augmented, dim=0)

    elif aug_type == "frame_drop":
        # Temporal frame dropping
        drop_prob = cfg.get("frame_drop_prob", 0.2)
        frames = temporal_frame_drop(frames, drop_prob)

    elif aug_type == "audio_noise":
        # Audio Gaussian noise
        noise_std = cfg.get("audio_noise_std", 0.01)
        mel = audio_gaussian_noise(mel, noise_std)

    return frames, mel


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 80)
    print("AUGMENTATION - UNIT TESTS")
    print("=" * 80)

    # Test 1: JPEG compression
    print("\n[Test 1] JPEG compression...")
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    compressed = jpeg_compress(img, quality=50)
    assert compressed.shape == (224, 224, 3)
    print("    ✅ PASSED")

    # Test 2: Gaussian blur
    print("\n[Test 2] Gaussian blur...")
    blurred = gaussian_blur(img, sigma=1.5)
    assert blurred.shape == (224, 224, 3)
    print("    ✅ PASSED")

    # Test 3: Temporal frame drop
    print("\n[Test 3] Temporal frame drop...")
    dummy_frames = torch.randn(16, 6, 224, 224)
    dropped = temporal_frame_drop(dummy_frames, drop_prob=0.3)
    assert dropped.shape == (16, 6, 224, 224)
    print("    ✅ PASSED")

    # Test 4: Audio noise
    print("\n[Test 4] Audio Gaussian noise...")
    dummy_mel = torch.randn(16, 80, 32)
    noisy = audio_gaussian_noise(dummy_mel, noise_std=0.01)
    assert noisy.shape == (16, 80, 32)
    assert not torch.allclose(dummy_mel, noisy), "Noise should change the tensor"
    print("    ✅ PASSED")

    # Test 5: Master augmentation function (training)
    print("\n[Test 5] Master augmentation (training)...")
    random.seed(42)
    frames = torch.rand(16, 6, 224, 224)
    mel = torch.rand(16, 80, 32)
    aug_frames, aug_mel = apply_augmentation(frames, mel, is_training=True)
    assert aug_frames.shape == (16, 6, 224, 224)
    assert aug_mel.shape == (16, 80, 32)
    print("    ✅ PASSED")

    # Test 6: No augmentation on validation
    print("\n[Test 6] No augmentation on validation...")
    val_frames, val_mel = apply_augmentation(frames, mel, is_training=False)
    assert torch.equal(frames, val_frames), "Val frames should be unchanged"
    assert torch.equal(mel, val_mel), "Val mel should be unchanged"
    print("    ✅ PASSED")

    print("\n" + "=" * 80)
    print("✅ ALL AUGMENTATION TESTS PASSED")
    print("=" * 80)
