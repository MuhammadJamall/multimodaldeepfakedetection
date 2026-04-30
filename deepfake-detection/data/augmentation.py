from __future__ import annotations

import subprocess

import cv2
import numpy as np


def jpeg_compress(image: np.ndarray, quality: int = 75) -> np.ndarray:
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    success, buffer = cv2.imencode(".jpg", image, encode_params)
    if not success:
        return image
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def add_gaussian_noise(image: np.ndarray, sigma: float = 5.0) -> np.ndarray:
    noise = np.random.normal(0.0, sigma, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def h264_compress(input_path: str, output_path: str, crf: int = 23) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-preset",
        "fast",
        output_path,
    ]
    subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
