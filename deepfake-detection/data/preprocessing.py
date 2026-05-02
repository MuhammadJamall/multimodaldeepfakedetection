"""
preprocessing.py
----------------
Complete preprocessing pipeline for multimodal deepfake detection.

Per CONTEXT.md §4.2:
  1. Extract 16 uniformly sampled frames from each video
  2. Apply MTCNN for face detection + bounding box extraction
  3. Generate full-face (224×224) and mouth (upscaled 96×96 → 224×224) crops
  4. Stack channel-wise → 6-channel tensor per frame
  5. Extract audio segment aligned to video duration
  6. Compute 80-band Mel Spectrogram windowed to match T=16 visual frames

Fallback logic (CONTEXT.md §2.1):
  - If MTCNN fails on a frame → copy bbox from last valid frame
  - If first frame fails → use static center crop (70% of frame)
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import librosa
from PIL import Image

try:
    import torchaudio
except (ImportError, OSError):
    # OSError happens on Windows when torchaudio binary is installed but
    # incompatible with the local torch/CPU runtime.
    torchaudio = None

try:
    from facenet_pytorch import MTCNN
except ImportError:
    MTCNN = None


# ── Configuration defaults ────────────────────────────────────────────────────

DEFAULT_NUM_FRAMES = 16
DEFAULT_FACE_SIZE = 224
DEFAULT_MOUTH_SIZE = 96
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_MEL_BINS = 80
DEFAULT_HOP_MS = 10
DEFAULT_WIN_MS = 25


# ── Video Frame Extraction ───────────────────────────────────────────────────

def extract_frames(
    video_path: str,
    num_frames: int = DEFAULT_NUM_FRAMES,
) -> Tuple[List[np.ndarray], float, int]:
    """
    Extract num_frames uniformly sampled frames from a video.

    Args:
        video_path: Path to the video file.
        num_frames: Number of frames to extract (default 16).

    Returns:
        frames: List of BGR numpy arrays (H, W, 3).
        fps: Video frame rate.
        total_frames: Total number of frames in video.

    Raises:
        RuntimeError: If video cannot be opened.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if total_frames <= 0:
        raise RuntimeError(f"Video has 0 frames: {video_path}")

    # Uniformly sample frame indices
    if total_frames >= num_frames:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:
        # If video has fewer frames, repeat last frame
        indices = list(range(total_frames))
        indices += [total_frames - 1] * (num_frames - total_frames)
        indices = np.array(indices, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        elif len(frames) > 0:
            # Fallback: duplicate last valid frame
            frames.append(frames[-1].copy())
        else:
            # Create black frame as last resort
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 224
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 224
            frames.append(np.zeros((h, w, 3), dtype=np.uint8))

    cap.release()
    return frames[:num_frames], fps, total_frames


# ── Face Detection with MTCNN + Fallback ─────────────────────────────────────

def _center_crop_bbox(h: int, w: int, crop_ratio: float = 0.7) -> Tuple[int, int, int, int]:
    """Static center crop fallback — returns (x1, y1, x2, y2)."""
    crop_w = int(w * crop_ratio)
    crop_h = int(h * crop_ratio)
    x1 = (w - crop_w) // 2
    y1 = (h - crop_h) // 2
    return x1, y1, x1 + crop_w, y1 + crop_h


def build_face_detector(device: Optional[str] = None) -> Optional[object]:
    """Build MTCNN face detector if available."""
    if MTCNN is None:
        return None
    return MTCNN(
        image_size=DEFAULT_FACE_SIZE,
        margin=20,
        keep_all=False,
        post_process=False,
        device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
    )


def detect_faces_with_fallback(
    frames: List[np.ndarray],
    detector=None,
) -> List[Tuple[int, int, int, int]]:
    """
    Detect face bounding boxes with MTCNN fallback logic per CONTEXT.md §2.1.

    Fallback rules:
      - If MTCNN fails on a frame → copy bbox from last valid frame
      - If first frame fails → use static center crop (70% of frame)

    Args:
        frames: List of BGR numpy arrays.
        detector: MTCNN detector instance (or None to skip detection).

    Returns:
        List of (x1, y1, x2, y2) bounding boxes, one per frame.
    """
    h, w = frames[0].shape[:2]
    center_bbox = _center_crop_bbox(h, w)
    bboxes: List[Tuple[int, int, int, int]] = []
    last_valid_bbox = center_bbox  # Default fallback

    for i, frame in enumerate(frames):
        bbox = None

        if detector is not None:
            try:
                # MTCNN expects RGB PIL Image
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(rgb_frame)
                boxes, _ = detector.detect(pil_frame)

                if boxes is not None and len(boxes) > 0:
                    # Take the first (most confident) face
                    b = boxes[0]
                    bbox = (
                        max(0, int(b[0])),
                        max(0, int(b[1])),
                        min(w, int(b[2])),
                        min(h, int(b[3])),
                    )
            except Exception:
                bbox = None

        if bbox is not None:
            last_valid_bbox = bbox
            bboxes.append(bbox)
        else:
            # Fallback: use last valid bbox (or center crop for first frame)
            bboxes.append(last_valid_bbox)

    return bboxes


# ── Face and Mouth Cropping ──────────────────────────────────────────────────

def crop_full_face(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    size: int = DEFAULT_FACE_SIZE,
) -> np.ndarray:
    """
    Crop full face region and resize to size×size.

    Args:
        frame: BGR numpy array (H, W, 3).
        bbox: (x1, y1, x2, y2) face bounding box.
        size: Output size (default 224).

    Returns:
        Resized face crop as numpy array (size, size, 3).
    """
    x1, y1, x2, y2 = bbox
    face = frame[y1:y2, x1:x2]
    if face.size == 0:
        face = frame  # Fallback to full frame
    return cv2.resize(face, (size, size), interpolation=cv2.INTER_LINEAR)


def crop_mouth_region(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    mouth_size: int = DEFAULT_MOUTH_SIZE,
    output_size: int = DEFAULT_FACE_SIZE,
) -> np.ndarray:
    """
    Crop mouth region (lower 40% of face bbox), resize to mouth_size,
    then upscale to output_size per CONTEXT.md §2.1.

    Args:
        frame: BGR numpy array (H, W, 3).
        bbox: (x1, y1, x2, y2) face bounding box.
        mouth_size: Intermediate size for mouth crop (default 96).
        output_size: Final output size (default 224).

    Returns:
        Upscaled mouth crop as numpy array (output_size, output_size, 3).
    """
    x1, y1, x2, y2 = bbox
    face_h = y2 - y1

    # Mouth region: lower 40% of face bounding box
    mouth_y1 = y1 + int(face_h * 0.6)
    mouth = frame[mouth_y1:y2, x1:x2]

    if mouth.size == 0:
        mouth = frame[y1:y2, x1:x2]  # Fallback to full face
    if mouth.size == 0:
        mouth = frame  # Fallback to full frame

    # First resize to mouth_size (96×96), then upscale to output_size (224×224)
    mouth_small = cv2.resize(mouth, (mouth_size, mouth_size), interpolation=cv2.INTER_LINEAR)
    mouth_up = cv2.resize(mouth_small, (output_size, output_size), interpolation=cv2.INTER_LINEAR)
    return mouth_up


def create_6channel_tensor(
    full_face: np.ndarray,
    mouth: np.ndarray,
) -> torch.Tensor:
    """
    Stack full-face and mouth crops channel-wise → 6-channel tensor.

    Args:
        full_face: (224, 224, 3) BGR numpy array.
        mouth:     (224, 224, 3) BGR numpy array.

    Returns:
        (6, 224, 224) float32 tensor, values in [0, 1].
    """
    # Convert BGR → RGB and normalize to [0, 1] without torchvision.
    face_rgb = cv2.cvtColor(full_face, cv2.COLOR_BGR2RGB)
    mouth_rgb = cv2.cvtColor(mouth, cv2.COLOR_BGR2RGB)

    face_t = torch.from_numpy(face_rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
    mouth_t = torch.from_numpy(mouth_rgb.astype(np.float32) / 255.0).permute(2, 0, 1)

    return torch.cat([face_t, mouth_t], dim=0)  # (6, 224, 224)


# ── Audio Extraction ─────────────────────────────────────────────────────────

def extract_audio(
    video_path: str,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> torch.Tensor:
    """
    Extract audio from video file, resampled to target sample rate.

    Critical per CONTEXT.md §2.2: Audio segment must perfectly match video
    duration (frame-aligned). No sliding window.

    Args:
        video_path: Path to video file.
        sample_rate: Target sample rate (default 16000 Hz).

    Returns:
        Waveform tensor of shape (1, num_samples).
    """
    if torchaudio is not None:
        waveform, orig_sr = torchaudio.load(video_path)

        # Convert stereo to mono if needed
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if necessary
        if orig_sr != sample_rate:
            resampler = torchaudio.transforms.Resample(orig_sr, sample_rate)
            waveform = resampler(waveform)

        return waveform  # (1, num_samples)

    # Fallback when torchaudio is unavailable: use librosa + torch tensor.
    audio, orig_sr = librosa.load(video_path, sr=sample_rate, mono=True)
    if orig_sr != sample_rate and audio.size > 0:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sample_rate)
    return torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0)


# ── Mel Spectrogram Windowed to Match Visual Frames ──────────────────────────

def compute_mel_windows(
    waveform: torch.Tensor,
    num_frames: int = DEFAULT_NUM_FRAMES,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    n_mels: int = DEFAULT_MEL_BINS,
    hop_ms: float = DEFAULT_HOP_MS,
    win_ms: float = DEFAULT_WIN_MS,
) -> torch.Tensor:
    """
    Compute Mel spectrogram and split into T windows matching visual frames.

    Per CONTEXT.md §2.2: Audio must be frame-aligned with video.
    We compute the full Mel spectrogram, then split into T equal windows.

    Args:
        waveform: (1, num_samples) audio waveform.
        num_frames: Number of visual frames to match (default 16).
        sample_rate: Audio sample rate in Hz (default 16000).
        n_mels: Number of Mel frequency bins (default 80).
        hop_ms: Hop length in milliseconds (default 10).
        win_ms: Window size in milliseconds (default 25).

    Returns:
        (T, n_mels, F) tensor where F = time frames per window.
    """
    hop_length = int(sample_rate * hop_ms / 1000)    # 160 samples
    win_length = int(sample_rate * win_ms / 1000)    # 400 samples

    if torchaudio is not None:
        # Compute full Mel spectrogram with torchaudio when available.
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            hop_length=hop_length,
            win_length=win_length,
            n_fft=max(win_length, 512),
        )
        mel_spec = mel_transform(waveform)  # (1, n_mels, total_time_frames)

        # Convert to log scale
        log_mel = torchaudio.transforms.AmplitudeToDB()(mel_spec)
        log_mel = log_mel.squeeze(0)  # (n_mels, total_time_frames)
    else:
        # librosa fallback for environments without torchaudio.
        waveform_np = waveform.detach().cpu().numpy().squeeze(0)
        mel_spec = librosa.feature.melspectrogram(
            y=waveform_np,
            sr=sample_rate,
            n_mels=n_mels,
            hop_length=hop_length,
            win_length=win_length,
            n_fft=max(win_length, 512),
            power=2.0,
        )
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        log_mel = torch.from_numpy(log_mel.astype(np.float32))

    total_time_frames = log_mel.shape[1]

    if total_time_frames < num_frames:
        # Pad if too short
        pad_amount = num_frames - total_time_frames
        log_mel = torch.nn.functional.pad(log_mel, (0, pad_amount), mode='constant', value=0)
        total_time_frames = log_mel.shape[1]

    # Split into T equal windows
    frames_per_window = total_time_frames // num_frames
    if frames_per_window == 0:
        frames_per_window = 1

    windows = []
    for i in range(num_frames):
        start = i * frames_per_window
        end = start + frames_per_window
        if end > total_time_frames:
            end = total_time_frames
        window = log_mel[:, start:end]  # (n_mels, F)
        windows.append(window)

    # Ensure all windows have the same F dimension (pad/truncate)
    target_f = frames_per_window
    aligned_windows = []
    for w in windows:
        if w.shape[1] < target_f:
            w = torch.nn.functional.pad(w, (0, target_f - w.shape[1]))
        elif w.shape[1] > target_f:
            w = w[:, :target_f]
        aligned_windows.append(w)

    return torch.stack(aligned_windows, dim=0)  # (T, n_mels, F)


# ── Full Video Processing Pipeline ──────────────────────────────────────────

def process_single_video(
    video_path: str,
    num_frames: int = DEFAULT_NUM_FRAMES,
    face_size: int = DEFAULT_FACE_SIZE,
    mouth_size: int = DEFAULT_MOUTH_SIZE,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    n_mels: int = DEFAULT_MEL_BINS,
    hop_ms: float = DEFAULT_HOP_MS,
    win_ms: float = DEFAULT_WIN_MS,
    detector=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Process a single video file into model-ready tensors.

    Full pipeline per CONTEXT.md §4.2:
      1. Extract 16 uniformly sampled frames
      2. MTCNN face detection with fallback
      3. Full-face + mouth crops → 6-channel stacking
      4. Audio extraction (aligned to video duration)
      5. Mel spectrogram windowed to T frames

    Args:
        video_path: Path to video file.
        num_frames: Number of frames to sample (default 16).
        face_size: Face crop size (default 224).
        mouth_size: Intermediate mouth size (default 96).
        sample_rate: Audio sample rate (default 16000).
        n_mels: Mel frequency bins (default 80).
        hop_ms: Mel hop length ms (default 10).
        win_ms: Mel window size ms (default 25).
        detector: Pre-built MTCNN detector (or None to build one).

    Returns:
        frames_tensor: (T, 6, 224, 224) float32
        mel_tensor:    (T, n_mels, F) float32
    """
    # ── Visual processing ─────────────────────────────────────────────────
    frames, fps, total = extract_frames(video_path, num_frames)
    bboxes = detect_faces_with_fallback(frames, detector)

    visual_tensors = []
    for frame, bbox in zip(frames, bboxes):
        full_face = crop_full_face(frame, bbox, face_size)
        mouth = crop_mouth_region(frame, bbox, mouth_size, face_size)
        six_ch = create_6channel_tensor(full_face, mouth)  # (6, 224, 224)
        visual_tensors.append(six_ch)

    frames_tensor = torch.stack(visual_tensors, dim=0)  # (T, 6, 224, 224)

    # ── Audio processing ──────────────────────────────────────────────────
    try:
        waveform = extract_audio(video_path, sample_rate)
        mel_tensor = compute_mel_windows(
            waveform, num_frames, sample_rate, n_mels, hop_ms, win_ms
        )
    except Exception:
        # If audio extraction fails, create silence (zeros)
        hop_length = int(sample_rate * hop_ms / 1000)
        dummy_f = 32  # Reasonable default
        mel_tensor = torch.zeros(num_frames, n_mels, dummy_f)

    return frames_tensor, mel_tensor


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 80)
    print("PREPROCESSING PIPELINE - UNIT TESTS")
    print("=" * 80)

    # Test 1: Center crop fallback
    print("\n[Test 1] Center crop fallback...")
    bbox = _center_crop_bbox(480, 640, 0.7)
    assert len(bbox) == 4, "Bbox should be 4-tuple"
    print(f"    Center crop for 640×480: {bbox}")
    print("    ✅ PASSED")

    # Test 2: Face cropping with synthetic frame
    print("\n[Test 2] Face cropping...")
    fake_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    face = crop_full_face(fake_frame, (100, 50, 400, 400), 224)
    assert face.shape == (224, 224, 3), f"Wrong face shape: {face.shape}"
    print(f"    Face crop shape: {face.shape}")
    print("    ✅ PASSED")

    # Test 3: Mouth cropping
    print("\n[Test 3] Mouth cropping...")
    mouth = crop_mouth_region(fake_frame, (100, 50, 400, 400), 96, 224)
    assert mouth.shape == (224, 224, 3), f"Wrong mouth shape: {mouth.shape}"
    print(f"    Mouth crop shape: {mouth.shape}")
    print("    ✅ PASSED")

    # Test 4: 6-channel stacking
    print("\n[Test 4] 6-channel tensor...")
    tensor_6ch = create_6channel_tensor(face, mouth)
    assert tensor_6ch.shape == (6, 224, 224), f"Wrong shape: {tensor_6ch.shape}"
    assert tensor_6ch.min() >= 0 and tensor_6ch.max() <= 1, "Values out of [0,1]"
    print(f"    6-channel tensor shape: {tensor_6ch.shape}")
    print("    ✅ PASSED")

    # Test 5: Mel windowing with synthetic audio
    print("\n[Test 5] Mel spectrogram windowing...")
    dummy_waveform = torch.randn(1, 16000 * 3)  # 3 seconds of audio
    mel_windows = compute_mel_windows(dummy_waveform, num_frames=16)
    assert mel_windows.shape[0] == 16, f"Expected 16 windows, got {mel_windows.shape[0]}"
    assert mel_windows.shape[1] == 80, f"Expected 80 mel bins, got {mel_windows.shape[1]}"
    print(f"    Mel windows shape: {mel_windows.shape}  (T=16, mel_bins=80, F={mel_windows.shape[2]})")
    print("    ✅ PASSED")

    # Test 6: MTCNN fallback
    print("\n[Test 6] Face detection fallback (no MTCNN)...")
    fake_frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(16)]
    bboxes = detect_faces_with_fallback(fake_frames, detector=None)
    assert len(bboxes) == 16, f"Expected 16 bboxes, got {len(bboxes)}"
    print(f"    Fallback bboxes: {len(bboxes)} (all center crop)")
    print("    ✅ PASSED")

    print("\n" + "=" * 80)
    print("✅ ALL PREPROCESSING TESTS PASSED")
    print("=" * 80)
