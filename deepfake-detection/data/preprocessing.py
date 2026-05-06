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

Improvements over v1:
  - MTCNN batch detection (all frames in one call, ~16x faster)
  - Sequential frame reading instead of random cap.set() seeks
  - Mel windowing via torch.reshape instead of Python loop
  - Vectorized BGR→RGB + normalization across all frames at once
  - torch.inference_mode() on all non-training compute paths
  - Computed audio fallback shape (no magic numbers)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import librosa
from PIL import Image

try:
    import torchaudio
except (ImportError, OSError):
    torchaudio = None

try:
    from facenet_pytorch import MTCNN
except ImportError:
    MTCNN = None


# ── Configuration defaults ────────────────────────────────────────────────────

DEFAULT_NUM_FRAMES  = 16
DEFAULT_FACE_SIZE   = 224
DEFAULT_MOUTH_SIZE  = 96
DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_MEL_BINS    = 80
DEFAULT_HOP_MS      = 10
DEFAULT_WIN_MS      = 25


# ── Video Frame Extraction ────────────────────────────────────────────────────

def extract_frames(
    video_path: str,
    num_frames: int = DEFAULT_NUM_FRAMES,
) -> Tuple[List[np.ndarray], float, int]:
    """
    Extract num_frames uniformly sampled frames from a video.

    IMPROVEMENT over v1: reads frames sequentially instead of using
    cap.set(CAP_PROP_POS_FRAMES) for every index.  Random seeks are
    expensive on compressed video because the decoder must scan back to
    the nearest keyframe each time.  Sequential reading then picking the
    right frames is significantly faster.

    Args:
        video_path: Path to the video file.
        num_frames: Number of frames to extract (default 16).

    Returns:
        frames     : List of BGR numpy arrays (H, W, 3).
        fps        : Video frame rate.
        total_frames: Total number of frames in video.

    Raises:
        RuntimeError: If video cannot be opened or has 0 frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0

    if total_frames <= 0:
        raise RuntimeError(f"Video has 0 frames: {video_path}")

    # Build the set of target indices once
    if total_frames >= num_frames:
        target_indices = set(np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist())
    else:
        target_indices = set(range(total_frames))

    # ── Single sequential pass through the video ──────────────────────────
    collected: dict[int, np.ndarray] = {}
    frame_idx = 0
    while frame_idx <= max(target_indices):
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in target_indices:
            collected[frame_idx] = frame
        frame_idx += 1
    cap.release()

    # Rebuild ordered list, applying fallbacks
    sorted_indices = sorted(target_indices)
    frames: List[np.ndarray] = []
    last_valid: Optional[np.ndarray] = None

    for idx in sorted_indices:
        if idx in collected:
            last_valid = collected[idx]
            frames.append(collected[idx])
        elif last_valid is not None:
            frames.append(last_valid.copy())
        else:
            # Black frame as absolute last resort
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 224
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 224
            frames.append(np.zeros((h, w, 3), dtype=np.uint8))

    # Handle videos shorter than num_frames
    while len(frames) < num_frames:
        frames.append(frames[-1].copy() if frames else np.zeros((224, 224, 3), dtype=np.uint8))

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


@torch.inference_mode()
def detect_faces_with_fallback(
    frames: List[np.ndarray],
    detector=None,
) -> List[Tuple[int, int, int, int]]:
    """
    Detect face bounding boxes with MTCNN fallback logic per CONTEXT.md §2.1.

    IMPROVEMENT over v1: converts all frames to PIL up-front and calls
    detector.detect() once with the full batch instead of calling it
    16 times in a loop.  This gives ~16x throughput on GPU and still
    meaningfully faster on CPU.

    Fallback rules:
      - If MTCNN fails on a frame → copy bbox from last valid frame
      - If first frame fails     → use static center crop (70%)

    Args:
        frames   : List of BGR numpy arrays.
        detector : MTCNN detector instance (or None to skip detection).

    Returns:
        List of (x1, y1, x2, y2) bounding boxes, one per frame.
    """
    h, w        = frames[0].shape[:2]
    center_bbox = _center_crop_bbox(h, w)

    raw_boxes: List[Optional[Tuple[int, int, int, int]]] = [None] * len(frames)

    if detector is not None:
        try:
            # Convert all frames to PIL RGB in one vectorised numpy op
            # stack → (N, H, W, 3) then flip channels once for the whole batch
            bgr_stack  = np.stack(frames, axis=0)                          # (N,H,W,3)
            rgb_stack  = bgr_stack[..., ::-1].copy()                       # BGR→RGB
            pil_frames = [Image.fromarray(rgb_stack[i]) for i in range(len(frames))]

            # ── Single batched MTCNN call ─────────────────────────────────
            batch_boxes, _ = detector.detect(pil_frames)
            # batch_boxes is a list of arrays (one per image), or None entries

            for i, boxes in enumerate(batch_boxes):
                if boxes is not None and len(boxes) > 0:
                    b = boxes[0]
                    raw_boxes[i] = (
                        max(0, int(b[0])),
                        max(0, int(b[1])),
                        min(w, int(b[2])),
                        min(h, int(b[3])),
                    )
        except Exception:
            pass  # All entries stay None → centre-crop fallback below

    # Apply fallback logic
    bboxes: List[Tuple[int, int, int, int]] = []
    last_valid = center_bbox

    for bbox in raw_boxes:
        if bbox is not None:
            last_valid = bbox
            bboxes.append(bbox)
        else:
            bboxes.append(last_valid)

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
        bbox : (x1, y1, x2, y2) face bounding box.
        size : Output size (default 224).

    Returns:
        Resized face crop as numpy array (size, size, 3).
    """
    x1, y1, x2, y2 = bbox
    face = frame[y1:y2, x1:x2]
    if face.size == 0:
        face = frame
    return cv2.resize(face, (size, size), interpolation=cv2.INTER_LINEAR)


def crop_mouth_region(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    mouth_size:  int = DEFAULT_MOUTH_SIZE,
    output_size: int = DEFAULT_FACE_SIZE,
) -> np.ndarray:
    """
    Crop mouth region (lower 40% of face bbox), resize to mouth_size,
    then upscale to output_size per CONTEXT.md §2.1.

    Args:
        frame      : BGR numpy array (H, W, 3).
        bbox       : (x1, y1, x2, y2) face bounding box.
        mouth_size : Intermediate crop size (default 96).
        output_size: Final output size (default 224).

    Returns:
        Upscaled mouth crop as numpy array (output_size, output_size, 3).
    """
    x1, y1, x2, y2 = bbox
    mouth_y1 = y1 + int((y2 - y1) * 0.6)
    mouth    = frame[mouth_y1:y2, x1:x2]

    if mouth.size == 0:
        mouth = frame[y1:y2, x1:x2]
    if mouth.size == 0:
        mouth = frame

    mouth_small = cv2.resize(mouth, (mouth_size, mouth_size),   interpolation=cv2.INTER_LINEAR)
    mouth_up    = cv2.resize(mouth_small, (output_size, output_size), interpolation=cv2.INTER_LINEAR)
    return mouth_up


def build_visual_tensor(
    frames: List[np.ndarray],
    bboxes: List[Tuple[int, int, int, int]],
    face_size:  int = DEFAULT_FACE_SIZE,
    mouth_size: int = DEFAULT_MOUTH_SIZE,
) -> torch.Tensor:
    """
    Build the full (T, 6, 224, 224) visual tensor from frames + bboxes.

    IMPROVEMENT over v1: replaces the per-frame Python loop in
    process_single_video with a single vectorised numpy operation.
    All face and mouth crops are stacked into a batch array, the
    BGR→RGB flip and /255 normalisation are applied to the whole
    batch at once, and a single torch.from_numpy call produces the
    final tensor.  No Python-level loop over pixels.

    Args:
        frames    : List of T BGR numpy arrays (H, W, 3).
        bboxes    : List of T bounding boxes (x1, y1, x2, y2).
        face_size : Crop output size (default 224).
        mouth_size: Intermediate mouth size (default 96).

    Returns:
        (T, 6, 224, 224) float32 tensor with values in [0, 1].
    """
    T = len(frames)
    face_crops  = np.empty((T, face_size, face_size, 3), dtype=np.uint8)
    mouth_crops = np.empty((T, face_size, face_size, 3), dtype=np.uint8)

    for i, (frame, bbox) in enumerate(zip(frames, bboxes)):
        face_crops[i]  = crop_full_face(frame, bbox, face_size)
        mouth_crops[i] = crop_mouth_region(frame, bbox, mouth_size, face_size)

    # Vectorised BGR→RGB + normalise + channel-first — all at once
    # face_crops/mouth_crops: (T, H, W, 3) uint8
    face_rgb  = face_crops[...,  ::-1].copy().astype(np.float32) / 255.0   # (T,H,W,3)
    mouth_rgb = mouth_crops[..., ::-1].copy().astype(np.float32) / 255.0   # (T,H,W,3)

    # (T,H,W,3) → (T,3,H,W)
    face_t  = torch.from_numpy(face_rgb).permute(0, 3, 1, 2)
    mouth_t = torch.from_numpy(mouth_rgb).permute(0, 3, 1, 2)

    return torch.cat([face_t, mouth_t], dim=1)   # (T, 6, 224, 224)


# kept for backwards compatibility / unit tests
def create_6channel_tensor(
    full_face: np.ndarray,
    mouth: np.ndarray,
) -> torch.Tensor:
    """Single-frame 6-channel stacking (kept for unit tests)."""
    face_rgb  = cv2.cvtColor(full_face, cv2.COLOR_BGR2RGB)
    mouth_rgb = cv2.cvtColor(mouth,     cv2.COLOR_BGR2RGB)
    face_t  = torch.from_numpy(face_rgb.astype(np.float32)  / 255.0).permute(2, 0, 1)
    mouth_t = torch.from_numpy(mouth_rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
    return torch.cat([face_t, mouth_t], dim=0)   # (6, 224, 224)


# ── Audio Extraction ──────────────────────────────────────────────────────────

def extract_audio(
    video_path:  str,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> torch.Tensor:
    """
    Extract audio from video file, resampled to target sample rate.

    Args:
        video_path : Path to video file.
        sample_rate: Target sample rate (default 16 000 Hz).

    Returns:
        Waveform tensor of shape (1, num_samples).
    """
    if torchaudio is not None:
        waveform, orig_sr = torchaudio.load(video_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if orig_sr != sample_rate:
            waveform = torchaudio.transforms.Resample(orig_sr, sample_rate)(waveform)
        return waveform

    audio, _ = librosa.load(video_path, sr=sample_rate, mono=True)
    return torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0)


# ── Mel Spectrogram Windowed to Match Visual Frames ──────────────────────────

@torch.inference_mode()
def compute_mel_windows(
    waveform:    torch.Tensor,
    num_frames:  int   = DEFAULT_NUM_FRAMES,
    sample_rate: int   = DEFAULT_SAMPLE_RATE,
    n_mels:      int   = DEFAULT_MEL_BINS,
    hop_ms:      float = DEFAULT_HOP_MS,
    win_ms:      float = DEFAULT_WIN_MS,
) -> torch.Tensor:
    """
    Compute Mel spectrogram and split into T windows matching visual frames.

    IMPROVEMENT over v1: replaces the Python for-loop over windows with a
    single torch reshape.  After padding to a length divisible by T the
    spectrogram is reshaped directly into (T, n_mels, F) — zero Python
    iterations, no per-window pad/truncate checks.

    Args:
        waveform   : (1, num_samples) audio waveform.
        num_frames : Number of visual frames to align to (default 16).
        sample_rate: Audio sample rate in Hz (default 16 000).
        n_mels     : Number of Mel frequency bins (default 80).
        hop_ms     : Hop length in milliseconds (default 10).
        win_ms     : Window size in milliseconds (default 25).

    Returns:
        (T, n_mels, F) float32 tensor.
    """
    hop_length = int(sample_rate * hop_ms / 1000)   # 160 samples
    win_length = int(sample_rate * win_ms / 1000)   # 400 samples
    n_fft      = max(win_length, 512)

    if torchaudio is not None:
        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            hop_length=hop_length,
            win_length=win_length,
            n_fft=n_fft,
        )(waveform)                                             # (1, n_mels, total_T)
        log_mel = torchaudio.transforms.AmplitudeToDB()(mel_spec).squeeze(0)  # (n_mels, total_T)
    else:
        waveform_np = waveform.detach().cpu().numpy().squeeze(0)
        mel_spec = librosa.feature.melspectrogram(
            y=waveform_np, sr=sample_rate, n_mels=n_mels,
            hop_length=hop_length, win_length=win_length,
            n_fft=n_fft, power=2.0,
        )
        log_mel = torch.from_numpy(
            librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)
        )                                                       # (n_mels, total_T)

    total_T = log_mel.shape[1]

    # ── Pad so total_T is exactly divisible by num_frames ────────────────
    remainder = total_T % num_frames
    if remainder != 0:
        pad_amount = num_frames - remainder
        log_mel    = F.pad(log_mel, (0, pad_amount))
        total_T    = log_mel.shape[1]

    frames_per_window = total_T // num_frames

    # ── Single reshape — no Python loop ──────────────────────────────────
    # (n_mels, total_T) → (n_mels, T, F) → (T, n_mels, F)
    return log_mel.reshape(n_mels, num_frames, frames_per_window).permute(1, 0, 2).contiguous()


# ── Full Video Processing Pipeline ───────────────────────────────────────────

def process_single_video(
    video_path:  str,
    num_frames:  int   = DEFAULT_NUM_FRAMES,
    face_size:   int   = DEFAULT_FACE_SIZE,
    mouth_size:  int   = DEFAULT_MOUTH_SIZE,
    sample_rate: int   = DEFAULT_SAMPLE_RATE,
    n_mels:      int   = DEFAULT_MEL_BINS,
    hop_ms:      float = DEFAULT_HOP_MS,
    win_ms:      float = DEFAULT_WIN_MS,
    detector=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Process a single video file into model-ready tensors.

    Full pipeline per CONTEXT.md §4.2:
      1. Extract 16 uniformly sampled frames (sequential read)
      2. Batch MTCNN face detection with fallback
      3. Vectorised full-face + mouth crops → 6-channel stacking
      4. Audio extraction (aligned to video duration)
      5. Mel spectrogram windowed to T frames via reshape

    Args:
        video_path : Path to video file.
        num_frames : Number of frames to sample (default 16).
        face_size  : Face crop size (default 224).
        mouth_size : Intermediate mouth size (default 96).
        sample_rate: Audio sample rate (default 16 000).
        n_mels     : Mel frequency bins (default 80).
        hop_ms     : Mel hop length ms (default 10).
        win_ms     : Mel window size ms (default 25).
        detector   : Pre-built MTCNN detector (or None to build one).

    Returns:
        frames_tensor: (T, 6, 224, 224) float32
        mel_tensor   : (T, n_mels, F)   float32
    """
    # ── Visual processing ─────────────────────────────────────────────────
    frames, fps, total = extract_frames(video_path, num_frames)
    bboxes = detect_faces_with_fallback(frames, detector)
    frames_tensor = build_visual_tensor(frames, bboxes, face_size, mouth_size)

    # ── Audio processing ──────────────────────────────────────────────────
    try:
        waveform   = extract_audio(video_path, sample_rate)
        mel_tensor = compute_mel_windows(waveform, num_frames, sample_rate, n_mels, hop_ms, win_ms)
    except Exception:
        # Compute the correct F dimension instead of using a magic number
        hop_length        = int(sample_rate * hop_ms / 1000)
        total_audio_frames = int(np.ceil(sample_rate / hop_length))  # ~1 sec estimate
        frames_per_window = max(1, total_audio_frames // num_frames)
        mel_tensor = torch.zeros(num_frames, n_mels, frames_per_window)

    return frames_tensor, mel_tensor


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 80)
    print("PREPROCESSING PIPELINE - UNIT TESTS (v2)")
    print("=" * 80)

    # Test 1: Center crop fallback
    print("\n[Test 1] Center crop fallback...")
    bbox = _center_crop_bbox(480, 640, 0.7)
    assert len(bbox) == 4
    print(f"    Center crop for 640×480: {bbox}  ✅")

    # Test 2: Face crop
    print("\n[Test 2] Face cropping...")
    fake_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    face = crop_full_face(fake_frame, (100, 50, 400, 400), 224)
    assert face.shape == (224, 224, 3)
    print(f"    Face crop shape: {face.shape}  ✅")

    # Test 3: Mouth crop
    print("\n[Test 3] Mouth cropping...")
    mouth = crop_mouth_region(fake_frame, (100, 50, 400, 400), 96, 224)
    assert mouth.shape == (224, 224, 3)
    print(f"    Mouth crop shape: {mouth.shape}  ✅")

    # Test 4: Vectorised 6-channel build
    print("\n[Test 4] Vectorised build_visual_tensor...")
    fake_frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(16)]
    fake_bboxes = [_center_crop_bbox(480, 640)] * 16
    vt = build_visual_tensor(fake_frames, fake_bboxes)
    assert vt.shape == (16, 6, 224, 224), f"Wrong shape: {vt.shape}"
    assert vt.min() >= 0.0 and vt.max() <= 1.0
    print(f"    Visual tensor: {vt.shape}  ✅")

    # Test 5: Mel windowing via reshape
    print("\n[Test 5] Mel spectrogram windowing (reshape)...")
    dummy_waveform = torch.randn(1, 16000 * 3)
    mel = compute_mel_windows(dummy_waveform, num_frames=16)
    assert mel.shape[0] == 16
    assert mel.shape[1] == 80
    print(f"    Mel windows: {mel.shape}  (T=16, mel_bins=80, F={mel.shape[2]})  ✅")

    # Test 6: Batch face detection fallback (no MTCNN)
    print("\n[Test 6] Batch face detection fallback...")
    bboxes = detect_faces_with_fallback(fake_frames, detector=None)
    assert len(bboxes) == 16
    print(f"    Bboxes returned: {len(bboxes)}  ✅")

    # Test 7: Single-frame helper still works
    print("\n[Test 7] create_6channel_tensor (backwards compat)...")
    t = create_6channel_tensor(face, mouth)
    assert t.shape == (6, 224, 224)
    print(f"    Shape: {t.shape}  ✅")

    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED  (v2)")
    print("=" * 80)