#!/usr/bin/env python3
"""
preprocess_to_hdf5.py
---------------------
Offline preprocessing script for FakeAVCeleb / DFDC datasets.

Processes raw video files into HDF5 format for efficient training.
Per CONTEXT.md §4.2: "Preprocess offline into HDF5 files.
On-the-fly MTCNN and Mel extraction will severely bottleneck GPU training."

HDF5 structure:
    /{split}/frames  → (N, 16, 6, 224, 224) float32
    /{split}/mel     → (N, 16, 80, F) float32
    /{split}/labels  → (N,) int64
    /{split}/methods → variable-length strings (forgery method)
    /{split}/paths   → variable-length strings (original video paths)

Usage:
    python scripts/preprocess_to_hdf5.py \\
        --data-dir /path/to/FakeAVCeleb \\
        --output ./data/preprocessed/fakeavceleb.h5 \\
        --num-frames 16

FakeAVCeleb expected directory structure:
    FakeAVCeleb/
    ├── RealVideo/
    │   └── *.mp4
    └── FakeVideo/
        ├── FaceSwap/
        │   └── *.mp4
        ├── Wav2Lip/
        │   └── *.mp4
        └── .../
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

try:
    import h5py
except ImportError:
    print("ERROR: h5py is required. Install with: pip install h5py")
    sys.exit(1)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.preprocessing import (
    build_face_detector,
    process_single_video,
)


# ── Dataset Discovery ────────────────────────────────────────────────────────

def discover_fakeavceleb(data_dir: str) -> List[Dict]:
    """
    Walk FakeAVCeleb directory and discover all video files with metadata.

    Returns list of dicts: {path, label, method}
      label: 0 = real, 1 = fake
      method: forgery method name (e.g., "FaceSwap", "Wav2Lip") or "real"
    """
    data_path = Path(data_dir)
    videos = []

    # Real videos
    real_dir = data_path / "RealVideo"
    if real_dir.exists():
        for vf in sorted(real_dir.rglob("*.mp4")):
            videos.append({"path": str(vf), "label": 0, "method": "real"})

    # Also check for alternative naming
    for alt_name in ["Real", "real", "RealVideo-RealAudio"]:
        alt_dir = data_path / alt_name
        if alt_dir.exists() and alt_dir != real_dir:
            for vf in sorted(alt_dir.rglob("*.mp4")):
                videos.append({"path": str(vf), "label": 0, "method": "real"})

    # Fake videos (organized by method)
    fake_dir = data_path / "FakeVideo"
    if fake_dir.exists():
        for method_dir in sorted(fake_dir.iterdir()):
            if method_dir.is_dir():
                method_name = method_dir.name
                for vf in sorted(method_dir.rglob("*.mp4")):
                    videos.append({
                        "path": str(vf),
                        "label": 1,
                        "method": method_name,
                    })

    # Also check alternative fake directories
    for alt_name in ["Fake", "fake", "FakeVideo-RealAudio", "FakeVideo-FakeAudio"]:
        alt_dir = data_path / alt_name
        if alt_dir.exists() and alt_dir != fake_dir:
            for method_dir in sorted(alt_dir.iterdir()):
                if method_dir.is_dir():
                    for vf in sorted(method_dir.rglob("*.mp4")):
                        videos.append({
                            "path": str(vf),
                            "label": 1,
                            "method": method_dir.name,
                        })

    print(f"[Discovery] Found {len(videos)} videos in {data_dir}")
    real_count = sum(1 for v in videos if v["label"] == 0)
    fake_count = sum(1 for v in videos if v["label"] == 1)
    print(f"  Real: {real_count}, Fake: {fake_count}")

    methods = {}
    for v in videos:
        methods[v["method"]] = methods.get(v["method"], 0) + 1
    for m, c in sorted(methods.items()):
        print(f"  Method '{m}': {c} videos")

    return videos


def split_dataset(
    videos: List[Dict],
    train_size: int = 14000,
    val_size: int = 3000,
    test_size: int = 3000,
    seed: int = 42,
) -> Dict[str, List[Dict]]:
    """
    Split videos into train/val/test, stratified by forgery method.

    Per CONTEXT.md §4.1:
      Training: 14,000 videos
      Validation: 3,000 videos
      Test: 3,000 videos
      Stratification: By forgery method
    """
    rng = np.random.RandomState(seed)

    # Group by method for stratified split
    by_method: Dict[str, List[Dict]] = {}
    for v in videos:
        by_method.setdefault(v["method"], []).append(v)

    total = len(videos)
    target_total = train_size + val_size + test_size

    splits: Dict[str, List[Dict]] = {"train": [], "val": [], "test": []}

    for method, method_videos in by_method.items():
        rng.shuffle(method_videos)
        n = len(method_videos)

        # Proportional split
        n_train = int(n * train_size / target_total)
        n_val = int(n * val_size / target_total)
        n_test = n - n_train - n_val

        splits["train"].extend(method_videos[:n_train])
        splits["val"].extend(method_videos[n_train:n_train + n_val])
        splits["test"].extend(method_videos[n_train + n_val:])

    # Shuffle each split
    for split_name in splits:
        rng.shuffle(splits[split_name])

    for split_name, split_videos in splits.items():
        print(f"  [{split_name}] {len(split_videos)} videos")

    return splits


# ── HDF5 Serialization ──────────────────────────────────────────────────────

def preprocess_and_save(
    splits: Dict[str, List[Dict]],
    output_path: str,
    num_frames: int = 16,
    sample_rate: int = 16000,
    n_mels: int = 80,
    device: str = "cpu",
):
    """
    Process all videos and save to HDF5.

    Args:
        splits: Dict of split_name → list of video dicts.
        output_path: Path to output HDF5 file.
        num_frames: Frames per video (default 16).
        sample_rate: Audio sample rate (default 16000).
        n_mels: Mel bins (default 80).
        device: Device for MTCNN (default "cpu").
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Build MTCNN detector (reuse across all videos)
    detector = build_face_detector(device=device)
    if detector is None:
        print("[WARNING] MTCNN not available — using center crop fallback for all frames.")

    # First pass: process one video to determine mel time dimension F
    first_video = None
    for split_videos in splits.values():
        if split_videos:
            first_video = split_videos[0]
            break

    if first_video is None:
        print("ERROR: No videos to process!")
        return

    print(f"\n[Probe] Processing first video to determine dimensions...")
    try:
        probe_frames, probe_mel = process_single_video(
            first_video["path"], num_frames=num_frames,
            sample_rate=sample_rate, n_mels=n_mels, detector=detector,
        )
        mel_f = probe_mel.shape[2]
        print(f"  Frames shape: {probe_frames.shape}")
        print(f"  Mel shape: {probe_mel.shape}  (F={mel_f})")
    except Exception as e:
        print(f"  Probe failed: {e}, using default F=32")
        mel_f = 32

    # Create HDF5 file
    dt_str = h5py.special_dtype(vlen=str)

    with h5py.File(output_path, "w") as hf:
        for split_name, split_videos in splits.items():
            n = len(split_videos)
            if n == 0:
                continue

            print(f"\n[{split_name}] Processing {n} videos...")
            grp = hf.create_group(split_name)

            # Pre-allocate datasets
            ds_frames = grp.create_dataset(
                "frames", shape=(n, num_frames, 6, 224, 224),
                dtype=np.float32, chunks=(1, num_frames, 6, 224, 224),
            )
            ds_mel = grp.create_dataset(
                "mel", shape=(n, num_frames, n_mels, mel_f),
                dtype=np.float32, chunks=(1, num_frames, n_mels, mel_f),
            )
            ds_labels = grp.create_dataset("labels", shape=(n,), dtype=np.int64)
            ds_methods = grp.create_dataset("methods", shape=(n,), dtype=dt_str)
            ds_paths = grp.create_dataset("paths", shape=(n,), dtype=dt_str)

            errors = 0
            for i, video in enumerate(tqdm(split_videos, desc=split_name)):
                try:
                    frames_t, mel_t = process_single_video(
                        video["path"], num_frames=num_frames,
                        sample_rate=sample_rate, n_mels=n_mels,
                        detector=detector,
                    )

                    # Ensure mel has correct F dimension
                    if mel_t.shape[2] != mel_f:
                        if mel_t.shape[2] < mel_f:
                            mel_t = torch.nn.functional.pad(
                                mel_t, (0, mel_f - mel_t.shape[2])
                            )
                        else:
                            mel_t = mel_t[:, :, :mel_f]

                    ds_frames[i] = frames_t.numpy()
                    ds_mel[i] = mel_t.numpy()
                    ds_labels[i] = video["label"]
                    ds_methods[i] = video["method"]
                    ds_paths[i] = video["path"]

                except Exception as e:
                    errors += 1
                    if errors <= 5:
                        print(f"\n  [ERROR] Video {video['path']}: {e}")
                    elif errors == 6:
                        print(f"\n  [ERROR] Suppressing further error messages...")

                    # Write zeros for failed videos
                    ds_frames[i] = np.zeros((num_frames, 6, 224, 224), dtype=np.float32)
                    ds_mel[i] = np.zeros((num_frames, n_mels, mel_f), dtype=np.float32)
                    ds_labels[i] = video["label"]
                    ds_methods[i] = video["method"]
                    ds_paths[i] = video["path"]

            if errors > 0:
                print(f"\n  [{split_name}] {errors}/{n} videos failed to process.")

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n✅ Saved HDF5 to {output_path} ({file_size_mb:.1f} MB)")


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess FakeAVCeleb videos into HDF5 for training"
    )
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Path to FakeAVCeleb root directory"
    )
    parser.add_argument(
        "--output", type=str, default="./data/preprocessed/fakeavceleb.h5",
        help="Output HDF5 file path"
    )
    parser.add_argument(
        "--num-frames", type=int, default=16,
        help="Number of frames to sample per video (default: 16)"
    )
    parser.add_argument(
        "--train-size", type=int, default=14000,
        help="Training set size (default: 14000)"
    )
    parser.add_argument(
        "--val-size", type=int, default=3000,
        help="Validation set size (default: 3000)"
    )
    parser.add_argument(
        "--test-size", type=int, default=3000,
        help="Test set size (default: 3000)"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device for MTCNN face detection (default: cpu)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for split (default: 42)"
    )
    args = parser.parse_args()

    print("=" * 80)
    print("FAKEAVCELEB → HDF5 PREPROCESSING")
    print("=" * 80)

    # Step 1: Discover videos
    videos = discover_fakeavceleb(args.data_dir)
    if not videos:
        print("ERROR: No videos found. Check --data-dir path.")
        sys.exit(1)

    # Step 2: Split dataset
    print(f"\n[Split] Creating train/val/test splits...")
    splits = split_dataset(
        videos,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )

    # Step 3: Process and save
    preprocess_and_save(
        splits, args.output,
        num_frames=args.num_frames,
        device=args.device,
    )

    print("\n" + "=" * 80)
    print("✅ PREPROCESSING COMPLETE")
    print(f"   Output: {args.output}")
    print(f"   To use: set 'use_dummy_data: false' and")
    print(f"           'hdf5_path: {args.output}' in configs/default.yaml")
    print("=" * 80)


if __name__ == "__main__":
    main()
