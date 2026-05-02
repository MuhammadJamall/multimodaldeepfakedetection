#!/usr/bin/env python3
"""
preprocess_dfdc.py
------------------
Optimized preprocessing script for DFDC dataset → HDF5.

Reads filtered_metadata.csv for labels, processes each video through:
  1. MTCNN face detection → 224×224 face + mouth crops → 6-channel tensor
  2. Audio extraction → mel spectrogram

Outputs HDF5 with train/val/test splits ready for model training.

Usage:
    python scripts/preprocess_dfdc.py

    # Or with custom paths:
    python scripts/preprocess_dfdc.py \
        --data-dir Dataset/dfdc_train_part_46/dfdc_train_part_46 \
        --csv Dataset/filtered_metadata/filtered_metadata.csv \
        --output data/preprocessed/dfdc.h5 \
        --num-workers 4
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from tqdm import tqdm

try:
    import h5py
except ImportError:
    print("ERROR: h5py required. Install: pip install h5py")
    sys.exit(1)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.preprocessing import build_face_detector, process_single_video


def _get_row_value(row: Dict[str, str], *candidate_keys: str) -> str:
    """Return the first non-empty value for the provided CSV keys."""
    lowered = {str(key).strip().lower(): value for key, value in row.items()}
    for candidate_key in candidate_keys:
        value = lowered.get(candidate_key.lower())
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _parse_label_value(label_value: str) -> Optional[int]:
    """Normalize DFDC label encodings to 0/1, or return None if unknown."""
    normalized = label_value.strip().lower()
    if normalized in {"fake", "1", "true", "yes"}:
        return 1
    if normalized in {"real", "0", "false", "no"}:
        return 0
    return None


# ── DFDC Dataset Discovery ───────────────────────────────────────────────────

def discover_dfdc(
    video_dir: str,
    csv_path: str,
) -> List[Dict]:
    """
    Read DFDC metadata CSV and match with video files on disk.

    CSV expected columns: filename, label (REAL/FAKE), ...
    Returns list of {path, label (0/1), method, filename}
    """
    video_dir = Path(video_dir)
    csv_path = Path(csv_path)

    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}")
        sys.exit(1)
    if not video_dir.exists():
        print(f"ERROR: Video directory not found: {video_dir}")
        sys.exit(1)

    # Read CSV
    entries = []
    skipped_rows = 0
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = _get_row_value(row, "filename", "file", "video", "video_name")
            label_str = _get_row_value(row, "label", "class", "target", "is_fake")

            if not filename or not label_str:
                skipped_rows += 1
                continue

            label = _parse_label_value(label_str)
            if label is None:
                skipped_rows += 1
                continue

            method = "deepfake" if label == 1 else "real"
            entries.append({
                "filename": filename,
                "label": label,
                "method": method,
            })

    # Match with actual files on disk
    existing_files = set(f.name for f in video_dir.glob("*.mp4"))
    videos = []
    missing = 0

    for entry in entries:
        if entry["filename"] in existing_files:
            videos.append({
                "path": str(video_dir / entry["filename"]),
                "label": entry["label"],
                "method": entry["method"],
                "filename": entry["filename"],
            })
        else:
            missing += 1

    # Stats
    real_count = sum(1 for v in videos if v["label"] == 0)
    fake_count = sum(1 for v in videos if v["label"] == 1)

    print(f"\n{'='*70}")
    print(f"  DFDC DATASET DISCOVERY")
    print(f"{'='*70}")
    print(f"  CSV entries:  {len(entries)}")
    print(f"  Files on disk: {len(existing_files)}")
    print(f"  Matched:       {len(videos)}")
    if skipped_rows > 0:
        print(f"  Skipped rows:  {skipped_rows} (missing filename/label)")
    if missing > 0:
        print(f"  Missing:       {missing} (in CSV but not on disk)")
    print(f"  Real videos:   {real_count}")
    print(f"  Fake videos:   {fake_count}")
    print(f"  Ratio:         {fake_count/(real_count+1e-9):.1f}:1 fake:real")

    return videos


def split_dataset(
    videos: List[Dict],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[Dict]]:
    """
    Stratified split by label (real/fake) into train/val/test.

    With 2200 videos:
      train: ~1540
      val:   ~330
      test:  ~330
    """
    rng = np.random.RandomState(seed)

    # Separate by label
    reals = [v for v in videos if v["label"] == 0]
    fakes = [v for v in videos if v["label"] == 1]
    rng.shuffle(reals)
    rng.shuffle(fakes)

    def do_split(vids):
        n = len(vids)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        return (
            vids[:n_train],
            vids[n_train:n_train + n_val],
            vids[n_train + n_val:],
        )

    r_train, r_val, r_test = do_split(reals)
    f_train, f_val, f_test = do_split(fakes)

    splits = {
        "train": r_train + f_train,
        "val":   r_val + f_val,
        "test":  r_test + f_test,
    }

    # Shuffle each split
    for k in splits:
        rng.shuffle(splits[k])

    print(f"\n  Split (seed={seed}):")
    for k, v in splits.items():
        r = sum(1 for x in v if x["label"] == 0)
        f = sum(1 for x in v if x["label"] == 1)
        print(f"    {k:5s}: {len(v):5d} videos  (real={r}, fake={f})")

    return splits


# ── Video Processing ─────────────────────────────────────────────────────────

def process_video_safe(
    video_path: str,
    num_frames: int,
    sample_rate: int,
    n_mels: int,
    detector,
) -> Tuple[Optional[np.ndarray], Optional[object]]:
    """
    Process a single video, returning numpy arrays or None on failure.
    """
    try:
        frames_t, mel_t = process_single_video(
            video_path,
            num_frames=num_frames,
            sample_rate=sample_rate,
            n_mels=n_mels,
            detector=detector, 
        )
        return frames_t.numpy(), mel_t.numpy()
    except Exception as e:
        return None, str(e)


# ── HDF5 Serialization ──────────────────────────────────────────────────────

def preprocess_and_save(
    splits: Dict[str, List[Dict]],
    output_path: str,
    num_frames: int = 16,
    sample_rate: int = 16000,
    n_mels: int = 80,
    resume: bool = False,
):
    """
    Process all videos and save to HDF5.
    Supports resume: if HDF5 exists, skip already-processed videos.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Build MTCNN (runs on CPU)
    print("\n[Init] Building MTCNN face detector...")
    detector = build_face_detector(device="cpu")
    if detector is None:
        print("  ⚠ MTCNN not available — using center crop fallback")
    else:
        print("  ✅ MTCNN ready")

    # Probe first video to get mel time dimension
    mel_f = 32  # default
    for split_vids in splits.values():
        if split_vids:
            print(f"\n[Probe] Testing with: {Path(split_vids[0]['path']).name}")
            result = process_video_safe(
                split_vids[0]["path"], num_frames, sample_rate, n_mels, detector
            )
            if result[0] is not None:
                mel_f = result[1].shape[2]
                print(f"  Frames: {result[0].shape}")
                print(f"  Mel:    {result[1].shape} (F={mel_f})")
            else:
                print(f"  Probe failed: {result[1]}, using F={mel_f}")
            break

    # Create/open HDF5
    dt_str = h5py.special_dtype(vlen=str)
    mode = "a" if resume else "w"

    start_time = time.time()

    with h5py.File(output_path, mode) as hf:
        for split_name, split_videos in splits.items():
            n = len(split_videos)
            if n == 0:
                continue

            # Check resume
            start_idx = 0
            if resume and split_name in hf:
                grp = hf[split_name]
                # Find first zero entry to resume from
                existing_paths = grp["paths"][:]
                for i, p in enumerate(existing_paths):
                    if not p:
                        start_idx = i
                        break
                else:
                    start_idx = n  # all done
                if start_idx >= n:
                    print(f"\n[{split_name}] Already complete, skipping.")
                    continue
                print(f"\n[{split_name}] Resuming from index {start_idx}/{n}")
            else:
                # Create fresh datasets
                if split_name in hf:
                    del hf[split_name]
                grp = hf.create_group(split_name)
                grp.create_dataset(
                    "frames", shape=(n, num_frames, 6, 224, 224),
                    dtype=np.float32, chunks=(1, num_frames, 6, 224, 224),
                    compression="gzip", compression_opts=1,
                )
                grp.create_dataset(
                    "mel", shape=(n, num_frames, n_mels, mel_f),
                    dtype=np.float32, chunks=(1, num_frames, n_mels, mel_f),
                    compression="gzip", compression_opts=1,
                )
                grp.create_dataset("labels", shape=(n,), dtype=np.int64)
                grp.create_dataset("methods", shape=(n,), dtype=dt_str)
                grp.create_dataset("paths", shape=(n,), dtype=dt_str)
                print(f"\n[{split_name}] Processing {n} videos...")

            grp = hf[split_name]
            errors = 0
            processed = 0

            pbar = tqdm(
                range(start_idx, n),
                desc=f"{split_name}",
                initial=start_idx,
                total=n,
                unit="vid",
            )

            for i in pbar:
                video = split_videos[i]
                frames_np, mel_or_err = process_video_safe(
                    video["path"], num_frames, sample_rate, n_mels, detector
                )

                if frames_np is not None:
                    mel_np = mel_or_err
                    # Pad/truncate mel time dimension
                    if mel_np.shape[2] != mel_f:
                        if mel_np.shape[2] < mel_f:
                            pad = np.zeros(
                                (*mel_np.shape[:2], mel_f - mel_np.shape[2]),
                                dtype=np.float32,
                            )
                            mel_np = np.concatenate([mel_np, pad], axis=2)
                        else:
                            mel_np = mel_np[:, :, :mel_f]

                    grp["frames"][i] = frames_np
                    grp["mel"][i] = mel_np
                    processed += 1
                else:
                    # Failed — write zeros
                    grp["frames"][i] = np.zeros(
                        (num_frames, 6, 224, 224), dtype=np.float32
                    )
                    grp["mel"][i] = np.zeros(
                        (num_frames, n_mels, mel_f), dtype=np.float32
                    )
                    errors += 1
                    if errors <= 3:
                        tqdm.write(f"  ✗ {Path(video['path']).name}: {mel_or_err}")

                grp["labels"][i] = video["label"]
                grp["methods"][i] = video["method"]
                grp["paths"][i] = video["path"]

                # Update progress bar
                elapsed = time.time() - start_time
                rate = (processed + errors) / max(elapsed, 1)
                remaining = (n - i - 1) / max(rate, 0.01)
                pbar.set_postfix(
                    ok=processed, err=errors,
                    eta=f"{remaining/60:.0f}m",
                )

                # Flush every 50 videos (crash safety)
                if (i + 1) % 50 == 0:
                    hf.flush()

            print(f"  [{split_name}] Done: {processed} ok, {errors} failed")

    elapsed = time.time() - start_time
    file_size = os.path.getsize(output_path) / (1024**2)

    print(f"\n{'='*70}")
    print(f"  ✅ PREPROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"  Output:    {output_path}")
    print(f"  Size:      {file_size:.0f} MB")
    print(f"  Time:      {elapsed/60:.1f} minutes")
    print(f"  Speed:     {sum(len(v) for v in splits.values())/max(elapsed,1):.1f} videos/sec")


# ── Entry Point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DFDC → HDF5 Preprocessing for DeepDetect"
    )
    parser.add_argument(
        "--data-dir", type=str,
        default=str(PROJECT_ROOT / "Dataset" / "dfdc_train_part_46" / "dfdc_train_part_46"),
        help="Path to folder containing .mp4 files",
    )
    parser.add_argument(
        "--csv", type=str,
        default=str(PROJECT_ROOT / "Dataset" / "filtered_metadata" / "filtered_metadata.csv"),
        help="Path to filtered_metadata.csv",
    )
    parser.add_argument(
        "--output", type=str,
        default=str(PROJECT_ROOT / "data" / "preprocessed" / "dfdc.h5"),
        help="Output HDF5 path",
    )
    parser.add_argument(
        "--num-frames", type=int, default=16,
        help="Frames per video (default: 16)",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.70,
        help="Train split ratio (default: 0.70)",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.15,
        help="Val split ratio (default: 0.15)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--num-workers", type=int, default=1,
        help="Reserved for future parallel preprocessing; currently runs serially",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing HDF5 (skip processed videos)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  DFDC → HDF5 PREPROCESSING PIPELINE")
    print("  DeepDetect Multimodal Deepfake Detection")
    print("=" * 70)
    print(f"  Video dir:  {args.data_dir}")
    print(f"  CSV:        {args.csv}")
    print(f"  Output:     {args.output}")
    print(f"  Frames/vid: {args.num_frames}")
    print(f"  Split:      {args.train_ratio:.0%} / {args.val_ratio:.0%} / {1-args.train_ratio-args.val_ratio:.0%}")
    if args.num_workers != 1:
        print(f"  Workers:    {args.num_workers} (accepted, but this script still processes videos serially)")

    # Step 1: Discover
    videos = discover_dfdc(args.data_dir, args.csv)
    if not videos:
        print("ERROR: No videos found!")
        sys.exit(1)

    # Step 2: Split
    splits = split_dataset(
        videos,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    # Step 3: Process + Save
    preprocess_and_save(
        splits,
        args.output,
        num_frames=args.num_frames,
        resume=args.resume,
    )

    print(f"\n  Next steps:")
    print(f"    1. Upload {args.output} to Google Drive")
    print(f"    2. Train on Colab: python training/train.py --config configs/default.yaml")
    print(f"    3. Download best_auroc.pt → checkpoints/")
    print(f"    4. Run: python web/server.py --checkpoint checkpoints/best_auroc.pt")


if __name__ == "__main__":
    main()
