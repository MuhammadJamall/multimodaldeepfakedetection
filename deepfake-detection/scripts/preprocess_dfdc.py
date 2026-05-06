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

Improvements over v1:
  - Real parallel processing via ProcessPoolExecutor (--num-workers actually works)
  - Per-worker MTCNN initialisation so GPU/CPU is used efficiently
  - mel_f computed from parameters — no fragile probe + magic-number fallback
  - h5py.special_dtype (deprecated since h5py 3.0) → h5py.string_dtype()
  - process_video_safe uses a proper named result instead of mixed return type
  - discover_dfdc raises exceptions instead of calling sys.exit
  - _get_row_value lowercased dict built once per row, not twice
  - Batch HDF5 writes (collect N results then write at once — fewer I/O calls)
  - Auto GPU detection for MTCNN; falls back to CPU gracefully
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

try:
    import h5py
except ImportError:
    print("ERROR: h5py required.  pip install h5py")
    sys.exit(1)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.preprocessing import build_face_detector, process_single_video


# ── Typed result container ────────────────────────────────────────────────────

@dataclass
class VideoResult:
    """Holds the outcome of processing one video."""
    index:    int
    label:    int
    method:   str
    path:     str
    frames:   Optional[np.ndarray] = None   # (T, 6, 224, 224) float32
    mel:      Optional[np.ndarray] = None   # (T, n_mels, F)   float32
    error:    Optional[str]        = None

    @property
    def ok(self) -> bool:
        return self.frames is not None and self.mel is not None


# ── CSV helpers ───────────────────────────────────────────────────────────────

def _get_row_value(lowered_row: Dict[str, str], *candidate_keys: str) -> str:
    """
    Return the first non-empty value found among candidate_keys.

    IMPROVEMENT: caller pre-lowercases the row dict once per row so
    we don't rebuild it on every call.
    """
    for key in candidate_keys:
        value = lowered_row.get(key.lower())
        if value is not None and value.strip():
            return value.strip()
    return ""


def _parse_label_value(label_value: str) -> Optional[int]:
    """Normalise DFDC label encodings to 0/1, or return None if unknown."""
    normalized = label_value.strip().lower()
    if normalized in {"fake", "1", "true", "yes"}:
        return 1
    if normalized in {"real", "0", "false", "no"}:
        return 0
    return None


# ── DFDC Dataset Discovery ────────────────────────────────────────────────────

def discover_dfdc(video_dir: str, csv_path: str) -> List[Dict]:
    """
    Read DFDC metadata CSV and match with video files on disk.

    IMPROVEMENT: raises ValueError/FileNotFoundError instead of
    calling sys.exit() — cleaner and testable.

    Returns:
        List of {path, label (0/1), method, filename}

    Raises:
        FileNotFoundError: If csv_path or video_dir do not exist.
        ValueError:        If no matching videos are found.
    """
    video_dir = Path(video_dir)
    csv_path  = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")

    entries: List[Dict] = []
    skipped_rows = 0

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Pre-lowercase once per row — shared by both _get_row_value calls
            lowered = {k.strip().lower(): v for k, v in row.items()}

            filename  = _get_row_value(lowered, "filename", "file", "video", "video_name")
            label_str = _get_row_value(lowered, "label", "class", "target", "is_fake")

            if not filename or not label_str:
                skipped_rows += 1
                continue

            label = _parse_label_value(label_str)
            if label is None:
                skipped_rows += 1
                continue

            entries.append({
                "filename": filename,
                "label":    label,
                "method":   "deepfake" if label == 1 else "real",
            })

    existing_files = {f.name for f in video_dir.glob("*.mp4")}
    videos: List[Dict] = []
    missing = 0

    for entry in entries:
        if entry["filename"] in existing_files:
            videos.append({
                "path":     str(video_dir / entry["filename"]),
                "label":    entry["label"],
                "method":   entry["method"],
                "filename": entry["filename"],
            })
        else:
            missing += 1

    real_count = sum(1 for v in videos if v["label"] == 0)
    fake_count = sum(1 for v in videos if v["label"] == 1)

    print(f"\n{'='*70}")
    print(f"  DFDC DATASET DISCOVERY")
    print(f"{'='*70}")
    print(f"  CSV entries:   {len(entries)}")
    print(f"  Files on disk: {len(existing_files)}")
    print(f"  Matched:       {len(videos)}")
    if skipped_rows: print(f"  Skipped rows:  {skipped_rows}")
    if missing:      print(f"  Missing:       {missing}")
    print(f"  Real:          {real_count}")
    print(f"  Fake:          {fake_count}")
    print(f"  Ratio:         {fake_count / max(real_count, 1):.1f}:1 fake:real")

    if not videos:
        raise ValueError("No videos matched between CSV and disk.")

    return videos


def split_dataset(
    videos: List[Dict],
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    seed: int = 42,
) -> Dict[str, List[Dict]]:
    """Stratified split by label (real/fake) into train/val/test."""
    rng   = np.random.RandomState(seed)
    reals = [v for v in videos if v["label"] == 0]
    fakes = [v for v in videos if v["label"] == 1]
    rng.shuffle(reals)
    rng.shuffle(fakes)

    def _split(vids: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        n       = len(vids)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)
        return vids[:n_train], vids[n_train:n_train + n_val], vids[n_train + n_val:]

    r_tr, r_va, r_te = _split(reals)
    f_tr, f_va, f_te = _split(fakes)

    splits = {
        "train": r_tr + f_tr,
        "val":   r_va + f_va,
        "test":  r_te + f_te,
    }
    for k in splits:
        rng.shuffle(splits[k])

    print(f"\n  Split (seed={seed}):")
    for k, v in splits.items():
        r = sum(1 for x in v if x["label"] == 0)
        f = sum(1 for x in v if x["label"] == 1)
        print(f"    {k:5s}: {len(v):5d} videos  (real={r}, fake={f})")

    return splits


# ── mel_f computation (no probe needed) ──────────────────────────────────────

def compute_mel_f(
    num_frames:  int   = 16,
    sample_rate: int   = 16_000,
    video_fps:   float = 30.0,
    hop_ms:      float = 10.0,
) -> int:
    """
    Compute the mel time dimension F from first principles.

    IMPROVEMENT over v1: v1 processed the first video just to read
    mel.shape[2], then fell back to the magic number 32 on failure.
    This function derives F directly from the parameters — always
    correct, zero I/O.

    Formula:
        total_audio_samples  = sample_rate * (num_frames / fps)
        total_mel_time_frames = ceil(total_audio_samples / hop_length)
        F = total_mel_time_frames // num_frames
    """
    hop_length           = int(sample_rate * hop_ms / 1000)          # 160
    duration_secs        = num_frames / video_fps                     # ~0.53 s
    total_audio_samples  = int(sample_rate * duration_secs)
    total_mel_frames     = math.ceil(total_audio_samples / hop_length)
    frames_per_window    = max(1, total_mel_frames // num_frames)
    return frames_per_window


# ── Per-worker initialiser for parallel processing ───────────────────────────

_worker_detector = None   # module-level so each worker process owns one

def _worker_init(device: str) -> None:
    """
    Initialise MTCNN once per worker process.

    IMPROVEMENT: v1 created a single detector in the main process and
    couldn't share it across workers.  With ProcessPoolExecutor each
    worker calls this initialiser once and stores the detector in a
    module-level variable — no repeated construction overhead.
    """
    global _worker_detector
    _worker_detector = build_face_detector(device=device)


def _process_one(
    args: Tuple[int, str, int, str, int, int, int],
) -> VideoResult:
    """
    Worker function — processes a single video.
    Called in a subprocess by ProcessPoolExecutor.

    Args:
        args: (index, path, label, method, num_frames, sample_rate, n_mels)
    """
    index, path, label, method, num_frames, sample_rate, n_mels = args
    try:
        frames_t, mel_t = process_single_video(
            path,
            num_frames=num_frames,
            sample_rate=sample_rate,
            n_mels=n_mels,
            detector=_worker_detector,
        )
        return VideoResult(
            index=index, label=label, method=method, path=path,
            frames=frames_t.numpy(), mel=mel_t.numpy(),
        )
    except Exception as exc:
        return VideoResult(
            index=index, label=label, method=method, path=path,
            error=str(exc),
        )


# ── HDF5 Serialization ────────────────────────────────────────────────────────

# How many results to collect before writing to disk.
# Larger = fewer I/O round-trips; smaller = less RAM.
_WRITE_BATCH = 64


def preprocess_and_save(
    splits:      Dict[str, List[Dict]],
    output_path: str,
    num_frames:  int   = 16,
    sample_rate: int   = 16_000,
    n_mels:      int   = 80,
    num_workers: int   = 1,
    resume:      bool  = False,
    device:      str   = "auto",
) -> None:
    """
    Process all videos in parallel and write results to HDF5.

    IMPROVEMENTS over v1:
      - num_workers actually spawns that many worker processes
      - mel_f is computed, not probed
      - h5py.string_dtype() replaces deprecated h5py.special_dtype(vlen=str)
      - Results are buffered and written in batches (fewer HDF5 I/O calls)
      - Resume uses a clean boolean mask dataset instead of path scanning
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Resolve device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Init] Device: {device}")

    # Compute mel F from parameters — no probe needed
    mel_f = compute_mel_f(num_frames=num_frames, sample_rate=sample_rate)
    print(f"[Init] mel_f={mel_f} (computed from params)")

    str_dtype  = h5py.string_dtype()   # replaces deprecated special_dtype
    hdf5_mode  = "a" if resume else "w"
    start_time = time.time()

    with h5py.File(output_path, hdf5_mode) as hf:
        for split_name, split_videos in splits.items():
            n = len(split_videos)
            if n == 0:
                continue

            # ── Resume support ────────────────────────────────────────────
            start_idx = 0
            if resume and split_name in hf:
                done_mask = hf[split_name].get("done")
                if done_mask is not None:
                    start_idx = int(np.argmin(done_mask[:]))  # first False
                    if done_mask[start_idx]:
                        print(f"\n[{split_name}] Already complete, skipping.")
                        continue
                    print(f"\n[{split_name}] Resuming from {start_idx}/{n}")
            else:
                # Fresh datasets
                if split_name in hf:
                    del hf[split_name]
                grp = hf.create_group(split_name)
                grp.create_dataset(
                    "frames",
                    shape=(n, num_frames, 6, 224, 224), dtype=np.float32,
                    chunks=(1, num_frames, 6, 224, 224),
                    compression="gzip", compression_opts=1,
                )
                grp.create_dataset(
                    "mel",
                    shape=(n, num_frames, n_mels, mel_f), dtype=np.float32,
                    chunks=(1, num_frames, n_mels, mel_f),
                    compression="gzip", compression_opts=1,
                )
                grp.create_dataset("labels",  shape=(n,), dtype=np.int64)
                grp.create_dataset("methods", shape=(n,), dtype=str_dtype)
                grp.create_dataset("paths",   shape=(n,), dtype=str_dtype)
                # Boolean mask — True = successfully written
                grp.create_dataset("done", shape=(n,), dtype=bool, data=np.zeros(n, dtype=bool))
                print(f"\n[{split_name}] Processing {n} videos "
                      f"with {num_workers} worker(s)...")

            grp      = hf[split_name]
            ok_count = int(np.sum(grp["done"][:start_idx]))
            err_count = start_idx - ok_count

            # Build work items for the remaining videos
            work = [
                (i, v["path"], v["label"], v["method"], num_frames, sample_rate, n_mels)
                for i, v in enumerate(split_videos)
                if i >= start_idx
            ]

            pbar   = tqdm(total=n, initial=start_idx, desc=split_name, unit="vid")
            buffer: List[VideoResult] = []

            def _flush_buffer(buf: List[VideoResult]) -> None:
                """Write a batch of results to HDF5 in one go."""
                for res in buf:
                    mel_np = res.mel if res.ok else np.zeros((num_frames, n_mels, mel_f), np.float32)
                    fr_np  = res.frames if res.ok else np.zeros((num_frames, 6, 224, 224), np.float32)

                    # Align mel F dimension (videos vary slightly in duration)
                    if res.ok and mel_np.shape[2] != mel_f:
                        if mel_np.shape[2] < mel_f:
                            pad    = np.zeros((*mel_np.shape[:2], mel_f - mel_np.shape[2]), np.float32)
                            mel_np = np.concatenate([mel_np, pad], axis=2)
                        else:
                            mel_np = mel_np[:, :, :mel_f]

                    grp["frames"][res.index]  = fr_np
                    grp["mel"][res.index]     = mel_np
                    grp["labels"][res.index]  = res.label
                    grp["methods"][res.index] = res.method
                    grp["paths"][res.index]   = res.path
                    grp["done"][res.index]    = res.ok
                hf.flush()

            # ── Parallel execution ────────────────────────────────────────
            executor_cls = (
                cf.ProcessPoolExecutor if num_workers > 1 else cf.ThreadPoolExecutor
            )
            init_kwargs = (
                {"initializer": _worker_init, "initargs": (device,)}
                if num_workers > 1 else {}
            )

            with executor_cls(max_workers=num_workers, **init_kwargs) as executor:
                # Submit all at once; iterate as they complete
                futures = {executor.submit(_process_one, w): w for w in work}

                for future in cf.as_completed(futures):
                    res = future.result()
                    buffer.append(res)

                    if res.ok:
                        ok_count += 1
                    else:
                        err_count += 1
                        if err_count <= 5:
                            tqdm.write(f"  ✗ {Path(res.path).name}: {res.error}")

                    pbar.update(1)
                    elapsed   = time.time() - start_time
                    total_done = ok_count + err_count
                    rate      = total_done / max(elapsed, 1)
                    remaining = (n - start_idx - total_done) / max(rate, 0.01)
                    pbar.set_postfix(ok=ok_count, err=err_count, eta=f"{remaining/60:.0f}m")

                    # Write when buffer is full
                    if len(buffer) >= _WRITE_BATCH:
                        _flush_buffer(buffer)
                        buffer.clear()

            # Write any remaining results
            if buffer:
                _flush_buffer(buffer)
                buffer.clear()

            pbar.close()
            print(f"  [{split_name}] Done: {ok_count} ok, {err_count} failed")

    elapsed   = time.time() - start_time
    file_size = os.path.getsize(output_path) / (1024 ** 2)
    total_vids = sum(len(v) for v in splits.values())

    print(f"\n{'='*70}")
    print(f"  ✅ PREPROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"  Output : {output_path}")
    print(f"  Size   : {file_size:.0f} MB")
    print(f"  Time   : {elapsed / 60:.1f} minutes")
    print(f"  Speed  : {total_vids / max(elapsed, 1):.1f} videos/sec")


# ── Entry Point ───────────────────────────────────────────────────────────────

def main() -> None:
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
    parser.add_argument("--num-frames",  type=int,   default=16)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio",   type=float, default=0.15)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument(
        "--num-workers", type=int, default=1,
        help="Number of parallel worker processes (default: 1 = serial)",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for MTCNN (default: auto-detect)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing HDF5 — skip already-processed videos",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  DFDC → HDF5 PREPROCESSING PIPELINE")
    print("  DeepDetect Multimodal Deepfake Detection")
    print("=" * 70)
    print(f"  Video dir  : {args.data_dir}")
    print(f"  CSV        : {args.csv}")
    print(f"  Output     : {args.output}")
    print(f"  Frames/vid : {args.num_frames}")
    print(f"  Workers    : {args.num_workers}")
    print(f"  Device     : {args.device}")
    print(f"  Split      : {args.train_ratio:.0%} / {args.val_ratio:.0%} / "
          f"{1 - args.train_ratio - args.val_ratio:.0%}")

    try:
        videos = discover_dfdc(args.data_dir, args.csv)
    except (FileNotFoundError, ValueError) as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)

    splits = split_dataset(
        videos,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    preprocess_and_save(
        splits,
        args.output,
        num_frames=args.num_frames,
        num_workers=args.num_workers,
        device=args.device,
        resume=args.resume,
    )

    print(f"\n  Next steps:")
    print(f"    1. Upload {args.output} to Google Drive")
    print(f"    2. Train: python training/train.py --config configs/default.yaml")
    print(f"    3. Run:   python web/server.py --checkpoint checkpoints/best_auroc.pt")


if __name__ == "__main__":
    main()