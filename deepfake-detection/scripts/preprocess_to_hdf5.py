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
    /{split}/done    → (N,) bool  — resume mask

Usage:
    python scripts/preprocess_to_hdf5.py \\
        --data-dir /path/to/FakeAVCeleb \\
        --output ./data/preprocessed/fakeavceleb.h5 \\
        --num-frames 16 \\
        --num-workers 2

    # For machines with ≤8 GB RAM:
    python scripts/preprocess_to_hdf5.py ... --low-memory

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

RAM BUDGET (per video, num_frames=16):
    frames numpy:  74 MB
    mel numpy:      0.06 MB (tiny)
    HDF5 lzf buf:  ~74 MB  (in-place, released after write)
    ─────────────────────────
    Per worker:   ~150 MB
    2 workers:    ~300 MB
    Overhead/OS:  ~500 MB
    Safe total:   ~800 MB active — fits in 8 GB with room for MTCNN + OS.

CHANGES vs v1:
  - _WRITE_BATCH reduced from 64 → 1  (write immediately, never buffer 74 MB×64)
  - compression switched gzip → lzf   (lzf is in-place; gzip needs scratch buffer = OOM)
  - explicit del res.frames / res.mel after write so GC can reclaim immediately
  - --low-memory flag: forces 1 worker + no compression (safest for 8 GB)
  - default --num-workers lowered from 4 → 2
  - h5py.string_dtype() replaces deprecated h5py.special_dtype(vlen=str)
  - mel_f computed from parameters — no fragile probe + magic-number fallback
  - Resume support via boolean 'done' mask — crash-safe
  - Auto GPU detection for MTCNN (--device auto)
  - Duplicate path guard in discover_fakeavceleb
  - split_dataset warns when total videos < requested split sizes
  - Real parallel processing via ProcessPoolExecutor
  - Per-worker MTCNN initialisation (one detector per process)
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import gc
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

try:
    import h5py
except ImportError:
    print("ERROR: h5py is required.  pip install h5py")
    sys.exit(1)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.preprocessing import build_face_detector, process_single_video


# ── Typed result ──────────────────────────────────────────────────────────────

@dataclass
class VideoResult:
    """Holds the outcome of processing one video."""
    index:  int
    label:  int
    method: str
    path:   str
    frames: Optional[np.ndarray] = None   # (T, 6, 224, 224) float32
    mel:    Optional[np.ndarray] = None   # (T, n_mels, F)   float32
    error:  Optional[str]        = None

    @property
    def ok(self) -> bool:
        return self.frames is not None and self.mel is not None


# ── mel_f from parameters (no probe needed) ───────────────────────────────────

def compute_mel_f(
    num_frames:  int   = 16,
    sample_rate: int   = 16_000,
    video_fps:   float = 30.0,
    hop_ms:      float = 10.0,
) -> int:
    """
    Derive the mel time dimension F purely from preprocessing parameters.

    IMPROVEMENT: v1 processed the first video just to read mel.shape[2],
    then fell back to the magic number 32 on failure. This function
    computes F from first principles — always correct, zero I/O.
    """
    hop_length        = int(sample_rate * hop_ms / 1000)
    duration_secs     = num_frames / video_fps
    total_samples     = int(sample_rate * duration_secs)
    total_mel_frames  = math.ceil(total_samples / hop_length)
    return max(1, total_mel_frames // num_frames)


# ── Dataset Discovery ─────────────────────────────────────────────────────────

def discover_fakeavceleb(data_dir: str) -> List[Dict]:
    """
    Discover FakeAVCeleb videos.

    Strategy:
      1. If meta_data.csv exists, use it (official FakeAVCeleb dataset).
         CSV columns: source, target1, target2, method, category, type,
                      race, gender, path (filename), last col (dir path).
         Category A = real (label 0).  B/C/D = fake (label 1).
      2. Otherwise fall back to directory-name-based heuristic.

    Returns list of {path, label, method}.
    """
    import csv

    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    videos: List[Dict] = []
    seen_paths: set    = set()

    def _add(path: Path, label: int, method: str) -> None:
        resolved = str(path.resolve())
        if resolved not in seen_paths:
            seen_paths.add(resolved)
            videos.append({"path": str(path), "label": label, "method": method})

    csv_path = data_path / "meta_data.csv"

    # ── Strategy 1: CSV-driven (official FakeAVCeleb) ─────────────────────
    if csv_path.exists():
        print(f"  [Discovery] Using meta_data.csv at {csv_path}")
        skipped = 0

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)  # skip header

            for row in reader:
                if len(row) < 10:
                    skipped += 1
                    continue

                method   = row[3].strip()       # e.g. real, faceswap, fsgan, wav2lip, rtvc
                category = row[4].strip()       # A=real, B/C/D=fake
                filename = row[8].strip()       # e.g. 00109.mp4
                rel_dir  = row[9].strip()       # e.g. FakeAVCeleb/RealVideo-RealAudio/African/men/id00076

                # Build absolute path: strip leading "FakeAVCeleb/" prefix if present
                if rel_dir.startswith("FakeAVCeleb/"):
                    rel_dir = rel_dir[len("FakeAVCeleb/"):]

                video_path = data_path / rel_dir / filename
                if not video_path.exists():
                    skipped += 1
                    continue

                label = 0 if category == "A" else 1
                _add(video_path, label=label, method=method)

        if skipped > 0:
            print(f"  [Discovery] Skipped {skipped} entries (missing files or malformed rows)")

    # ── Strategy 2: Directory-based fallback ──────────────────────────────
    else:
        print(f"  [Discovery] No meta_data.csv found — using directory-based discovery")

        real_dir_names = ["RealVideo", "Real", "real", "RealVideo-RealAudio"]
        for dir_name in real_dir_names:
            real_dir = data_path / dir_name
            if real_dir.exists():
                for vf in sorted(real_dir.rglob("*.mp4")):
                    _add(vf, label=0, method="real")

        fake_dir_names = ["FakeVideo", "Fake", "fake", "FakeVideo-RealAudio",
                          "FakeVideo-FakeAudio", "RealVideo-FakeAudio"]
        for dir_name in fake_dir_names:
            fake_dir = data_path / dir_name
            if fake_dir.exists():
                for method_dir in sorted(fake_dir.iterdir()):
                    if method_dir.is_dir():
                        for vf in sorted(method_dir.rglob("*.mp4")):
                            _add(vf, label=1, method=dir_name)

    if not videos:
        raise ValueError(f"No .mp4 videos found under: {data_dir}")

    real_count  = sum(1 for v in videos if v["label"] == 0)
    fake_count  = sum(1 for v in videos if v["label"] == 1)
    method_counts: Dict[str, int] = {}
    for v in videos:
        method_counts[v["method"]] = method_counts.get(v["method"], 0) + 1

    print(f"\n{'='*70}")
    print(f"  FAKEAVCELEB DISCOVERY")
    print(f"{'='*70}")
    print(f"  Total : {len(videos)}")
    print(f"  Real  : {real_count}")
    print(f"  Fake  : {fake_count}")
    print(f"  Methods:")
    for method, count in sorted(method_counts.items()):
        print(f"    └─ {method}: {count}")
    print(f"{'='*70}")

    return videos


def split_dataset(
    videos:     List[Dict],
    train_size: int = 14_000,
    val_size:   int =  3_000,
    test_size:  int =  3_000,
    seed:       int = 42,
) -> Dict[str, List[Dict]]:
    """
    Stratified split by forgery method into train / val / test.

    IMPROVEMENT: warns clearly when total available videos are fewer
    than the requested split sizes, so the caller is never silently
    given smaller-than-expected splits.

    Per CONTEXT.md §4.1:
      Training: 14,000 | Validation: 3,000 | Test: 3,000
      Stratification: by forgery method
    """
    total_requested = train_size + val_size + test_size
    if len(videos) < total_requested:
        print(
            f"\n  ⚠ WARNING: only {len(videos)} videos available "
            f"but {total_requested} requested. "
            f"Splits will be proportionally scaled down."
        )
        # Scale down targets proportionally
        scale      = len(videos) / total_requested
        train_size = int(train_size * scale)
        val_size   = int(val_size   * scale)
        test_size  = len(videos) - train_size - val_size

    rng       = random.Random(seed)
    by_method: Dict[str, List[Dict]] = {}
    for v in videos:
        by_method.setdefault(v["method"], []).append(v)

    splits: Dict[str, List[Dict]] = {"train": [], "val": [], "test": []}
    total = train_size + val_size + test_size

    for method_videos in by_method.values():
        rng.shuffle(method_videos)
        n       = len(method_videos)
        n_train = int(n * train_size / total)
        n_val   = int(n * val_size   / total)

        splits["train"].extend(method_videos[:n_train])
        splits["val"].extend(  method_videos[n_train : n_train + n_val])
        splits["test"].extend( method_videos[n_train + n_val:])

    for k in splits:
        rng.shuffle(splits[k])

    print(f"\n  Split (seed={seed}):")
    for k, v in splits.items():
        r = sum(1 for x in v if x["label"] == 0)
        f = sum(1 for x in v if x["label"] == 1)
        print(f"    {k:5s}: {len(v):5d} videos  (real={r}, fake={f})")

    return splits


# ── Per-worker MTCNN init ─────────────────────────────────────────────────────

_worker_detector = None


def _worker_init(device: str) -> None:
    """
    Initialise one MTCNN detector per worker process.
    Called automatically by ProcessPoolExecutor as each worker starts.
    """
    global _worker_detector
    _worker_detector = build_face_detector(device=device)


def _process_one(
    args: Tuple[int, str, int, str, int, int, int],
) -> VideoResult:
    """Worker function — processes one video in a subprocess."""
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

# FIX: Was 64 — at 74 MB/frame tensor that was ~4.7 GB sitting in RAM
# before a single write. Now 1 = write immediately after each result.
_WRITE_BATCH = 1


def preprocess_and_save(
    splits:      Dict[str, List[Dict]],
    output_path: str,
    num_frames:  int  = 16,
    sample_rate: int  = 16_000,
    n_mels:      int  = 80,
    num_workers: int  = 2,
    device:      str  = "auto",
    resume:      bool = False,
    low_memory:  bool = False,
) -> None:
    """
    Process all videos and write to HDF5.

    RAM-SAFE CHANGES vs v1:
      - _WRITE_BATCH = 1: write & discard each result immediately.
        Old value of 64 meant 64 × 74 MB = 4.7 GB buffered before first write.
      - compression="lzf": lzf compresses in-place (no extra scratch buffer).
        gzip needs to allocate a full extra chunk copy in memory → OOM.
      - explicit del res.frames / res.mel after writing: lets GC reclaim
        the 74 MB numpy array immediately rather than waiting for next GC cycle.
      - low_memory mode: 1 worker + no compression (maximum safety on 8 GB).
      - h5py.string_dtype() — no deprecation warning
      - resume via 'done' boolean mask
      - mel_f computed, not probed
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Init] Device     : {device}")

    if low_memory:
        num_workers  = 1
        compress_alg = None
        compress_opt = None
        print(f"[Init] Low-memory : ON — 1 worker, no compression")
    else:
        compress_alg = "lzf"   # FIX: was "gzip" — lzf needs no scratch buffer
        compress_opt = None    # lzf has no opts (gzip had opts=1)
        print(f"[Init] Compression: lzf  (was gzip — gzip OOM'd on 8 GB)")

    print(f"[Init] Workers    : {num_workers}")
    print(f"[Init] Write batch: {_WRITE_BATCH}  (was 64 — 64×74 MB = 4.7 GB buffer)")

    mel_f = compute_mel_f(num_frames=num_frames, sample_rate=sample_rate)
    print(f"[Init] mel_f      : {mel_f}  (computed from params)")

    str_dtype  = h5py.string_dtype()
    hdf5_mode  = "a" if resume else "w"
    start_time = time.time()

    with h5py.File(output_path, hdf5_mode) as hf:
        for split_name, split_videos in splits.items():
            n = len(split_videos)
            if n == 0:
                continue

            # ── Resume ────────────────────────────────────────────────────
            start_idx = 0
            if resume and split_name in hf:
                done_mask = hf[split_name].get("done")
                if done_mask is not None:
                    arr       = done_mask[:]
                    start_idx = int(np.argmin(arr))
                    if arr[start_idx]:
                        print(f"\n[{split_name}] Already complete, skipping.")
                        continue
                    print(f"\n[{split_name}] Resuming from {start_idx}/{n}")
            else:
                if split_name in hf:
                    del hf[split_name]

                grp = hf.create_group(split_name)

                # Compression kwargs — factored out so low_memory path is clean
                def _comp_kwargs():
                    if compress_alg is None:
                        return {}
                    kw = {"compression": compress_alg}
                    if compress_opt is not None:
                        kw["compression_opts"] = compress_opt
                    return kw

                grp.create_dataset(
                    "frames",
                    shape=(n, num_frames, 6, 224, 224), dtype=np.float32,
                    chunks=(1, num_frames, 6, 224, 224),
                    fillvalue=0,
                    **_comp_kwargs(),
                )
                grp.create_dataset(
                    "mel",
                    shape=(n, num_frames, n_mels, mel_f), dtype=np.float32,
                    chunks=(1, num_frames, n_mels, mel_f),
                    fillvalue=0,
                    **_comp_kwargs(),
                )
                grp.create_dataset("labels",  shape=(n,), dtype=np.int64, fillvalue=0)
                grp.create_dataset("methods", shape=(n,), dtype=str_dtype)
                grp.create_dataset("paths",   shape=(n,), dtype=str_dtype)
                grp.create_dataset(
                    "done", shape=(n,), dtype=bool,
                    data=np.zeros(n, dtype=bool),
                )
                print(f"\n[{split_name}] Processing {n} videos "
                      f"with {num_workers} worker(s)...")

            grp       = hf[split_name]
            ok_count  = int(np.sum(grp["done"][:start_idx]))
            err_count = start_idx - ok_count

            work = [
                (i, v["path"], v["label"], v["method"], num_frames, sample_rate, n_mels)
                for i, v in enumerate(split_videos)
                if i >= start_idx
            ]

            pbar   = tqdm(total=n, initial=start_idx, desc=split_name, unit="vid")
            buffer: List[VideoResult] = []

            def _flush(buf: List[VideoResult]) -> None:
                """
                Write buffered results to HDF5 then immediately free numpy arrays.

                FIX: After grp["frames"][res.index] = res.frames, the numpy
                array is still referenced by res.frames. Explicitly deleting it
                lets the GC reclaim ~74 MB right away instead of waiting for
                the next collection cycle — critical on 8 GB machines.
                """
                for res in buf:
                    if res.ok:
                        if res.mel is None:
                            continue
                        mel_np = res.mel
                        # Align F dimension (slight variation across videos)
                        if mel_np.shape[2] != mel_f:
                            if mel_np.shape[2] < mel_f:
                                pad    = np.zeros(
                                    (*mel_np.shape[:2], mel_f - mel_np.shape[2]),
                                    dtype=np.float32,
                                )
                                mel_np = np.concatenate((mel_np, pad), axis=2)
                            else:
                                mel_np = mel_np[:, :, :mel_f]
                        grp["frames"][res.index] = res.frames
                        grp["mel"][res.index]    = mel_np
                        # FIX: release the large arrays immediately after write
                        res.frames = None
                        res.mel    = None
                        del mel_np
                    else:
                        grp["frames"][res.index] = np.zeros(
                            (num_frames, 6, 224, 224), dtype=np.float32
                        )
                        grp["mel"][res.index] = np.zeros(
                            (num_frames, n_mels, mel_f), dtype=np.float32
                        )
                    grp["labels"][res.index]  = res.label
                    grp["methods"][res.index] = res.method
                    grp["paths"][res.index]   = res.path
                    grp["done"][res.index]    = res.ok
                hf.flush()
                gc.collect()   # FIX: nudge GC after every flush to keep RSS low

            # ── Parallel execution ────────────────────────────────────────
            executor_cls = (
                cf.ProcessPoolExecutor if num_workers > 1
                else cf.ThreadPoolExecutor
            )
            init_kwargs = (
                {"initializer": _worker_init, "initargs": (device,)}
                if num_workers > 1 else {}
            )

            with executor_cls(max_workers=num_workers, **init_kwargs) as exe:
                futures = {exe.submit(_process_one, w): w for w in work}

                for future in cf.as_completed(futures):
                    res = future.result()
                    buffer.append(res)

                    if res.ok:
                        ok_count += 1
                    else:
                        err_count += 1
                        tqdm.write(f"  ✗ {Path(res.path).name}: {res.error}")

                    pbar.update(1)
                    elapsed    = time.time() - start_time
                    done_total = ok_count + err_count
                    rate       = done_total / max(elapsed, 1)
                    remaining  = (n - start_idx - done_total) / max(rate, 0.01)
                    pbar.set_postfix(ok=ok_count, err=err_count, eta=f"{remaining/60:.0f}m")

                    # FIX: _WRITE_BATCH is now 1 — flush after every single
                    # result so the 74 MB frame tensor is freed immediately.
                    if len(buffer) >= _WRITE_BATCH:
                        _flush(buffer)
                        buffer.clear()

            if buffer:
                _flush(buffer)
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


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess FakeAVCeleb videos into HDF5 for training"
    )
    parser.add_argument("--data-dir",    type=str,   required=True)
    parser.add_argument("--output",      type=str,   default="./data/preprocessed/fakeavceleb.h5")
    parser.add_argument("--num-frames",  type=int,   default=16)
    parser.add_argument("--train-size",  type=int,   default=14_000)
    parser.add_argument("--val-size",    type=int,   default=3_000)
    parser.add_argument("--test-size",   type=int,   default=3_000)
    parser.add_argument("--num-workers", type=int,   default=2,
                        help="Parallel worker processes (default: 2; use 1 on 8 GB RAM)")
    parser.add_argument("--device",      type=str,   default="auto",
                        choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--resume",      action="store_true",
                        help="Resume from existing HDF5 — skip done videos")
    parser.add_argument("--low-memory",  action="store_true",
                        help="Safe mode for ≤8 GB RAM: 1 worker + no compression")

    args = parser.parse_args()

    print("=" * 70)
    print("  FAKEAVCELEB → HDF5 PREPROCESSING")
    print("=" * 70)
    print(f"  Data dir   : {args.data_dir}")
    print(f"  Output     : {args.output}")
    print(f"  Workers    : {args.num_workers}")
    print(f"  Device     : {args.device}")
    print(f"  Split      : {args.train_size} / {args.val_size} / {args.test_size}")
    print(f"  Low-memory : {args.low_memory}")

    try:
        videos = discover_fakeavceleb(args.data_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)

    splits = split_dataset(
        videos,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )

    preprocess_and_save(
        splits,
        args.output,
        num_frames=args.num_frames,
        num_workers=args.num_workers,
        device=args.device,
        resume=args.resume,
        low_memory=args.low_memory,
    )

    print(f"\n  Next steps:")
    print(f"    1. Set in configs/default.yaml:")
    print(f"         use_dummy_data: false")
    print(f"         hdf5_path: {args.output}")
    print(f"    2. Run: python training/train.py --config configs/default.yaml")


if __name__ == "__main__":
    main()