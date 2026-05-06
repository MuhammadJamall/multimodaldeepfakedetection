"""
PyTorch Dataset Class for Multimodal Deepfake Detection
==========================================================

This module loads real preprocessed data from HDF5 files for training
and validation of the deepfake detection model.

Expected HDF5 structure:
    /{split}/frames  → (N, T, 6, 224, 224)
    /{split}/mel     → (N, T, 80, F)
    /{split}/labels  → (N,)

Key names returned by __getitem__:
    'frames' : visual tensor  (T, 6, 224, 224)
    'mel'    : audio tensor   (T, 80, F)
    'label'  : scalar label   (0=real, 1=fake)
"""

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import Dict, Optional, Tuple
import h5py
import numpy as np
from pathlib import Path

# Lazy import for augmentation
apply_augmentation = None

def _ensure_augmentation():
    global apply_augmentation
    if apply_augmentation is not None:
        return
    try:
        from data.augmentation import apply_augmentation as _fn
    except ImportError:
        try:
            from augmentation import apply_augmentation as _fn
        except ImportError:
            # Fallback: no augmentation available, return inputs unchanged
            def _fn(frames, mel, cfg=None, is_training=True):
                return frames, mel
    apply_augmentation = _fn


class BasicDataset(Dataset):
    """
    Dataset class for loading real deepfake detection data from HDF5 files.

    Attributes:
        hdf5_path : Path to the HDF5 file
        split     : Data split ("train", "val", or "test")
        device    : Device to load tensors on
    """

    def __init__(
        self,
        hdf5_path: Path | str,
        split: str = "train",
        device: str = "cpu",
        augmentation_cfg: Optional[Dict] = None,
    ):
        """
        Initialize the dataset.

        Args:
            hdf5_path       : Path to the HDF5 file containing preprocessed data
            split           : Data split — "train", "val", or "test"
            device          : Device to load tensors on ("cpu" or "cuda")
            augmentation_cfg: Optional augmentation settings dict

        Raises:
            FileNotFoundError : If hdf5_path does not exist
            ValueError        : If hdf5_path is not a valid path
            KeyError          : If the requested split is not found in the HDF5 file
        """
        super().__init__()

        if isinstance(hdf5_path, str):
            hdf5_path = Path(hdf5_path)
        if not isinstance(hdf5_path, Path):
            raise ValueError("hdf5_path must be a string or Path object")
        if not hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

        self.hdf5_path = hdf5_path
        self.split = split
        self.device = device
        self.augmentation_cfg = augmentation_cfg or {}
        self.is_training = (split == "train")

        # Validate split and read dataset size
        with h5py.File(self.hdf5_path, "r") as f:
            if split not in f:
                available = list(f.keys())
                raise KeyError(
                    f"Split '{split}' not found in HDF5 file. "
                    f"Available splits: {available}"
                )
            self.num_samples = len(f[split]["frames"])

        print(f"[Dataset] Loaded HDF5 from {self.hdf5_path}")
        print(f"[Dataset] Split: {self.split} | Samples: {self.num_samples}")

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load a single sample from the HDF5 file.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Dictionary containing:
            - 'frames' : Visual tensor  (T, 6, 224, 224)
            - 'mel'    : Audio tensor   (T, 80, F)
            - 'label'  : Label float    (0.0=real, 1.0=fake) shaped (1,)

        Raises:
            IndexError: If idx is out of bounds
        """
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(
                f"Index {idx} out of bounds for dataset of size {self.num_samples}"
            )

        with h5py.File(self.hdf5_path, "r") as f:
            frames = torch.from_numpy(f[self.split]["frames"][idx]).float()
            mel    = torch.from_numpy(f[self.split]["mel"][idx]).float()
            label  = torch.tensor(f[self.split]["labels"][idx]).float().unsqueeze(0)

        # Apply augmentation (training only)
        _ensure_augmentation()
        frames, mel = apply_augmentation(
            frames, mel,
            cfg=self.augmentation_cfg,
            is_training=self.is_training,
        )

        return {
            "frames": frames.to(self.device),
            "mel":    mel.to(self.device),
            "label":  label.to(self.device),
        }

    def get_balanced_sampler(self) -> WeightedRandomSampler:
        """
        Create a WeightedRandomSampler for balanced real/fake sampling.

        Loads all labels from the HDF5 file and assigns inverse-frequency
        weights so that real and fake samples are seen equally during training.

        Returns:
            WeightedRandomSampler configured for 1:1 real/fake ratio
        """
        with h5py.File(self.hdf5_path, "r") as f:
            labels = f[self.split]["labels"][:]

        unique_classes = np.unique(labels)
        weights = np.zeros(self.num_samples)
        for cls in unique_classes:
            cls_count = np.sum(labels == cls)
            weights[labels == cls] = 1.0 / cls_count

        return WeightedRandomSampler(
            weights=weights.tolist(),
            num_samples=self.num_samples,
            replacement=True,
        )


def build_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders from a config dict.

    Args:
        cfg: Full config dict (from configs/default.yaml). Expected structure:
            data:
                hdf5_path  : str   — path to the HDF5 file
                num_workers: int   — number of DataLoader worker processes
                pin_memory : bool  — pin memory for faster GPU transfer
                persistent_workers: bool
            training:
                batch_size : int
            augmentation:
                ...augmentation settings...

    Returns:
        (train_loader, val_loader) tuple of DataLoaders

    Raises:
        ValueError: If hdf5_path is missing from config
    """
    dcfg = cfg.get("data", {})
    tcfg = cfg.get("training", {})

    hdf5_path = dcfg.get("hdf5_path")
    if not hdf5_path:
        raise ValueError("cfg['data']['hdf5_path'] must be set for real data loading.")

    batch_size  = tcfg.get("batch_size", 32)
    num_workers = dcfg.get("num_workers", 4)

    aug_cfg = cfg.get("augmentation", {})
    aug_cfg.setdefault(
        "compression_augmentation_prob",
        dcfg.get("compression_augmentation_prob", 0.3),
    )

    # ── Train loader ──────────────────────────────────────────────────────────
    train_dataset = BasicDataset(
        hdf5_path=hdf5_path,
        split="train",
        augmentation_cfg=aug_cfg,
    )

    train_loader_kw: Dict = dict(
        batch_size=batch_size,
        sampler=train_dataset.get_balanced_sampler(),
        num_workers=num_workers,
        pin_memory=dcfg.get("pin_memory", True),
    )
    if num_workers > 0 and dcfg.get("persistent_workers", False):
        train_loader_kw["persistent_workers"] = True

    train_loader = DataLoader(train_dataset, **train_loader_kw)

    # ── Val loader ────────────────────────────────────────────────────────────
    val_dataset = BasicDataset(
        hdf5_path=hdf5_path,
        split="val",
    )

    val_loader_kw: Dict = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=dcfg.get("pin_memory", True),
    )
    if num_workers > 0 and dcfg.get("persistent_workers", False):
        val_loader_kw["persistent_workers"] = True

    val_loader = DataLoader(val_dataset, **val_loader_kw)

    print(
        f"[DataLoaders] Train: {len(train_dataset)} samples | "
        f"Val: {len(val_dataset)} samples | "
        f"Batch size: {batch_size}"
    )

    return train_loader, val_loader


if __name__ == "__main__":
    """Quick sanity-check — point HDF5_PATH at a real file before running."""

    HDF5_PATH = "data/deepfake_dataset.h5"  # ← update this path

    print("=" * 80)
    print("DATASET CLASS - REAL DATA TEST")
    print("=" * 80)

    # Test 1: Load dataset
    print("\n[Test 1] Loading train dataset...")
    dataset = BasicDataset(hdf5_path=HDF5_PATH, split="train")
    print(f"✅ Dataset loaded with {len(dataset)} samples")

    # Test 2: Fetch a single sample
    print("\n[Test 2] Fetching sample at index 0...")
    sample = dataset[0]
    print(f"    Frames shape : {sample['frames'].shape}")
    print(f"    Mel shape    : {sample['mel'].shape}")
    print(f"    Label        : {sample['label'].item()}")

    # Test 3: Build dataloaders via config
    print("\n[Test 3] Building dataloaders from config...")
    cfg = {
        "data": {
            "hdf5_path": HDF5_PATH,
            "num_workers": 0,
            "pin_memory": False,
        },
        "training": {"batch_size": 16},
    }
    train_loader, val_loader = build_dataloaders(cfg)
    print(f"✅ Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # Test 4: Inspect one batch
    print("\n[Test 4] Inspecting first training batch...")
    batch = next(iter(train_loader))
    print(f"    Frames : {batch['frames'].shape}")
    print(f"    Mel    : {batch['mel'].shape}")
    print(f"    Labels : {batch['label'].shape}")
    unique, counts = torch.unique(batch["label"], return_counts=True)
    for lbl, cnt in zip(unique, counts):
        print(f"      Label {lbl.item():.0f}: {cnt.item()} samples")

    print("\n" + "=" * 80)
    print("✅ REAL DATA TEST PASSED")
    print("=" * 80)