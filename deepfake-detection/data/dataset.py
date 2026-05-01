"""
PyTorch Dataset Class for Multimodal Deepfake Detection
==========================================================

This module provides a flexible Dataset class that can work with:
1. Dummy data (for testing during development)
2. HDF5 preprocessed data (for real training)

Supports batching with WeightedRandomSampler for balanced real/fake sampling.

Key names returned by __getitem__:
    'frames' : visual tensor  (T, 6, 224, 224)
    'mel'    : audio tensor   (T, 80, F)
    'label'  : scalar label   (0=real, 1=fake)
"""

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import Dict, Optional, Tuple
try:
    import h5py  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - handled at runtime when needed
    h5py = None
import numpy as np
from pathlib import Path

# Support running both as `python data/dataset.py` and as `from data.dataset import ...`
try:
    from data.dummy_dataset import generate_dummy_batch
    from data.augmentation import apply_augmentation
except ImportError:
    from dummy_dataset import generate_dummy_batch
    from augmentation import apply_augmentation


class BasicDataset(Dataset):
    """
    Basic Dataset class supporting both dummy and real data.

    This class can:
    - Generate dummy data on-the-fly (for testing)
    - Load data from HDF5 files (for real training)
    - Support balanced sampling (1:1 real/fake ratio)

    Attributes:
        num_samples: Number of samples in the dataset
        batch_size: Batch size for dummy data generation
        use_dummy_data: Whether to use dummy or real data
        hdf5_path: Path to HDF5 file (if using real data)
        device: Device to load tensors on
    """

    def __init__(
        self,
        num_samples: int = 100,
        batch_size: int = 32,
        use_dummy_data: bool = True,
        hdf5_path: Optional[Path] = None,
        device: str = "cpu",
        split: str = "train",
        augmentation_cfg: Optional[Dict] = None,
    ):
        """
        Initialize the dataset.

        Args:
            num_samples: Number of samples in the dataset
            batch_size: Batch size for dummy data generation
            use_dummy_data: If True, generate dummy data; if False, load from HDF5
            hdf5_path: Path to HDF5 file (required if use_dummy_data=False)
            device: Device to load tensors on ("cpu" or "cuda")
            split: Data split ("train", "val", or "test")

        Raises:
            ValueError: If use_dummy_data=False but hdf5_path is not provided
            FileNotFoundError: If hdf5_path doesn't exist
        """
        super().__init__()
        self.num_samples = num_samples
        self.batch_size = min(batch_size, num_samples)
        self.use_dummy_data = use_dummy_data
        self.device = device
        self.split = split
        self.hdf5_path: Optional[Path] = None
        self.augmentation_cfg = augmentation_cfg or {}
        self.is_training = (split == "train")

        if use_dummy_data:
            self.data = None
            print(f"[Dataset] Initialized with DUMMY DATA (num_samples={num_samples})")
        else:
            if hdf5_path is None:
                raise ValueError(
                    "hdf5_path must be provided when use_dummy_data=False"
                )
            
            if h5py is None:
                raise ModuleNotFoundError(
                    "h5py is required for HDF5 datasets. Install it or enable dummy data."
                )

            if isinstance(hdf5_path, str):
                hdf5_path = Path(hdf5_path)
            if not isinstance(hdf5_path, Path):
                raise ValueError("hdf5_path must be a string or Path when use_dummy_data=False")
            if not hdf5_path.exists():
                raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")

            self.hdf5_path = hdf5_path
            
            # Load HDF5 file
            with h5py.File(hdf5_path, 'r') as f:
                self.num_samples = len(f[split]['frames'])
                print(f"[Dataset] Loaded HDF5 from {hdf5_path}")
                print(f"[Dataset] Split: {split}, Samples: {self.num_samples}")

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Dictionary containing:
            - 'frames': Visual tensor  (T, 6, 224, 224)
            - 'mel':    Audio tensor   (T, 80, F)
            - 'label':  Label float    (0.0=real, 1.0=fake) shaped (1,)

        Raises:
            IndexError: If idx is out of bounds
        """
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(
                f"Index {idx} out of bounds for dataset of size {self.num_samples}"
            )

        if self.use_dummy_data:
            # Generate a single dummy sample
            batch = generate_dummy_batch(batch_size=1, seed=idx)
            frames = batch['frames'][0]
            mel    = batch['mel'][0]
            label  = batch['labels'][0].float().unsqueeze(0)
        else:
            # Load from HDF5
            # Expected HDF5 structure:
            #   /{split}/frames  → (N, T, 6, 224, 224)
            #   /{split}/mel     → (N, T, 80, F)
            #   /{split}/labels  → (N,)
            if h5py is None or self.hdf5_path is None:
                raise RuntimeError("HDF5 dataset not available. Check h5py and hdf5_path.")
            with h5py.File(self.hdf5_path, 'r') as f:
                frames = torch.from_numpy(f[self.split]['frames'][idx]).float()
                mel    = torch.from_numpy(f[self.split]['mel'][idx]).float()
                label  = torch.tensor(f[self.split]['labels'][idx]).float().unsqueeze(0)

        # Apply augmentation (training only, 30% per-video probability)
        frames, mel = apply_augmentation(
            frames, mel,
            cfg=self.augmentation_cfg,
            is_training=self.is_training,
        )

        return {
            'frames': frames.to(self.device),
            'mel':    mel.to(self.device),
            'label':  label.to(self.device),
        }

    def get_balanced_sampler(self) -> WeightedRandomSampler:
        """
        Create a WeightedRandomSampler for balanced real/fake sampling.

        Returns:
            WeightedRandomSampler for 1:1 real/fake ratio

        Note:
            Only works with dummy data (labels are balanced by design).
            For HDF5 data, will require loading all labels first.
        """
        if self.use_dummy_data:
            # Generate all labels for sampling weights
            all_labels = []
            for i in range(self.num_samples):
                batch = generate_dummy_batch(batch_size=1, seed=i)
                all_labels.append(batch['labels'][0].item())
            labels = np.array(all_labels)
        else:
            # Load all labels from HDF5
            if h5py is None or self.hdf5_path is None:
                raise RuntimeError("HDF5 dataset not available. Check h5py and hdf5_path.")
            with h5py.File(self.hdf5_path, 'r') as f:
                labels = f[self.split]['labels'][:]

        # Calculate sampling weights (inverse frequency)
        unique_classes = np.unique(labels)
        weights = np.zeros(self.num_samples)
        
        for cls in unique_classes:
            cls_count = np.sum(labels == cls)
            weights[labels == cls] = 1.0 / cls_count

        return WeightedRandomSampler(
            weights=weights.tolist(),
            num_samples=self.num_samples,
            replacement=True
        )


def create_dummy_dataloader(
    num_samples: int = 100,
    batch_size: int = 32,
    num_workers: int = 0,
    shuffle: bool = True,
    device: str = "cpu",
    balanced_sampling: bool = True,
) -> DataLoader:
    """
    Create a DataLoader with dummy data for testing.

    Args:
        num_samples: Number of samples to generate
        batch_size: Batch size
        num_workers: Number of worker processes
        shuffle: Whether to shuffle data
        device: Device to load data on
        balanced_sampling: If True, use WeightedRandomSampler for 1:1 ratio

    Returns:
        DataLoader configured for training
    """
    dataset = BasicDataset(
        num_samples=num_samples,
        batch_size=batch_size,
        use_dummy_data=True,
        device=device,
        split="train"
    )

    if balanced_sampling:
        sampler = dataset.get_balanced_sampler()
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

    return dataloader


def build_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders from config.

    Currently uses dummy data. When real HDF5 data is available,
    set data.use_dummy_data=False and provide data.hdf5_path in config.

    Args:
        cfg: Full config dict (from configs/default.yaml)

    Returns:
        (train_loader, val_loader) tuple of DataLoaders
    """
    dcfg = cfg.get("data", {})
    tcfg = cfg.get("training", {})

    use_dummy   = dcfg.get("use_dummy_data", True)
    hdf5_path   = dcfg.get("hdf5_path", None)
    batch_size  = tcfg.get("batch_size", 32)
    num_workers = dcfg.get("num_workers", 4)
    aug_cfg = cfg.get("augmentation", {})
    # Also pass compression_augmentation_prob from data section
    aug_cfg.setdefault("compression_augmentation_prob",
                       dcfg.get("compression_augmentation_prob", 0.3))

    # ── Train loader ──────────────────────────────────────────────────────────
    train_dataset = BasicDataset(
        num_samples=dcfg.get("train_split", 14000) if use_dummy else 0,
        batch_size=batch_size,
        use_dummy_data=use_dummy,
        hdf5_path=hdf5_path,
        split="train",
        augmentation_cfg=aug_cfg,
    )

    train_sampler = train_dataset.get_balanced_sampler()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=dcfg.get("pin_memory", True),
    )

    # ── Val loader ────────────────────────────────────────────────────────────
    val_dataset = BasicDataset(
        num_samples=dcfg.get("val_split", 3000) if use_dummy else 0,
        batch_size=batch_size,
        use_dummy_data=use_dummy,
        hdf5_path=hdf5_path,
        split="val",
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=dcfg.get("pin_memory", True),
    )

    print(f"[DataLoaders] Train: {len(train_dataset)} samples, "
          f"Val: {len(val_dataset)} samples, "
          f"Batch size: {batch_size}")

    return train_loader, val_loader


if __name__ == "__main__":
    """Test dataset and dataloader"""
    print("=" * 80)
    print("DATASET CLASS - PHASE 1 TEST")
    print("=" * 80)

    # Test 1: Create dataset with dummy data
    print("\n[Test 1] Creating dataset with dummy data...")
    dataset = BasicDataset(num_samples=100, batch_size=32, use_dummy_data=True)
    print(f"✅ Dataset created with {len(dataset)} samples")

    # Test 2: Get single sample
    print("\n[Test 2] Getting single sample...")
    sample = dataset[0]
    print(f"    Visual shape: {sample['frames'].shape}")
    print(f"    Audio shape:  {sample['mel'].shape}")
    print(f"    Label:        {sample['label'].item()}")

    # Test 3: Create dataloader with balanced sampling
    print("\n[Test 3] Creating dataloader with balanced sampling...")
    dataloader = create_dummy_dataloader(
        num_samples=64,
        batch_size=32,
        balanced_sampling=True
    )
    print(f"✅ DataLoader created with {len(dataloader)} batches")

    # Test 4: Iterate through dataloader
    print("\n[Test 4] Iterating through dataloader...")
    for batch_idx, batch in enumerate(dataloader):
        print(f"    Batch {batch_idx}:")
        print(f"      Frames: {batch['frames'].shape}")
        print(f"      Mel:    {batch['mel'].shape}")
        print(f"      Labels: {batch['label'].shape}")
        unique, counts = torch.unique(batch['label'], return_counts=True)
        for label, count in zip(unique, counts):
            print(f"        Label {label.item()}: {count.item()} samples")
        break  # Just show first batch

    # Test 5: build_dataloaders with dummy config
    print("\n[Test 5] Testing build_dataloaders...")
    dummy_cfg = {
        "data": {
            "use_dummy_data": True,
            "train_split": 64,
            "val_split": 32,
            "num_workers": 0,
            "pin_memory": False,
        },
        "training": {"batch_size": 16},
    }
    train_loader, val_loader = build_dataloaders(dummy_cfg)
    print(f"✅ Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    print("\n" + "=" * 80)
    print("✅ DATASET CLASS TEST PASSED")
    print("=" * 80)