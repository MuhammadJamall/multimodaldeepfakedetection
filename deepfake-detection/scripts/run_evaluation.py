#!/usr/bin/env python3
"""
run_evaluation.py
-----------------
Standalone evaluation script for trained DeepfakeDetector models.

Per CONTEXT.md §5:
  - In-distribution: FakeAVCeleb test split (AUROC >= 0.93, Acc >= 88%, EER < 8%)
  - Cross-dataset: DFDC zero-shot (AUROC drop < 5%)
  - Per-method analysis: accuracy by forgery method (FaceSwap, Wav2Lip, etc.)

Usage:
    # Evaluate on FakeAVCeleb test split
    python scripts/run_evaluation.py \\
        --checkpoint checkpoints/best_auroc.pt \\
        --config configs/default.yaml

    # Cross-dataset evaluation on DFDC
    python scripts/run_evaluation.py \\
        --checkpoint checkpoints/best_auroc.pt \\
        --hdf5-path data/preprocessed/dfdc.h5 \\
        --split test

    # Quick evaluation with dummy data (for testing the script itself)
    python scripts/run_evaluation.py --dummy
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.detector import DeepfakeDetector
from evaluation.evaluate import compute_metrics, compute_eer
from data.dataset import BasicDataset

try:
    import h5py
except ImportError:
    h5py = None


# ── Model Loading ────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device) -> tuple:
    """Load model from checkpoint."""
    print(f"[Model] Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    cfg = ckpt.get("config", {})
    mcfg = cfg.get("model", {})

    model = DeepfakeDetector(
        vit_model=mcfg.get("vit_model", "google/vit-base-patch16-224-in21k"),
        vit_hidden_dim=mcfg.get("vit_hidden_dim", 512),
        audio_hidden_dim=mcfg.get("audio_hidden_dim", 512),
        num_heads=mcfg.get("num_heads", 8),
        ffn_hidden_dim=mcfg.get("ffn_hidden_dim", 256),
        dropout=mcfg.get("dropout", 0.3),
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()

    epoch = ckpt.get("epoch", "?")
    train_metrics = ckpt.get("metrics", {})
    print(f"[Model] Loaded from epoch {epoch}")
    if train_metrics:
        print(f"[Model] Training metrics: {train_metrics}")

    return model, cfg


def build_model_fresh(device: torch.device) -> DeepfakeDetector:
    """Build a fresh model (for dummy evaluation)."""
    model = DeepfakeDetector().to(device)
    model.eval()
    return model


# ── Evaluation Functions ─────────────────────────────────────────────────────

def evaluate_split(
    model: DeepfakeDetector,
    loader: DataLoader,
    device: torch.device,
    desc: str = "eval",
) -> Dict:
    """
    Run evaluation on a DataLoader.

    Returns dict with overall metrics + per-sample predictions.
    """
    model.eval()
    all_probs: List[float] = []
    all_labels: List[float] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            frames = batch["frames"].to(device)
            mel = batch["mel"].to(device)
            labels = batch["label"].to(device)

            prob = model(frames, mel)  # (B, 1) sigmoid output

            all_probs.extend(prob.cpu().numpy().flatten().tolist())
            all_labels.extend(labels.cpu().numpy().flatten().tolist())

    labels_np = np.array(all_labels, dtype=np.float32)
    scores_np = np.array(all_probs, dtype=np.float32)

    metrics = compute_metrics(labels_np, scores_np, threshold=0.5)
    metrics["num_samples"] = len(labels_np)
    metrics["num_real"] = int((labels_np == 0).sum())
    metrics["num_fake"] = int((labels_np == 1).sum())

    return metrics


def evaluate_per_method(
    model: DeepfakeDetector,
    hdf5_path: str,
    split: str,
    device: torch.device,
    batch_size: int = 32,
) -> Dict[str, Dict]:
    """
    Per-method accuracy breakdown (FaceSwap, Wav2Lip, etc.).
    Per CONTEXT.md §5.2: "Report accuracy by deepfake method for generalization proof"

    Requires HDF5 file with 'methods' dataset.
    """
    if h5py is None:
        print("[WARNING] h5py not available — skipping per-method analysis")
        return {}

    with h5py.File(hdf5_path, "r") as f:
        if "methods" not in f[split]:
            print("[WARNING] No 'methods' dataset in HDF5 — skipping per-method analysis")
            return {}

        methods = [m.decode() if isinstance(m, bytes) else m for m in f[split]["methods"][:]]
        labels = f[split]["labels"][:]

    # Get all predictions first
    dataset = BasicDataset(
        use_dummy_data=False,
        hdf5_path=hdf5_path,
        split=split,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="per-method", leave=False):
            prob = model(batch["frames"].to(device), batch["mel"].to(device))
            all_probs.extend(prob.cpu().numpy().flatten().tolist())

    probs_np = np.array(all_probs, dtype=np.float32)
    labels_np = np.array(labels, dtype=np.float32)

    # Group by method
    unique_methods = sorted(set(methods))
    method_metrics = {}

    for method in unique_methods:
        mask = np.array([m == method for m in methods])
        if mask.sum() == 0:
            continue

        m_probs = probs_np[mask]
        m_labels = labels_np[mask]

        preds = (m_probs >= 0.5).astype(float)
        acc = (preds == m_labels).mean()

        try:
            from sklearn.metrics import roc_auc_score
            auroc = roc_auc_score(m_labels, m_probs)
        except (ValueError, ImportError):
            auroc = float("nan")

        method_metrics[method] = {
            "accuracy": float(acc),
            "auroc": float(auroc),
            "num_samples": int(mask.sum()),
        }

    return method_metrics


# ── Result Formatting ────────────────────────────────────────────────────────

def print_results(
    metrics: Dict,
    method_metrics: Dict = None,
    dataset_name: str = "FakeAVCeleb",
    targets: Dict = None,
):
    """Pretty-print evaluation results with target comparisons."""
    if targets is None:
        targets = {
            "auroc": 0.93,
            "accuracy": 0.88,
            "eer": 0.08,
        }

    print("\n" + "=" * 70)
    print(f"  EVALUATION RESULTS — {dataset_name}")
    print("=" * 70)

    print(f"\n  Samples: {metrics['num_samples']} "
          f"(Real: {metrics['num_real']}, Fake: {metrics['num_fake']})")

    print(f"\n  {'Metric':<20} {'Value':>10} {'Target':>10} {'Status':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}")

    # AUROC
    auroc = metrics["auroc"]
    auroc_target = targets.get("auroc", 0.93)
    auroc_status = "✅ PASS" if auroc >= auroc_target else "❌ FAIL"
    print(f"  {'AUROC':<20} {auroc:>10.4f} {f'>= {auroc_target}':>10} {auroc_status:>10}")

    # Accuracy
    acc = metrics["accuracy"]
    acc_target = targets.get("accuracy", 0.88)
    acc_status = "✅ PASS" if acc >= acc_target else "❌ FAIL"
    print(f"  {'Accuracy':<20} {acc:>10.4f} {f'>= {acc_target}':>10} {acc_status:>10}")

    # EER
    eer = metrics["eer"]
    eer_target = targets.get("eer", 0.08)
    eer_status = "✅ PASS" if eer <= eer_target else "❌ FAIL"
    print(f"  {'EER':<20} {eer:>10.4f} {f'< {eer_target}':>10} {eer_status:>10}")

    # EER threshold
    print(f"  {'EER Threshold':<20} {metrics.get('eer_threshold', 0):>10.4f}")

    # Per-method breakdown
    if method_metrics:
        print(f"\n  {'Method':<25} {'Accuracy':>10} {'AUROC':>10} {'Samples':>10}")
        print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")
        for method, m in sorted(method_metrics.items()):
            print(f"  {method:<25} {m['accuracy']:>10.4f} {m['auroc']:>10.4f} {m['num_samples']:>10}")

    print("\n" + "=" * 70)


def save_results(
    metrics: Dict,
    method_metrics: Dict,
    output_path: str,
    dataset_name: str = "FakeAVCeleb",
):
    """Save results to JSON file."""
    results = {
        "dataset": dataset_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "overall": metrics,
        "per_method": method_metrics,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Results] Saved to {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate DeepfakeDetector")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config YAML (used if checkpoint has no config)")
    parser.add_argument("--hdf5-path", type=str, default=None,
                        help="Path to HDF5 dataset (overrides config)")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split to evaluate (default: test)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--dataset-name", type=str, default="FakeAVCeleb",
                        help="Dataset name for reporting")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save JSON results")
    parser.add_argument("--dummy", action="store_true",
                        help="Run with dummy data (for testing the script)")
    parser.add_argument("--per-method", action="store_true",
                        help="Compute per-method accuracy breakdown")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # ── Load model ────────────────────────────────────────────────────────
    if args.checkpoint and os.path.exists(args.checkpoint):
        model, cfg = load_model(args.checkpoint, device)
    elif args.dummy:
        print("[Model] Using fresh model (dummy mode)")
        model = build_model_fresh(device)
        cfg = {}
    else:
        print("ERROR: Provide --checkpoint or use --dummy mode")
        sys.exit(1)

    # ── Build data loader ─────────────────────────────────────────────────
    if args.dummy:
        dataset = BasicDataset(
            num_samples=32,
            use_dummy_data=True,
            split=args.split,
        )
    elif args.hdf5_path:
        dataset = BasicDataset(
            use_dummy_data=False,
            hdf5_path=args.hdf5_path,
            split=args.split,
        )
    else:
        dcfg = cfg.get("data", {})
        hdf5_path = dcfg.get("hdf5_path", None)
        if hdf5_path and os.path.exists(hdf5_path):
            dataset = BasicDataset(
                use_dummy_data=False,
                hdf5_path=hdf5_path,
                split=args.split,
            )
        else:
            print("[WARNING] No HDF5 data found — falling back to dummy data")
            dataset = BasicDataset(
                num_samples=32,
                use_dummy_data=True,
                split=args.split,
            )

    loader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=0,
    )

    # ── Evaluate ──────────────────────────────────────────────────────────
    print(f"\n[Eval] Running on {len(dataset)} samples ({args.split} split)...")
    metrics = evaluate_split(model, loader, device, desc=args.split)

    # ── Per-method analysis ───────────────────────────────────────────────
    method_metrics = {}
    if args.per_method and args.hdf5_path:
        print("\n[Eval] Computing per-method breakdown...")
        method_metrics = evaluate_per_method(
            model, args.hdf5_path, args.split,
            device, args.batch_size,
        )

    # ── Print results ─────────────────────────────────────────────────────
    targets = {"auroc": 0.93, "accuracy": 0.88, "eer": 0.08}
    if args.dataset_name == "DFDC":
        # Cross-dataset: just report, no hard targets
        targets = {"auroc": 0.0, "accuracy": 0.0, "eer": 1.0}

    print_results(metrics, method_metrics, args.dataset_name, targets)

    # ── Save results ──────────────────────────────────────────────────────
    if args.output:
        save_results(metrics, method_metrics, args.output, args.dataset_name)
    elif not args.dummy:
        # Auto-save
        output = f"results/eval_{args.dataset_name}_{args.split}_{int(time.time())}.json"
        save_results(metrics, method_metrics, output, args.dataset_name)

    return 0


if __name__ == "__main__":
    sys.exit(main())
