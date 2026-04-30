from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float(fpr[idx]), float(thresholds[idx])


def compute_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    preds = (scores >= threshold).astype(np.int64)
    accuracy = accuracy_score(labels, preds)

    try:
        auroc = roc_auc_score(labels, scores)
    except ValueError:
        auroc = float("nan")

    try:
        eer, eer_threshold = compute_eer(labels, scores)
    except ValueError:
        eer = float("nan")
        eer_threshold = float("nan")

    return {
        "accuracy": float(accuracy),
        "auroc": float(auroc),
        "eer": float(eer),
        "eer_threshold": float(eer_threshold),
    }


def evaluate_model(
    model: torch.nn.Module, loader: DataLoader, device: str, threshold: float = 0.5
) -> Dict[str, float]:
    model.eval()
    all_scores: List[float] = []
    all_labels: List[float] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="eval", leave=False):
            video = batch["video"].to(device)
            audio = batch["audio"].to(device)
            labels = batch["label"].to(device)

            logits = model(video, audio)
            scores = torch.sigmoid(logits)

            all_scores.extend(scores.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    labels_np = np.asarray(all_labels, dtype=np.float32)
    scores_np = np.asarray(all_scores, dtype=np.float32)
    return compute_metrics(labels_np, scores_np, threshold=threshold)
