from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from data.dataset import DeepfakeDataset
from evaluation.evaluate import evaluate_model
from models.detector import DeepfakeDetector
from training.losses import combined_loss
from training.scheduler import build_scheduler


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_value(raw: str):
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def apply_overrides(cfg: Dict[str, Any], overrides: List[str]) -> None:
    for item in overrides:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        parts = key.split(".")
        target = cfg
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                target[part] = {}
            target = target[part]
        target[parts[-1]] = parse_value(value)


def build_model(cfg: Dict[str, Any]) -> DeepfakeDetector:
    visual_cfg = cfg["model"]["visual"]
    audio_cfg = cfg["model"]["audio"]
    attention_cfg = cfg["model"]["attention"]
    fusion_cfg = cfg["model"]["fusion"]

    return DeepfakeDetector(
        visual_name=visual_cfg["name"],
        visual_pretrained=visual_cfg["pretrained"],
        visual_frozen=visual_cfg["frozen"],
        audio_in_channels=audio_cfg["in_channels"],
        audio_out_dim=audio_cfg["out_dim"],
        attention_embed_dim=attention_cfg["embed_dim"],
        attention_heads=attention_cfg["num_heads"],
        attention_dropout=attention_cfg["dropout"],
        fusion_hidden_dim=fusion_cfg["hidden_dim"],
        fusion_dropout=fusion_cfg["dropout"],
    )


def build_dataloaders(cfg: Dict[str, Any], device: str):
    data_cfg = cfg["data"]

    train_dataset = DeepfakeDataset(
        data_cfg["manifest"],
        split="train",
        image_size=data_cfg["image_size"],
        clip_len=data_cfg["clip_len"],
        sample_rate=data_cfg["sample_rate"],
        n_mels=data_cfg["n_mels"],
        hop_length=data_cfg["hop_length"],
        win_length=data_cfg["win_length"],
        max_audio_seconds=data_cfg["max_audio_seconds"],
        use_face_crop=data_cfg["use_face_crop"],
        device=device,
    )

    val_dataset = DeepfakeDataset(
        data_cfg["manifest"],
        split="val",
        image_size=data_cfg["image_size"],
        clip_len=data_cfg["clip_len"],
        sample_rate=data_cfg["sample_rate"],
        n_mels=data_cfg["n_mels"],
        hop_length=data_cfg["hop_length"],
        win_length=data_cfg["win_length"],
        max_audio_seconds=data_cfg["max_audio_seconds"],
        use_face_crop=data_cfg["use_face_crop"],
        device=device,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    loss_cfg: Dict[str, Any],
    grad_clip: float,
) -> float:
    model.train()
    running_loss = 0.0

    for batch in tqdm(loader, desc="train", leave=False):
        video = batch["video"].to(device)
        audio = batch["audio"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(video, audio)
        loss = combined_loss(
            logits,
            labels,
            bce_weight=loss_cfg["bce_weight"],
            lse_weight=loss_cfg["lse_weight"],
            lse_margin=loss_cfg["lse_margin"],
        )
        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        running_loss += loss.item()

    return running_loss / max(len(loader), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Deepfake detection training")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="runs/exp")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    with open(args.config, "r") as handle:
        cfg = yaml.safe_load(handle)

    if args.override:
        apply_overrides(cfg, args.override)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    set_seed(cfg["project"]["seed"])

    train_loader, val_loader = build_dataloaders(cfg, device)
    model = build_model(cfg).to(device)

    train_cfg = cfg["train"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )
    scheduler = build_scheduler(optimizer, train_cfg["epochs"])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "config.yaml").open("w") as handle:
        yaml.safe_dump(cfg, handle)

    metrics_path = output_dir / "metrics.jsonl"

    for epoch in range(1, train_cfg["epochs"] + 1):
        loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            cfg["loss"],
            train_cfg["grad_clip"],
        )
        scheduler.step()

        if epoch % train_cfg["eval_interval"] == 0:
            metrics = evaluate_model(
                model,
                val_loader,
                device=device,
                threshold=cfg["eval"]["threshold"],
            )
            metrics["epoch"] = epoch
            metrics["train_loss"] = loss
            with metrics_path.open("a") as handle:
                handle.write(json.dumps(metrics) + "\n")

        torch.save(model.state_dict(), output_dir / "last.pt")

    print(f"Training complete. Outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
