"""
Training pipeline for PanNuke Faster R-CNN baseline.
Supports YAML configuration, learning rate scheduling, validation, and checkpointing.
"""

import argparse
import os
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import PanNukeCOCODataset
from model import build_faster_rcnn_model


def collate_fn(batch):
    return tuple(zip(*batch))


def train_one_epoch(model, optimizer, data_loader, device, epoch, log_interval=10):
    model.train()
    running_loss = 0.0

    pbar = tqdm(data_loader, desc=f"Epoch {epoch} [Train]")
    for step, (images, targets) in enumerate(pbar):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        running_loss += losses.item()
        if (step + 1) % log_interval == 0:
            avg_loss = running_loss / (step + 1)
            pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

    return running_loss / max(1, len(data_loader))


@torch.no_grad()
def evaluate_loss(model, data_loader, device):
    """Evaluate loss on validation set (runs model in train mode to get loss dict)."""
    model.train()
    total_loss = 0.0

    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()

    return total_loss / max(1, len(data_loader))


def main():
    parser = argparse.ArgumentParser(description="Train Faster R-CNN on PanNuke")
    parser.add_argument("--config", type=str, default="configs/baseline_config.yaml", help="Path to config file")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        fallback_cfg = Path(__file__).resolve().parent.parent / "configs" / "baseline_config.yaml"
        if fallback_cfg.exists():
            cfg_path = fallback_cfg

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Device selection
    device_name = cfg["training"].get("device", "cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(device_name)

    print(f"Using device: {device}")

    # Datasets
    train_dataset = PanNukeCOCODataset(
        images_dir=cfg["data"]["train_images_dir"],
        annotations_file=cfg["data"]["train_annotations"]
    )
    val_dataset = PanNukeCOCODataset(
        images_dir=cfg["data"]["val_images_dir"],
        annotations_file=cfg["data"]["val_annotations"]
    )

    batch_size = cfg["training"].get("batch_size", 4)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Model
    model = build_faster_rcnn_model(
        num_classes=cfg["data"].get("num_classes", 6),
        pretrained_backbone=cfg["model"].get("pretrained", True),
        trainable_backbone_layers=cfg["model"].get("trainable_backbone_layers", 3)
    )
    model.to(device)

    # Optimizer & Scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=cfg["training"].get("lr", 0.001),
        momentum=cfg["training"].get("momentum", 0.9),
        weight_decay=cfg["training"].get("weight_decay", 0.0005)
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg["training"].get("lr_step_size", 4),
        gamma=cfg["training"].get("lr_gamma", 0.1)
    )

    checkpoint_dir = Path(cfg["training"].get("checkpoint_dir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    epochs = cfg["training"].get("epochs", 5)
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, optimizer, train_loader, device, epoch,
            log_interval=cfg["training"].get("log_interval", 10)
        )
        val_loss = evaluate_loss(model, val_loader, device)
        lr_scheduler.step()

        print(f"Epoch {epoch}/{epochs} Summary -> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save latest
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg,
            "val_loss": val_loss,
        }, checkpoint_dir / "latest_model.pt")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "config": cfg,
                "val_loss": val_loss,
            }, checkpoint_dir / "best_model.pt")
            print(f"[*] New best model saved with Val Loss: {best_val_loss:.4f}")

    print("Training complete!")


if __name__ == "__main__":
    main()
