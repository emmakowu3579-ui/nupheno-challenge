"""
Inference and submission generation script for PanNuke Detection.
Takes a trained checkpoint, evaluates test images, and outputs a valid submission JSON file.
"""

import argparse
import json
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from dataset import PanNukeCOCODataset
from model import build_faster_rcnn_model


def collate_fn(batch):
    return tuple(zip(*batch))


@torch.no_grad()
def run_inference(
    model,
    data_loader,
    device,
    conf_threshold: float = 0.3
):
    """
    Run inference across the test data loader and return list of COCO-style detection predictions.
    """
    model.eval()
    predictions = []

    for images, targets in tqdm(data_loader, desc="Generating Predictions"):
        images = list(img.to(device) for img in images)
        outputs = model(images)

        for target, output in zip(targets, outputs):
            img_id = int(target["image_id"].item())

            boxes = output["boxes"].cpu().numpy()
            scores = output["scores"].cpu().numpy()
            labels = output["labels"].cpu().numpy()

            keep_idx = scores >= conf_threshold

            for box, score, label in zip(boxes[keep_idx], scores[keep_idx], labels[keep_idx]):
                x1, y1, x2, y2 = box
                width = float(x2 - x1)
                height = float(y2 - y1)

                predictions.append({
                    "image_id": img_id,
                    "category_id": int(label),
                    "bbox": [
                        round(float(x1), 2),
                        round(float(y1), 2),
                        round(float(width), 2),
                        round(float(height), 2)
                    ],
                    "score": round(float(score), 4)
                })

    return predictions


def main():
    parser = argparse.ArgumentParser(description="Generate PanNuke Detection Submission")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to checkpoint")
    parser.add_argument("--config", type=str, default="configs/baseline_config.yaml", help="Path to config file")
    parser.add_argument("--output-json", type=str, default="sample_submission/my_submission.json", help="Path to save output JSON")
    parser.add_argument("--conf-threshold", type=float, default=0.3, help="Confidence score threshold")
    parser.add_argument("--test-annotations", type=str, default=None, help="Override test annotations file")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        fallback_cfg = Path(__file__).resolve().parent.parent / "configs" / "baseline_config.yaml"
        if fallback_cfg.exists():
            cfg_path = fallback_cfg

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset
    test_ann = args.test_annotations or cfg["data"].get("test_annotations", "data/mock_coco/annotations_test.json")
    test_dir = cfg["data"].get("test_images_dir", "data/mock_coco/images")

    test_dataset = PanNukeCOCODataset(
        images_dir=test_dir,
        annotations_file=test_ann,
        is_test=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn
    )

    # Model
    model = build_faster_rcnn_model(
        num_classes=cfg["data"].get("num_classes", 6),
        pretrained_backbone=False
    )

    checkpoint_path = Path(args.checkpoint)
    if checkpoint_path.exists():
        print(f"Loading checkpoint weights from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print(f"Notice: Checkpoint {checkpoint_path} not found. Running with initialized weights.")

    model.to(device)

    # Inference
    preds = run_inference(model, test_loader, device, conf_threshold=args.conf_threshold)

    # Save output
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(preds, f, indent=2)

    print(f"Successfully generated {len(preds)} predictions saved to {output_path}")


if __name__ == "__main__":
    main()
