"""
Official Auto-Grader Evaluation Script for PanNuke Challenge.
Computes primary mAP@[0.50:0.95] score and outputs SCORE=<score> for GitHub Actions.
Usage: python competition/evaluate.py <submission_csv> <ground_truth_csv_or_json>
"""

import sys
import os
import json
from pathlib import Path
import pandas as pd
import numpy as np

# Enable UTF-8 console output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add src and evaluation for metrics
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evaluation"))

from metrics import compute_coco_metrics


def main():
    if len(sys.argv) < 3:
        print("❌ Error: Missing arguments. Usage: python evaluate.py <submission_csv> <truth_file>")
        sys.exit(1)

    sub_path = sys.argv[1]
    truth_path = sys.argv[2]

    print(f"📂 Grading NuPheno submission: {sub_path}")
    print(f"📂 Against hidden truth: {truth_path}")

    try:
        # Load submission CSV
        sub_df = pd.read_csv(sub_path)
        predictions = []
        for _, row in sub_df.iterrows():
            predictions.append({
                "image_id": int(row["image_id"]),
                "category_id": int(row["category_id"]),
                "bbox": [float(row["x"]), float(row["y"]), float(row["width"]), float(row["height"])],
                "score": float(row["score"])
            })

        # Load ground truth (JSON COCO or hidden labels CSV)
        if truth_path.endswith(".json"):
            with open(truth_path, "r", encoding="utf-8") as f:
                gt_coco = json.load(f)
        else:
            # Reconstruct COCO dictionary from hidden CSV
            gt_df = pd.read_csv(truth_path)
            images_meta = [{"id": int(img_id)} for img_id in gt_df["image_id"].unique()]
            annotations = []
            for idx, row in gt_df.iterrows():
                annotations.append({
                    "id": idx + 1,
                    "image_id": int(row["image_id"]),
                    "category_id": int(row["category_id"]),
                    "bbox": [float(row["x"]), float(row["y"]), float(row["width"]), float(row["height"])],
                    "area": float(row["width"] * row["height"]),
                    "iscrowd": 0
                })
            gt_coco = {
                "images": images_meta,
                "annotations": annotations,
                "categories": [
                    {"id": 1, "name": "Neoplastic"},
                    {"id": 2, "name": "Inflammatory"},
                    {"id": 3, "name": "Connective"},
                    {"id": 4, "name": "Dead"},
                    {"id": 5, "name": "Epithelial"},
                ]
            }

        # Compute metrics
        results = compute_coco_metrics(gt_coco, predictions)
        score = results["mAP_50_95"]

        # Output score for GitHub Actions regex extraction
        print(f"SCORE={score:.4f}")
        print(f"mAP@50={results['mAP_50']:.4f}")

    except Exception as e:
        print(f"❌ Grading Failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
