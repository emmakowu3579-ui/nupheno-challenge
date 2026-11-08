"""
Official evaluation script for PanNuke Nuclei Detection Challenge.
Validates submission, computes official mAP@[0.50:0.95] and class breakdown,
and outputs formatted leaderboard scoreboard.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from metrics import compute_coco_metrics
from validate_submission import validate_submission_format


def print_scorecard(results: dict):
    """Prints a formatted evaluation scorecard to standard output."""
    print("\n" + "=" * 78)
    print("      OFFICIAL EVALUATION SCOREBOARD: PANNUKE NUCLEI DETECTION        ")
    print("=" * 78)
    print(f" Total Evaluated Images     : {results['total_images']}")
    print(f" Ground Truth Nuclei Boxes  : {results['total_ground_truth_boxes']}")
    print(f" Submitted Predicted Boxes  : {results['total_predicted_boxes']}")
    print("-" * 78)
    print(f" PRIMARY RANK METRIC (mAP@[0.50:0.95]) : {results['mAP_50_95']:.4f}")
    print(f" SECONDARY METRIC   (mAP@0.50)         : {results['mAP_50']:.4f}")
    print("-" * 78)
    print(" PER-CLASS PHENOTYPE PERFORMANCE BREAKDOWN:")
    print(f" {'Class Name':<15} | {'GT Count':<8} | {'Preds':<8} | {'AP@50':<8} | {'AP50:95':<8} | {'F1@50':<8}")
    print("-" * 78)

    for cat_name, metrics in results["per_class_results"].items():
        print(f" {cat_name:<15} | {metrics['total_ground_truth']:<8} | {metrics['total_predictions']:<8} | {metrics['AP_50']:<8.4f} | {metrics['AP_50_95']:<8.4f} | {metrics['F1_50']:<8.4f}")

    print("=" * 78 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Official PanNuke Evaluation Script")
    parser.add_argument("--predictions", type=str, required=True, help="Path to participant submission JSON")
    parser.add_argument("--ground-truth", type=str, required=True, help="Path to ground truth COCO JSON")
    parser.add_argument("--output-json", type=str, default=None, help="Optional output JSON path for metrics")
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    gt_path = Path(args.ground_truth)

    if not pred_path.exists():
        print(f"[ERROR] Predictions file not found: {pred_path}")
        sys.exit(1)

    if not gt_path.exists():
        print(f"[ERROR] Ground truth file not found: {gt_path}")
        sys.exit(1)

    # Load files
    with open(pred_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    with open(gt_path, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    # 1. Validation check
    is_valid, errors = validate_submission_format(predictions)
    if not is_valid:
        print("[ERROR] Submission validation failed! Evaluation aborted.")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)

    # 2. Compute metrics
    results = compute_coco_metrics(ground_truth, predictions)

    # 3. Print scoreboard
    print_scorecard(results)

    # 4. Save results JSON if requested
    if args.output_json:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Metrics saved to {out_p}")


if __name__ == "__main__":
    main()
