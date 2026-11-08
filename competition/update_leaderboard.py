"""
Updates leaderboard/leaderboard.csv from all submissions for PanNuke Challenge.
"""

import pandas as pd
import os
import glob
import subprocess
import json
from pathlib import Path

SUBMISSION_DIR = "submissions"
TRUTH_FILE = "data/test_labels_hidden.csv"
LEADERBOARD_CSV = "leaderboard/leaderboard.csv"

# Add evaluation path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evaluation"))
from metrics import compute_coco_metrics


def main():
    if not os.path.exists(TRUTH_FILE):
        print(f"❌ Error: Hidden truth file {TRUTH_FILE} not found.")
        return

    # Load truth
    gt_df = pd.read_csv(TRUTH_FILE)
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

    files = glob.glob(f"{SUBMISSION_DIR}/**/*.csv", recursive=True)
    results = []
    print(f"🔍 Found {len(files)} submission CSV files.")

    for file_path in files:
        if "sample_submission.csv" in file_path:
            continue
        try:
            pred_df = pd.read_csv(file_path)
            predictions = []
            for _, row in pred_df.iterrows():
                predictions.append({
                    "image_id": int(row["image_id"]),
                    "category_id": int(row["category_id"]),
                    "bbox": [float(row["x"]), float(row["y"]), float(row["width"]), float(row["height"])],
                    "score": float(row["score"])
                })

            eval_res = compute_coco_metrics(gt_coco, predictions)
            score = eval_res["mAP_50_95"]

            filename = os.path.basename(file_path)
            team_name = os.path.splitext(filename)[0]

            # Git commit date check
            enc_file = file_path + ".enc"
            commit_date = ""
            if os.path.exists(enc_file):
                try:
                    cmd = f'git log -1 --format="%cd" --date=format:"%Y-%m-%d %H:%M" -- "{enc_file}"'
                    commit_date = subprocess.check_output(cmd, shell=True).decode("utf-8").strip()
                except Exception:
                    pass

            if not commit_date:
                commit_date = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")

            print(f"✅ Scored {team_name}: mAP={score:.4f}")
            results.append({
                "team": team_name,
                "score": round(score, 4),
                "date": commit_date
            })

        except Exception as e:
            print(f"⚠️ Failed to process {file_path}: {e}")

    if not results:
        print("No valid submissions found.")
        return

    df = pd.DataFrame(results)
    df = df.sort_values(by="score", ascending=False).drop_duplicates(subset=["team"], keep="first")

    os.makedirs("leaderboard", exist_ok=True)
    df.to_csv(LEADERBOARD_CSV, index=False)
    print(f"✅ Leaderboard CSV saved to {LEADERBOARD_CSV}")


if __name__ == "__main__":
    main()
