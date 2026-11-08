"""
Submission Validator for PanNuke Challenge Auto-Grader.
Enforces schema requirements, bounding-box constraints, and one-submission-per-team policy.
Usage: python validate_submission.py <submission_csv> <test_images_csv> <leaderboard_csv>
"""

import pandas as pd
import sys
import os
from pathlib import Path

# Enable UTF-8 console output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def main(submission_path: str, test_images_path: str, leaderboard_path: str):
    team_name = os.path.splitext(os.path.basename(submission_path))[0]
    print(f"🔍 Validating submission for Team: {team_name}")

    # 1. Team duplicate check
    if os.path.exists(leaderboard_path):
        try:
            lb_df = pd.read_csv(leaderboard_path)
            existing_teams = set(lb_df["team"].astype(str).str.lower())
            if team_name.lower() in existing_teams:
                print(f"❌ REJECTED: Team '{team_name}' has already submitted.")
                print("   Per competition rules, only one official submission is permitted.")
                sys.exit(1)
        except Exception as e:
            print(f"⚠️ Warning: Could not read leaderboard (Skipping check): {e}")

    # 2. Load submission & test reference
    try:
        preds = pd.read_csv(submission_path)
        test_images = pd.read_csv(test_images_path)
    except Exception as e:
        print(f"❌ Error: Could not read CSV files: {e}")
        sys.exit(1)

    # 3. Required columns check
    required_cols = {"image_id", "category_id", "x", "y", "width", "height", "score"}
    missing = required_cols - set(preds.columns)
    if missing:
        print(f"❌ Error: CSV headers must include {required_cols}. Missing: {missing}")
        sys.exit(1)

    # 4. Check image_ids exist in test set
    test_image_ids = set(test_images["image_id"])
    pred_image_ids = set(preds["image_id"])
    invalid_ids = pred_image_ids - test_image_ids
    if invalid_ids:
        print(f"❌ Error: Found {len(invalid_ids)} image_ids not present in test set: {list(invalid_ids)[:5]}")
        sys.exit(1)

    # 5. Value checks
    if not pd.api.types.is_numeric_dtype(preds["category_id"]):
        print("❌ Error: 'category_id' must be numeric.")
        sys.exit(1)

    valid_cats = {1, 2, 3, 4, 5}
    if not preds["category_id"].isin(valid_cats).all():
        print(f"❌ Error: Invalid 'category_id' values. Allowed: {valid_cats}")
        sys.exit(1)

    if (preds["width"] <= 0).any() or (preds["height"] <= 0).any():
        print("❌ Error: Box width and height must be positive numbers.")
        sys.exit(1)

    if not ((preds["score"] >= 0.0) & (preds["score"] <= 1.0)).all():
        print("❌ Error: Confidence 'score' must be in range [0.0, 1.0].")
        sys.exit(1)

    print(f"✅ VALID SUBMISSION for Team: {team_name} (Total detections: {len(preds)})")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python validate_submission.py <submission_csv> <test_images_csv> <leaderboard_csv>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
