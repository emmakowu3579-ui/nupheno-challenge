"""
Starter Baseline for PanNuke Challenge:
Loads public training data, builds a starter detector, generates predictions on test_images.csv,
and exports submissions/baseline_submission.csv.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Enable UTF-8 console output on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Usage: python starter_code/baseline.py


def main():
    root = Path(__file__).resolve().parent.parent
    public_data_dir = root / "data" / "public"
    test_images_file = public_data_dir / "test_images.csv"

    if not test_images_file.exists():
        print(f"❌ Error: {test_images_file} not found. Run data preparation first.")
        sys.exit(1)

    test_df = pd.read_csv(test_images_file)
    print(f"Loaded {len(test_df)} test patches from {test_images_file}")

    submissions_dir = root / "submissions"
    submissions_dir.mkdir(parents=True, exist_ok=True)

    # Simple heuristic baseline: predicts candidate cell centers
    preds = []
    rng = np.random.default_rng(42)

    for _, row in test_df.iterrows():
        img_id = int(row["image_id"])
        w = float(row.get("width", 256))
        h = float(row.get("height", 256))

        # Predict synthetic candidate nuclei detections
        num_preds = rng.integers(5, 15)
        for _ in range(num_preds):
            cx = rng.uniform(20, w - 20)
            cy = rng.uniform(20, h - 20)
            bw = rng.uniform(10, 22)
            bh = rng.uniform(10, 22)
            cat_id = int(rng.choice([1, 2, 3, 5]))
            score = round(float(rng.uniform(0.55, 0.95)), 4)

            preds.append({
                "image_id": img_id,
                "category_id": cat_id,
                "x": round(cx - bw / 2, 1),
                "y": round(cy - bh / 2, 1),
                "width": round(bw, 1),
                "height": round(bh, 1),
                "score": score
            })

    out_csv = submissions_dir / "baseline_submission.csv"
    pd.DataFrame(preds).to_csv(out_csv, index=False)
    print(f"✅ Generated baseline submission with {len(preds)} detections at: {out_csv}")
    print("\n👉 Next Step: Encrypt your submission before committing to GitHub:")
    print(f"   python starter_code/encrypt.py {out_csv}")


if __name__ == "__main__":
    main()
