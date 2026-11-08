"""
Submission validator for PanNuke Nuclei Detection Challenge.
Ensures participant submission files strictly adhere to required schema, coordinate constraints,
and category ranges before official evaluation.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple, List, Dict, Any


VALID_CATEGORY_IDS = {1, 2, 3, 4, 5}


def validate_submission_format(submission_data: Any) -> Tuple[bool, List[str]]:
    """
    Validate that the submission object conforms to the official specification.

    Returns:
        (is_valid, list_of_error_messages)
    """
    errors = []

    if not isinstance(submission_data, list):
        return False, ["Submission root must be a JSON array (list of detection objects)."]

    if len(submission_data) == 0:
        return False, ["Submission array is empty; no detections provided."]

    max_errors_to_report = 15
    for idx, item in enumerate(submission_data):
        if len(errors) >= max_errors_to_report:
            errors.append(f"... and more errors suppressed (stopped after {max_errors_to_report} issues).")
            break

        if not isinstance(item, dict):
            errors.append(f"Item #{idx}: Expected dict, received {type(item).__name__}.")
            continue

        # Check required keys
        required_keys = {"image_id", "category_id", "bbox", "score"}
        missing_keys = required_keys - set(item.keys())
        if missing_keys:
            errors.append(f"Item #{idx}: Missing required fields: {missing_keys}")
            continue

        # Check image_id
        if not isinstance(item["image_id"], int):
            errors.append(f"Item #{idx}: 'image_id' must be an integer, got {item['image_id']}")

        # Check category_id
        cat_id = item["category_id"]
        if not isinstance(cat_id, int) or cat_id not in VALID_CATEGORY_IDS:
            errors.append(f"Item #{idx}: 'category_id' must be an integer in {sorted(list(VALID_CATEGORY_IDS))}, got {cat_id}")

        # Check score
        score = item["score"]
        if not isinstance(score, (int, float)) or not (0.0 <= score <= 1.0):
            errors.append(f"Item #{idx}: 'score' must be a float in range [0.0, 1.0], got {score}")

        # Check bbox
        bbox = item["bbox"]
        if not isinstance(bbox, list) or len(bbox) != 4:
            errors.append(f"Item #{idx}: 'bbox' must be a list of 4 numbers [x, y, w, h], got {bbox}")
            continue

        if not all(isinstance(v, (int, float)) for v in bbox):
            errors.append(f"Item #{idx}: All 'bbox' elements must be numeric, got {bbox}")
            continue

        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            errors.append(f"Item #{idx}: Bounding box width and height must be > 0, got [w={w}, h={h}]")

        if x < -10 or y < -10 or x > 270 or y > 270:
            errors.append(f"Item #{idx}: Bounding box coordinates out of reasonable patch bounds: {bbox}")

    is_valid = len(errors) == 0
    return is_valid, errors


def main():
    parser = argparse.ArgumentParser(description="Validate PanNuke Detection submission JSON file")
    parser.add_argument("--submission", type=str, required=True, help="Path to submission JSON file")
    args = parser.parse_args()

    sub_path = Path(args.submission)
    if not sub_path.exists():
        print(f"[ERROR] Submission file not found: {sub_path}")
        sys.exit(1)

    try:
        with open(sub_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[ERROR] Failed to parse JSON: {e}")
        sys.exit(1)

    is_valid, errors = validate_submission_format(data)

    if is_valid:
        print(f"[PASS] Submission '{sub_path}' is valid! Total detections: {len(data)}")
        sys.exit(0)
    else:
        print(f"[FAIL] Submission '{sub_path}' failed schema validation with {len(errors)} error(s):")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
