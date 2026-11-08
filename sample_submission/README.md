# Sample Submission (PanNuke Detection)

This directory contains a valid schema-compliant sample submission file (`sample_submission.json`) illustrating the exact structure expected by the benchmark evaluation system.

## Schema Format:
```json
[
  {
    "image_id": 56,
    "category_id": 1,
    "bbox": [112.5, 84.0, 20.4, 18.2],
    "score": 0.941
  }
]
```

## Field Constraints:
- `image_id`: Integer corresponding to an image `id` in the test set.
- `category_id`: Integer in range `[1, 5]`:
  - 1: Neoplastic
  - 2: Inflammatory
  - 3: Connective
  - 4: Dead
  - 5: Epithelial
- `bbox`: Array of 4 floats `[x_min, y_min, width, height]` in pixel coordinates.
- `score`: Confidence float in range `[0.0, 1.0]`.
