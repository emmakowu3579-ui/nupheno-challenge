"""
Unit tests for PanNuke evaluation metrics and submission validation.
"""

import importlib.util
from pathlib import Path
import pytest

eval_dir = Path(__file__).resolve().parent.parent / "evaluation"

spec_m = importlib.util.spec_from_file_location("pannuke_metrics_mod", eval_dir / "metrics.py")
p_metrics_mod = importlib.util.module_from_spec(spec_m)
spec_m.loader.exec_module(p_metrics_mod)
compute_coco_metrics = p_metrics_mod.compute_coco_metrics
box_iou_xywh = p_metrics_mod.box_iou_xywh

spec_v = importlib.util.spec_from_file_location("pannuke_val_mod", eval_dir / "validate_submission.py")
p_val_mod = importlib.util.module_from_spec(spec_v)
spec_v.loader.exec_module(p_val_mod)
validate_submission_format = p_val_mod.validate_submission_format


def test_box_iou():
    box_a = [10.0, 10.0, 20.0, 20.0]
    box_b = [10.0, 10.0, 20.0, 20.0]
    assert pytest.approx(box_iou_xywh(box_a, box_b), 0.001) == 1.0

    box_c = [50.0, 50.0, 10.0, 10.0]
    assert box_iou_xywh(box_a, box_c) == 0.0


def test_submission_validation():
    valid_sub = [
        {"image_id": 1, "category_id": 1, "bbox": [10.0, 10.0, 20.0, 20.0], "score": 0.95}
    ]
    is_valid, errors = validate_submission_format(valid_sub)
    assert is_valid
    assert len(errors) == 0

    invalid_sub = [
        {"image_id": 1, "category_id": 99, "bbox": [10.0, 10.0, -5.0, 20.0], "score": 1.5}
    ]
    is_valid_inv, errors_inv = validate_submission_format(invalid_sub)
    assert not is_valid_inv
    assert len(errors_inv) >= 2


def test_metric_computation():
    gt = {
        "images": [{"id": 1, "file_name": "patch_1.png", "width": 256, "height": 256}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [20.0, 20.0, 15.0, 15.0]},
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [100.0, 100.0, 10.0, 10.0]}
        ],
        "categories": [
            {"id": 1, "name": "Neoplastic"},
            {"id": 2, "name": "Inflammatory"},
            {"id": 3, "name": "Connective"},
            {"id": 4, "name": "Dead"},
            {"id": 5, "name": "Epithelial"}
        ]
    }

    # Perfect prediction for cat 1, slight offset for cat 2
    preds = [
        {"image_id": 1, "category_id": 1, "bbox": [20.0, 20.0, 15.0, 15.0], "score": 0.99},
        {"image_id": 1, "category_id": 2, "bbox": [101.0, 100.0, 10.0, 10.0], "score": 0.90}
    ]

    metrics = compute_coco_metrics(gt, preds)
    assert metrics["mAP_50"] > 0.9
    assert "Neoplastic" in metrics["per_class_results"]
    assert metrics["per_class_results"]["Neoplastic"]["AP_50"] == 1.0
