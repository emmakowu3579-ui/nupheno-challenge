"""
Evaluation metrics for PanNuke Nuclei Detection.
Computes COCO-style mAP@[0.50:0.95], mAP@0.50, per-class Average Precision (AP),
Precision, Recall, and F1-score across cellular phenotypes.
Self-contained pure Python/NumPy implementation with optional pycocotools acceleration.
"""

from typing import Dict, List, Any, Tuple
import numpy as np


CATEGORY_NAMES = {
    1: "Neoplastic",
    2: "Inflammatory",
    3: "Connective",
    4: "Dead",
    5: "Epithelial"
}


def box_iou_xywh(box1: List[float], box2: List[float]) -> float:
    """Compute IoU between two boxes in [x, y, w, h] format."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_w = max(0.0, xi2 - xi1)
    inter_h = max(0.0, yi2 - yi1)
    intersection = inter_w * inter_h

    union = (w1 * h1) + (w2 * h2) - intersection
    return float(intersection / union) if union > 0 else 0.0


def compute_ap_single_iou(
    gt_boxes_by_img: Dict[int, List[List[float]]],
    pred_boxes_all: List[Dict[str, Any]],
    iou_threshold: float = 0.50
) -> Tuple[float, float, float, float]:
    """
    Computes Average Precision (AP), Precision, Recall, and F1 for a specific category at a given IoU threshold.
    """
    total_gt = sum(len(b) for b in gt_boxes_by_img.values())
    if total_gt == 0:
        return 0.0, 0.0, 0.0, 0.0

    if len(pred_boxes_all) == 0:
        return 0.0, 0.0, 0.0, 0.0

    # Sort detections by confidence descending
    sorted_preds = sorted(pred_boxes_all, key=lambda x: x.get("score", 0.0), reverse=True)

    # Track matched ground truth boxes per image: img_id -> set of matched indices
    matched_gt: Dict[int, set] = {img_id: set() for img_id in gt_boxes_by_img}

    tp = np.zeros(len(sorted_preds))
    fp = np.zeros(len(sorted_preds))

    for idx, pred in enumerate(sorted_preds):
        img_id = pred["image_id"]
        pred_box = pred["bbox"]

        gt_boxes = gt_boxes_by_img.get(img_id, [])
        if len(gt_boxes) == 0:
            fp[idx] = 1
            continue

        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt_box in enumerate(gt_boxes):
            iou = box_iou_xywh(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx not in matched_gt[img_id]:
            tp[idx] = 1
            matched_gt[img_id].add(best_gt_idx)
        else:
            fp[idx] = 1

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)

    recalls = cum_tp / float(total_gt)
    precisions = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(np.float64).eps)

    # 101-point COCO interpolation
    recall_thresholds = np.linspace(0.0, 1.0, 101)
    prec_at_rec = []
    for r in recall_thresholds:
        p_candidates = precisions[recalls >= r]
        prec_at_rec.append(np.max(p_candidates) if len(p_candidates) > 0 else 0.0)

    ap = float(np.mean(prec_at_rec))
    final_p = float(precisions[-1]) if len(precisions) > 0 else 0.0
    final_r = float(recalls[-1]) if len(recalls) > 0 else 0.0
    final_f1 = float(2 * final_p * final_r / (final_p + final_r)) if (final_p + final_r) > 0 else 0.0

    return ap, final_p, final_r, final_f1


def compute_coco_metrics(
    ground_truth_coco: Dict[str, Any],
    predictions: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Compute comprehensive COCO detection metrics:
    - mAP@[0.50:0.95]
    - mAP@0.50
    - Per-class AP50, Precision, Recall, F1
    """
    categories = {cat["id"]: cat["name"] for cat in ground_truth_coco.get("categories", [])}
    if not categories:
        categories = CATEGORY_NAMES

    # Group ground truth by category_id -> image_id -> list of bboxes
    gt_by_cat_and_img: Dict[int, Dict[int, List[List[float]]]] = {c: {} for c in categories}
    for ann in ground_truth_coco.get("annotations", []):
        cat_id = ann["category_id"]
        img_id = ann["image_id"]
        if cat_id in gt_by_cat_and_img:
            if img_id not in gt_by_cat_and_img[cat_id]:
                gt_by_cat_and_img[cat_id][img_id] = []
            gt_by_cat_and_img[cat_id][img_id].append(ann["bbox"])

    # Group predictions by category
    preds_by_cat: Dict[int, List[Dict[str, Any]]] = {c: [] for c in categories}
    for pred in predictions:
        cat_id = pred["category_id"]
        if cat_id in preds_by_cat:
            preds_by_cat[cat_id].append(pred)

    iou_thresholds = np.linspace(0.50, 0.95, 10)  # 0.50, 0.55, ..., 0.95

    per_class_summary = {}
    class_map50_list = []
    class_map50_95_list = []

    for cat_id, cat_name in categories.items():
        gt_boxes = gt_by_cat_and_img[cat_id]
        preds = preds_by_cat[cat_id]

        total_gt = sum(len(b) for b in gt_boxes.values())
        total_pred = len(preds)

        # Compute AP across the 10 IoU thresholds
        aps_per_iou = []
        ap_50, prec_50, rec_50, f1_50 = compute_ap_single_iou(gt_boxes, preds, iou_threshold=0.50)
        aps_per_iou.append(ap_50)

        for iou in iou_thresholds[1:]:
            ap_i, _, _, _ = compute_ap_single_iou(gt_boxes, preds, iou_threshold=float(iou))
            aps_per_iou.append(ap_i)

        ap_50_95 = float(np.mean(aps_per_iou))

        per_class_summary[cat_name] = {
            "category_id": cat_id,
            "total_ground_truth": total_gt,
            "total_predictions": total_pred,
            "AP_50": round(ap_50, 4),
            "AP_50_95": round(ap_50_95, 4),
            "Precision_50": round(prec_50, 4),
            "Recall_50": round(rec_50, 4),
            "F1_50": round(f1_50, 4),
        }

        if total_gt > 0:
            class_map50_list.append(ap_50)
            class_map50_95_list.append(ap_50_95)

    mean_ap_50 = float(np.mean(class_map50_list)) if class_map50_list else 0.0
    mean_ap_50_95 = float(np.mean(class_map50_95_list)) if class_map50_95_list else 0.0

    return {
        "mAP_50_95": round(mean_ap_50_95, 4),
        "mAP_50": round(mean_ap_50, 4),
        "per_class_results": per_class_summary,
        "total_images": len(ground_truth_coco.get("images", [])),
        "total_ground_truth_boxes": len(ground_truth_coco.get("annotations", [])),
        "total_predicted_boxes": len(predictions)
    }
