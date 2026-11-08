"""
Utility functions for PanNuke nuclei detection:
Color palettes, bounding-box coordinate conversions, and visualization tools.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


CATEGORY_NAMES = {
    1: "Neoplastic",
    2: "Inflammatory",
    3: "Connective",
    4: "Dead",
    5: "Epithelial",
}

CATEGORY_COLORS = {
    1: "#E41A1C",  # Red - Neoplastic
    2: "#4DAF4A",  # Green - Inflammatory
    3: "#377EB8",  # Blue - Connective
    4: "#999999",  # Gray - Dead
    5: "#984EA3",  # Magenta/Purple - Epithelial
}


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert [x, y, w, h] to [x1, y1, x2, y2]."""
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    boxes = np.array(boxes, dtype=np.float32)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    return np.column_stack([x1, y1, x2, y2])


def xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    """Convert [x1, y1, x2, y2] to [x, y, w, h]."""
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    boxes = np.array(boxes, dtype=np.float32)
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    return np.column_stack([x, y, w, h])


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """
    Compute Intersection over Union (IoU) of two boxes in [x1, y1, x2, y2] format.
    """
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])

    intersection = max(0.0, xb - xa) * max(0.0, yb - ya)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    return float(intersection / union) if union > 0 else 0.0


def visualize_detections(
    image: Image.Image,
    boxes: List[List[float]],
    labels: List[int],
    scores: Optional[List[float]] = None,
    box_format: str = "xyxy",
    title: str = "PanNuke Nuclei Detections",
    figsize: Tuple[int, int] = (8, 8),
    save_path: Optional[str] = None
):
    """
    Visualize an H&E histology patch with bounding boxes colored by cell phenotype.
    """
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(image)

    for i, (box, label) in enumerate(zip(boxes, labels)):
        if box_format == "xywh":
            x, y, w, h = box
        else:
            x1, y1, x2, y2 = box
            x, y, w, h = x1, y1, x2 - x1, y2 - y1

        cat_name = CATEGORY_NAMES.get(label, f"Class {label}")
        color = CATEGORY_COLORS.get(label, "#FFFF00")

        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=2,
            edgecolor=color,
            facecolor="none"
        )
        ax.add_patch(rect)

        caption = cat_name
        if scores is not None and i < len(scores):
            caption += f" {scores[i]:.2f}"

        ax.text(
            x, max(0, y - 3),
            caption,
            bbox=dict(facecolor=color, alpha=0.6, pad=1, edgecolor="none"),
            fontsize=8,
            color="white",
            weight="bold"
        )

    ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()
    else:
        plt.show()
