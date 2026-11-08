"""
PyTorch Dataset for PanNuke COCO-format nuclei detection.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class PanNukeCOCODataset(Dataset):
    """
    Loads images and annotations from a COCO-style JSON file for PanNuke detection.
    Boxes are converted from [x_min, y_min, width, height] to [x_min, y_min, x_max, y_max].
    """

    def __init__(
        self,
        images_dir: str,
        annotations_file: str,
        transforms=None,
        is_test: bool = False
    ):
        super().__init__()
        self.images_dir = Path(images_dir)
        self.transforms = transforms
        self.is_test = is_test

        with open(annotations_file, "r", encoding="utf-8") as f:
            self.coco_data = json.load(f)

        self.images = self.coco_data.get("images", [])
        self.categories = {cat["id"]: cat["name"] for cat in self.coco_data.get("categories", [])}

        # Index annotations by image_id
        self.image_to_anns: Dict[int, List[Dict[str, Any]]] = {}
        for ann in self.coco_data.get("annotations", []):
            img_id = ann["image_id"]
            if img_id not in self.image_to_anns:
                self.image_to_anns[img_id] = []
            self.image_to_anns[img_id].append(ann)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img_info = self.images[idx]
        img_id = img_info["id"]
        img_path = self.images_dir / img_info["file_name"]

        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert("RGB")
        img_w, img_h = image.size

        # Convert to Tensor (normalized [0, 1])
        img_tensor = torch.from_numpy(np.array(image, dtype=np.float32) / 255.0).permute(2, 0, 1)

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        anns = self.image_to_anns.get(img_id, [])
        for ann in anns:
            x, y, w, h = ann["bbox"]
            # Convert to [x1, y1, x2, y2]
            x1 = max(0.0, float(x))
            y1 = max(0.0, float(y))
            x2 = min(float(img_w), float(x + w))
            y2 = min(float(img_h), float(y + h))

            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(int(ann["category_id"]))
                areas.append(float(ann.get("area", w * h)))
                iscrowd.append(int(ann.get("iscrowd", 0)))

        target = {}
        if len(boxes) > 0:
            target["boxes"] = torch.tensor(boxes, dtype=torch.float32)
            target["labels"] = torch.tensor(labels, dtype=torch.int64)
            target["area"] = torch.tensor(areas, dtype=torch.float32)
            target["iscrowd"] = torch.tensor(iscrowd, dtype=torch.int64)
        else:
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)
            target["area"] = torch.zeros((0,), dtype=torch.float32)
            target["iscrowd"] = torch.zeros((0,), dtype=torch.int64)

        target["image_id"] = torch.tensor([img_id], dtype=torch.int64)

        return img_tensor, target

    @staticmethod
    def collate_fn(batch):
        """Custom collate function for detection batches."""
        images = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        return torch.stack(images, dim=0), targets
