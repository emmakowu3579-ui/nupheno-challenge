"""
Converts official PanNuke .npy dataset files into standard COCO object detection format.
Extracts individual 256x256 image patches and computes bounding boxes from instance masks.
"""

import argparse
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import find_objects


CATEGORIES = [
    {"id": 1, "name": "Neoplastic", "supercategory": "cell"},
    {"id": 2, "name": "Inflammatory", "supercategory": "cell"},
    {"id": 3, "name": "Connective", "supercategory": "cell"},
    {"id": 4, "name": "Dead", "supercategory": "cell"},
    {"id": 5, "name": "Epithelial", "supercategory": "cell"},
]


def process_fold(fold_dir: Path, output_images_dir: Path, fold_name: str, start_img_id: int):
    """
    Process a single PanNuke fold containing images.npy, masks.npy, types.npy.
    """
    images_file = fold_dir / "images.npy"
    masks_file = fold_dir / "masks.npy"
    types_file = fold_dir / "types.npy"

    if not images_file.exists() or not masks_file.exists():
        raise FileNotFoundError(f"Missing images.npy or masks.npy in {fold_dir}")

    images = np.load(images_file)  # (N, 256, 256, 3)
    masks = np.load(masks_file)    # (N, 256, 256, 5 or 6)
    types = np.load(types_file) if types_file.exists() else None

    print(f"Loading {fold_name}: {images.shape[0]} images, masks shape: {masks.shape}")

    images_meta = []
    annotations = []
    ann_id = 1
    current_img_id = start_img_id

    for idx in range(images.shape[0]):
        img_arr = images[idx].astype(np.uint8)
        img_filename = f"{fold_name}_patch_{idx:05d}.png"
        img_path = output_images_dir / img_filename
        
        # Save image as PNG
        Image.fromarray(img_arr).save(img_path)

        tissue_type = str(types[idx]) if types is not None else "unknown"

        images_meta.append({
            "id": current_img_id,
            "file_name": img_filename,
            "width": 256,
            "height": 256,
            "tissue_type": tissue_type,
            "fold": fold_name
        })

        mask_tensor = masks[idx]  # (256, 256, channels)
        num_channels = min(mask_tensor.shape[-1], 5)

        for cat_channel in range(num_channels):
            category_id = cat_channel + 1  # 1 to 5
            channel_mask = mask_tensor[:, :, cat_channel].astype(np.int32)

            if channel_mask.max() == 0:
                continue

            # In PanNuke, channel_mask values are instance IDs
            instance_ids = np.unique(channel_mask)
            instance_ids = instance_ids[instance_ids > 0]

            for inst_id in instance_ids:
                bin_mask = (channel_mask == inst_id)
                y_indices, x_indices = np.where(bin_mask)
                if len(y_indices) == 0:
                    continue

                x_min = int(np.min(x_indices))
                x_max = int(np.max(x_indices))
                y_min = int(np.min(y_indices))
                y_max = int(np.max(y_indices))

                width = int(x_max - x_min + 1)
                height = int(y_max - y_min + 1)
                area = int(np.sum(bin_mask))

                # Ignore tiny 1-2 pixel artifacts
                if width < 3 or height < 3 or area < 6:
                    continue

                annotations.append({
                    "id": (current_img_id * 10000) + ann_id,
                    "image_id": current_img_id,
                    "category_id": category_id,
                    "bbox": [float(x_min), float(y_min), float(width), float(height)],
                    "area": float(area),
                    "iscrowd": 0
                })
                ann_id += 1

        current_img_id += 1

    return images_meta, annotations, current_img_id


def convert_pannuke_to_coco(raw_dir: str, output_dir: str):
    """
    Convert PanNuke Fold 1, Fold 2, Fold 3 into train, val, and test COCO splits.
    Standard convention:
      - Train: Fold 1
      - Validation: Fold 2
      - Test: Fold 3
    """
    raw_path = Path(raw_dir)
    out_path = Path(output_dir)
    images_dir = out_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    fold_map = {
        "Fold 1": "train",
        "Fold 2": "val",
        "Fold 3": "test"
    }

    start_img_id = 1
    for fold_name, split_name in fold_map.items():
        fold_dir = raw_path / fold_name
        if not fold_dir.exists():
            print(f"Warning: {fold_dir} not found. Skipping {fold_name}.")
            continue

        images_meta, annotations, start_img_id = process_fold(fold_dir, images_dir, fold_name, start_img_id)

        coco_data = {
            "info": {
                "description": f"PanNuke {split_name} split in COCO format",
                "version": "1.0",
                "year": 2026
            },
            "licenses": [],
            "images": images_meta,
            "annotations": annotations,
            "categories": CATEGORIES
        }

        ann_file = out_path / f"annotations_{split_name}.json"
        with open(ann_file, "w", encoding="utf-8") as f:
            json.dump(coco_data, f, indent=2)

        print(f"Successfully generated {split_name} split: {len(images_meta)} images, {len(annotations)} annotations.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PanNuke .npy to COCO format")
    parser.add_argument("--raw-dir", type=str, default="data/pannuke_raw", help="Path to raw PanNuke folds")
    parser.add_argument("--output-dir", type=str, default="data/processed_coco", help="Output directory for COCO dataset")
    args = parser.parse_args()

    convert_pannuke_to_coco(args.raw_dir, args.output_dir)
