"""
PanNuke Data Acquisition and Mock Generator.
Supports fetching official dataset instructions or generating synthetic mock H&E patches with COCO annotations
for testing, CI, and local rapid development.
"""

import argparse
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


CATEGORIES = [
    {"id": 1, "name": "Neoplastic", "supercategory": "cell"},
    {"id": 2, "name": "Inflammatory", "supercategory": "cell"},
    {"id": 3, "name": "Connective", "supercategory": "cell"},
    {"id": 4, "name": "Dead", "supercategory": "cell"},
    {"id": 5, "name": "Epithelial", "supercategory": "cell"},
]


def generate_mock_pannuke_patch(image_id: int, rng: np.random.Generator, size: int = 256):
    """
    Generate an H&E-like histology patch (pink/purple background) with multiple cell nuclei.
    Returns the PIL Image and a list of COCO annotation dicts.
    """
    # Base H&E tissue color (pinkish-purple stroma/cytoplasm)
    base_r = rng.integers(210, 245)
    base_g = rng.integers(180, 215)
    base_b = rng.integers(215, 245)
    img_arr = np.full((size, size, 3), [base_r, base_g, base_b], dtype=np.uint8)

    # Add subtle texture noise
    noise = rng.normal(0, 8, (size, size, 3)).astype(np.int16)
    img_arr = np.clip(img_arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    image = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(image)

    # Number of nuclei per patch
    num_nuclei = rng.integers(12, 35)
    annotations = []

    # Category colors in purple/blue hematoxylin stains
    cat_stains = {
        1: (80, 40, 110),   # Neoplastic: dark purple/blue, larger
        2: (50, 30, 90),    # Inflammatory: small, dense dark purple
        3: (110, 70, 130),  # Connective: elongated spindle
        4: (140, 120, 140), # Dead: pyknotic / fragmented gray-purple
        5: (90, 50, 120),   # Epithelial: clustered medium purple
    }

    for ann_idx in range(num_nuclei):
        category_id = int(rng.choice([1, 2, 3, 4, 5], p=[0.35, 0.25, 0.20, 0.05, 0.15]))
        
        # Dimensions based on category
        if category_id == 1:
            rx = rng.integers(8, 16)
            ry = rng.integers(8, 16)
        elif category_id == 2:
            rx = rng.integers(4, 7)
            ry = rng.integers(4, 7)
        elif category_id == 3:
            rx = rng.integers(10, 20)
            ry = rng.integers(4, 8)
        elif category_id == 4:
            rx = rng.integers(3, 6)
            ry = rng.integers(3, 6)
        else:
            rx = rng.integers(7, 13)
            ry = rng.integers(7, 13)

        cx = rng.integers(rx + 4, size - rx - 4)
        cy = rng.integers(ry + 4, size - ry - 4)

        x_min = cx - rx
        y_min = cy - ry
        x_max = cx + rx
        y_max = cy + ry
        width = x_max - x_min
        height = y_max - y_min

        stain = cat_stains[category_id]
        draw.ellipse([x_min, y_min, x_max, y_max], fill=stain)

        annotations.append({
            "id": (image_id * 1000) + ann_idx,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [float(x_min), float(y_min), float(width), float(height)],
            "area": float(width * height),
            "iscrowd": 0,
        })

    return image, annotations


def generate_mock_dataset(output_dir: str = "data/mock_coco", num_train: int = 40, num_val: int = 15, num_test: int = 15, seed: int = 42):
    """
    Generate mock train/val/test COCO datasets with simulated H&E pathology patches.
    """
    out_path = Path(output_dir)
    images_dir = out_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    splits = {
        "train": num_train,
        "val": num_val,
        "test": num_test
    }

    current_img_id = 1
    for split_name, count in splits.items():
        images_meta = []
        all_annotations = []

        for _ in range(count):
            img_filename = f"patch_{current_img_id:05d}.png"
            img_file_path = images_dir / img_filename
            
            img, anns = generate_mock_pannuke_patch(current_img_id, rng)
            img.save(img_file_path)

            images_meta.append({
                "id": current_img_id,
                "file_name": img_filename,
                "width": 256,
                "height": 256
            })
            all_annotations.extend(anns)
            current_img_id += 1

        coco_data = {
            "info": {
                "description": f"PanNuke Mock Dataset ({split_name})",
                "year": 2026,
                "version": "1.0",
            },
            "licenses": [],
            "images": images_meta,
            "annotations": all_annotations,
            "categories": CATEGORIES,
        }

        ann_path = out_path / f"annotations_{split_name}.json"
        with open(ann_path, "w", encoding="utf-8") as f:
            json.dump(coco_data, f, indent=2)

        print(f"Generated {split_name} split: {len(images_meta)} images, {len(all_annotations)} bounding boxes at {ann_path}")


def print_official_download_instructions():
    print("""
========================================================================
OFFICIAL PANNUKE DATASET DOWNLOAD INSTRUCTIONS:
========================================================================
1. Official Kaggle Dataset (Recommended for Kaggle Notebooks):
   - URL: https://www.kaggle.com/datasets/andrewmvd/cancer-instance-segmentation-and-classification-2
   - Kaggle CLI: kaggle datasets download -d andrewmvd/cancer-instance-segmentation-and-classification-2

2. Official Warwick TIA Centre & Zenodo Archives:
   - Warwick TIA: https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke
   - Zenodo: https://zenodo.org/records/3901832
   
   Download Fold 1, Fold 2, and Fold 3 (images.npy, masks.npy, types.npy)
   and place them in `data/pannuke_raw/`:
   data/pannuke_raw/
   ├── Fold 1/
   │   ├── images.npy
   │   ├── masks.npy
   │   └── types.npy
   ├── Fold 2/
   └── Fold 3/

3. Run conversion script:
   python data/prepare_coco_pannuke.py --raw-dir data/pannuke_raw/ --output-dir data/processed_coco/
========================================================================
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download or generate PanNuke data.")
    parser.add_argument("--generate-mock", action="store_true", help="Generate synthetic mock COCO dataset for testing")
    parser.add_argument("--download-official", action="store_true", help="Print official download instructions")
    parser.add_argument("--output-dir", type=str, default="data/mock_coco", help="Output directory for mock dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if args.download_official:
        print_official_download_instructions()
    else:
        # Default action is generating mock for seamless testing
        generate_mock_dataset(output_dir=args.output_dir, seed=args.seed)
